"""
Turns a MESA .data file of a massive star into a .h5 file of NCCs for a d3 run.

Usage:
    make_coreCZ_nccs.py [options]

Options:
    --Nmax=<N>      Maximum radial coefficients [default: 63]
    --file=<f>      Path to MESA log file [default: MESA_Models_Dedalus_Full_Sphere/LOGS/h1_0.6.data]
"""
import os
import time

import numpy as np
import h5py
from mpi4py import MPI
import mesa_reader as mr
import matplotlib.pyplot as plt
from dedalus.core import coords, distributor, basis, field, operators, arithmetic
import dedalus.public as de
from docopt import docopt

from astropy import units as u
from astropy import constants
from scipy.signal import savgol_filter
from numpy.polynomial import Chebyshev as Pfit

args = docopt(__doc__)

def plot_ncc_figure(r, mesa_y, dedalus_y, N, ylabel="", fig_name="", out_dir='.', zero_line=False):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    if zero_line:
        ax1.axhline(0, c='k', lw=0.5)

    ax1.plot(r, mesa_y, label='mesa', c='k', lw=3)
    ax1.plot(r, dedalus_y, label='dedalus', c='red')
    plt.legend(loc='best')
    ax1.set_xlabel('radius/L', labelpad=-3)
    ax1.set_ylabel('{}'.format(ylabel))
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')

    ax2 = fig.add_subplot(2,1,2)
    difference = np.abs(1 - dedalus_y/mesa_y)
    ax2.plot(r, np.abs(difference).flatten())
    ax2.set_ylabel('abs(1 - dedalus/mesa)')
    ax2.set_xlabel('radius/L')
    ax2.set_yscale('log')
    fig.suptitle('coeff bandwidth = {}'.format(N))
    fig.savefig('{:s}/{}.png'.format(out_dir, fig_name), bbox_inches='tight', dpi=200)


#def load_data(nr1, nr2, r_int, get_dimensions=False):
Nmax = int(args['--Nmax'])
read_file = args['--file']
out_dir  = read_file.replace('/LOGS/', '_').replace('.data', '_core')
out_file = '{:s}/nccs_{}.h5'.format(out_dir, Nmax)
if not os.path.exists('{:s}'.format(out_dir)):
    os.mkdir('{:s}'.format(out_dir))


p = mr.MesaData(read_file)


c = coords.SphericalCoordinates('φ', 'θ', 'r')
d = distributor.Distributor((c,), mesh=None)
b = basis.BallBasis(c, (1, 1, Nmax+1), radius=1, dtype=np.float64, dealias=(1, 1, 1))
φg, θg, rg = b.global_grids((1, 1, 1))

grad = lambda A: operators.Gradient(A, c)
dot  = lambda A, B: arithmetic.DotProduct(A, B)


mass = p.mass[::-1] * u.M_sun
mass = mass.to('g')

r = p.radius[::-1] * u.R_sun
r = r.to('cm')

rho = 10**p.logRho[::-1] * u.g / u.cm**3
P = 10**p.logP[::-1] * u.g / u.cm / u.s**2
eps = p.eps_nuc[::-1] * u.erg / u.g / u.s
nablaT = p.gradT[::-1] #dlnT/dlnP
T = 10**p.logT[::-1] * u.K
cp = p.cp[::-1]  * u.erg / u.K / u.g
cv = p.cv[::-1]  * u.erg / u.K / u.g
mu = p.mu[::-1]
N2 = p.brunt_N2[::-1] / u.s**2
Luminosity = p.luminosity[::-1] * u.L_sun
Luminosity = Luminosity.to('erg/s')
L_conv = p.conv_L_div_L[::-1]*Luminosity

cgs_G = constants.G.to('cm^3/(g*s^2)')
g = cgs_G*mass/r**2
gamma = cp/cv


#Find edge of core cz
cz_bool = (L_conv.value > 1)*(mass < 0.9*mass[-1])
core_cz_bound = 0.995*mass[cz_bool][-1] # 0.9 to avoid some of the cz->rz transition region.
bound_ind = np.argmin(np.abs(mass - core_cz_bound))


L  = r[bound_ind]
g0 = g[bound_ind] 
rho0 = rho[0]
P0 = P[0]
T0 = T[0]
cp0 = cp[0]
gamma0 = gamma[0]

r_cz = r[cz_bool]/L


r_vec  = field.Field(dist=d, bases=(b,), dtype=np.float64, tensorsig=(c,))
r_vec['g'][2,:] = 1

### Log Density
N = 10
frac = int((Nmax+1)/N)
φgf, θgf, rgf = b.global_grids((1, 1, 1/frac))
ln_rho_field  = field.Field(dist=d, bases=(b,), dtype=np.float64)
ln_rho = np.log(rho/rho0)[cz_bool]
deg = 10
ln_rho_fit = Pfit.fit(r_cz, ln_rho, deg)(r_cz)
ln_rho_interp = np.interp(rg, r_cz, ln_rho_fit)
ln_rho_interp2 = np.interp(rg, r_cz, ln_rho)
ln_rho_field['g'] = ln_rho_interp
ln_rho_field['c'][:, :, N:] = 0
#ln_rho_field.require_scales(1/frac)
#ln_rho_field['g'] = ln_rho_interp
#ln_rho_field['c']
plot_ncc_figure(rg.flatten(), (-1)+ln_rho_interp2.flatten(), (-1)+ln_rho_field['g'].flatten(), N, ylabel=r"$\ln\rho - 1$", fig_name="ln_rho", out_dir=out_dir)
ln_rho_field.require_scales(1)

grad_ln_rho_field  = field.Field(dist=d, bases=(b,), dtype=np.float64)
grad_ln_rho = np.gradient(ln_rho,r_cz)
grad_ln_rho_interp = np.interp(rg, r_cz, grad_ln_rho)
grad_ln_rho_field['g'] = dot(r_vec, grad(ln_rho_field)).evaluate()['g']
plot_ncc_figure(rg.flatten(), grad_ln_rho_interp.flatten(), grad_ln_rho_field['g'].flatten(), N, ylabel=r"$\nabla\ln\rho$", fig_name="grad_ln_rho", out_dir=out_dir)

### log (temp)
N = 30
ln_T_field  = field.Field(dist=d, bases=(b,), dtype=np.float64)
ln_T = np.log((T)[cz_bool]/T0)
deg = 10
ln_T_fit = Pfit.fit(r_cz, ln_T, deg)(r_cz)
ln_T_interp = np.interp(rg, r_cz, ln_T_fit)
ln_T_interp2 = np.interp(rg, r_cz, ln_T)
ln_T_field['g'] = ln_T_interp
ln_T_field['c'][:, :, N:] = 0
plot_ncc_figure(rg.flatten(), (-1)+ln_T_interp2.flatten(), (-1)+ln_T_field['g'].flatten(), N, ylabel=r"$\ln(T) - 1$", fig_name="ln_T", out_dir=out_dir)

grad_ln_T_field  = field.Field(dist=d, bases=(b,), dtype=np.float64)
grad_ln_T = np.gradient(ln_T,r_cz)
grad_ln_T_interp = np.interp(rg, r_cz, grad_ln_T)
grad_ln_T_field['g'] = dot(r_vec, grad(ln_T_field)).evaluate()['g']
grad_ln_T_field['c'][:, :, N:] = 0
plot_ncc_figure(rg.flatten(), grad_ln_T_interp.flatten(), grad_ln_T_field['g'].flatten(), N, ylabel=r"$\nabla\ln(T)$", fig_name="grad_ln_T", out_dir=out_dir)

#plt.show()




### inverse Temperature
N = 5
inv_T_field = field.Field(dist=d, bases=(b,), dtype=np.float64)
inv_T = (T0) / (T)[cz_bool]
inv_T_interp = np.interp(rg, r_cz, inv_T)
inv_T_field['g'] = inv_T_interp
inv_T_field['c'][:, :, N:] = 0
plot_ncc_figure(rg.flatten(), inv_T_interp.flatten(), inv_T_field['g'].flatten(), N, ylabel=r"$(T/[T_c])^{-1}$", fig_name="inv_T", out_dir=out_dir)

### effective heating / (rho * T)
N = 40
H = rho * eps
C = (np.gradient(Luminosity-L_conv,r)/(4*np.pi*r**2))
H_eff = H - C

#dr = np.gradient(r)
#dLum = 4*np.pi*r**2 * H_eff
#sumLum = np.zeros(dLum.shape)
#for i in range(len(sumLum)-1):
#    sumLum[i+1] = sumLum[i] + dLum[i].value*dr[i].value
#plt.figure()
#plt.plot(r/L, sumLum)
#plt.show()
H0 = H_eff[0]
H_NCC = ((H_eff / (rho*T)) * (rho0*T0) / H0)[cz_bool]
H_field = field.Field(dist=d, bases=(b,), dtype=np.float64)
H_interp = np.interp(rg, r_cz, H_NCC)
H_field['g'] = H_interp
H_field['c'][:, :, N:] = 0
plot_ncc_figure(rg.flatten(), H_interp.flatten(), H_field['g'].flatten(), N, ylabel=r"$(H_{eff}/(\rho c_p T))$ (nondimensional)", fig_name="H_eff", out_dir=out_dir, zero_line=True)


tau = (H0/L**2/rho0)**(-1/3)
tau = tau.cgs
print('one time unit is {:.2e}'.format(tau))
#pomegac = T0*R/mu[0]
u = L/tau

Ma2 = u**2 / ((gamma0-1)*cp0*T0)

#if get_dimensions:
#    return L, tau, Ma2

### Effective gravity
N = 40
g_eff = (g[cz_bool]/(cp[cz_bool]/cp0)) * (tau**2/L)*Ma2*(gamma0-1)
g_eff_field = field.Field(dist=d, bases=(b,), dtype=np.float64, tensorsig=(c,))
g_eff_interp = np.interp(rg, r_cz, g_eff)
g_eff_field['g'][2,:] = g_eff_interp
g_eff_field['c'][:, :, :, N:] = 0
plot_ncc_figure(rg.flatten(), g_eff_interp.flatten(), g_eff_field['g'][2].flatten(), N, ylabel=r"$g_{eff}$", fig_name="g_eff", out_dir=out_dir)



with h5py.File('{:s}'.format(out_file), 'w') as f:
    f['r']     = rg
    f['g_eff'] = g_eff_field['g']
    f['inv_T'] = inv_T_field['g']
    f['H_eff'] = H_field['g']
    f['ln_ρ']  = ln_rho_field['g'] 
    f['ln_T']  = ln_T_field['g']

    f['L']   = L
    f['g0']  = g0
    f['ρ0']  = rho0
    f['P0']  = P0
    f['T0']  = T0
    f['H0']  = H0
    f['tau'] = tau 
    f['Ma2'] = tau 

#plt.figure()
#Ta = np.exp(ln_T_field['g'])
#cpa = (1/inv_T_field['g'])/Ta
#plt.plot(r_cz, cp[cz_bool]/cp0)
#plt.plot(rg.flatten(), cpa.flatten())
#plt.show()
