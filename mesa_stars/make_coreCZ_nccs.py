import time

import numpy as np
import h5py
from mpi4py import MPI
import mesa_reader as mr
import matplotlib.pyplot as plt
from dedalus.core import coords, distributor, basis, field, operators
import dedalus.public as de

from astropy import units as u
from astropy import constants


def plot_ncc_figure(r, mesa_y, dedalus_y, N, ylabel="", fig_name=""):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)

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
    fig.savefig('{}.png'.format(fig_name), bbox_inches='tight', dpi=200)


#def load_data(nr1, nr2, r_int, get_dimensions=False):
read_file = "MESA_Models_Dedalus_Full_Sphere/LOGS/h1_0.6.data"
p = mr.MesaData(read_file)

Nmax = 127

c = coords.SphericalCoordinates('φ', 'θ', 'r')
d = distributor.Distributor((c,), mesh=None)
b = basis.BallBasis(c, (1, 1, Nmax+1), radius=1, dtype=np.float64, dealias=(1, 1, 1))
φg, θg, rg = b.global_grids((1, 1, 1))

grad = lambda A: operators.Gradient(A, c)


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
mu = p.mu[::-1]
N2 = p.brunt_N2[::-1] / u.s**2
Luminosity = p.luminosity[::-1] * u.L_sun
Luminosity = Luminosity.to('erg/s')
L_conv = p.conv_L_div_L[::-1]*Luminosity

cgs_G = constants.G.to('cm^3/(g*s^2)')
g = cgs_G*mass/r**2


#Find edge of core cz
cz_bool = (L_conv.value > 1)*(mass < 0.9*mass[-1])
core_cz_bound = mass[cz_bool][-1]
bound_ind = np.argmin(np.abs(mass - core_cz_bound))


L  = r[bound_ind]
g0 = g[bound_ind] 
rho0 = rho[0]
P0 = P[0]
T0 = T[0]

r_cz = r[cz_bool]/L

### Log Density
N = 20
ln_rho_field  = field.Field(dist=d, bases=(b,), dtype=np.float64)
ln_rho = np.log(rho[cz_bool]/rho0)
ln_rho_interp = np.interp(rg, r_cz, ln_rho)
ln_rho_field['g'] = ln_rho_interp
ln_rho_field['c'][:, :, N:] = 0
plot_ncc_figure(rg.flatten(), (-1)+ln_rho_interp.flatten(), (-1)+ln_rho_field['g'].flatten(), N, ylabel=r"$\ln\rho - 1$", fig_name="ln_rho")

grad_ln_rho_field  = field.Field(dist=d, bases=(b,), dtype=np.float64)
grad_ln_rho = np.gradient(ln_rho,r_cz)
grad_ln_rho_interp = np.interp(rg, r_cz, grad_ln_rho)
grad_ln_rho_field['g'] = grad(ln_rho_field).evaluate()['g'][-1,:]
grad_ln_rho_field['c'][:, :, N:] = 0
plot_ncc_figure(rg.flatten(), grad_ln_rho_interp.flatten(), grad_ln_rho_field['g'].flatten(), N, ylabel=r"$\nabla\ln\rho$", fig_name="grad_ln_rho")

### log (density * temp)
N = 20
ln_rhoT_field  = field.Field(dist=d, bases=(b,), dtype=np.float64)
ln_rhoT = np.log((rho*T)[cz_bool]/rho0/T0)
ln_rhoT_interp = np.interp(rg, r_cz, ln_rhoT)
ln_rhoT_field['g'] = ln_rhoT_interp
ln_rhoT_field['c'][:, :, N:] = 0
plot_ncc_figure(rg.flatten(), (-1)+ln_rhoT_interp.flatten(), (-1)+ln_rhoT_field['g'].flatten(), N, ylabel=r"$\ln(\rho T) - 1$", fig_name="ln_rhoT")


### Effective gravity
N = 64
g_eff = ((g/cp)*(L/T0))[cz_bool]
g_eff_field = field.Field(dist=d, bases=(b,), dtype=np.float64)
g_eff_interp = np.interp(rg, r_cz, g_eff)
g_eff_field['g'] = g_eff_interp
g_eff_field['c'][:, :, N:] = 0
plot_ncc_figure(rg.flatten(), g_eff_interp.flatten(), g_eff_field['g'].flatten(), N, ylabel=r"$g_{eff}$", fig_name="g_eff")


### inverse Temperature
N = 10
invT_field = field.Field(dist=d, bases=(b,), dtype=np.float64)
invT_interp = np.interp(rg, r_cz, T0/T[cz_bool])
invT_field['g'] = invT_interp
invT_field['c'][:, :, N:] = 0
plot_ncc_figure(rg.flatten(), invT_interp.flatten(), invT_field['g'].flatten(), N, ylabel=r"$(T/T_c)^{-1}$", fig_name="invT")

### effective heating / (rho * T)
N = 20
H = rho * eps
C = (np.gradient(Luminosity-L_conv,r)/(4*np.pi*r**2))
H_eff = H - C
H0 = H_eff[0]
H_NCC = ((H_eff / (rho*T)) * (rho0*T0) / H0)[cz_bool]
H_field = field.Field(dist=d, bases=(b,), dtype=np.float64)
H_interp = np.interp(rg, r_cz, H_NCC)
H_field['g'] = H_interp
H_field['c'][:, :, N:] = 0
plot_ncc_figure(rg.flatten(), H_interp.flatten(), H_field['g'].flatten(), N, ylabel=r"$(H_{eff}/(\rho T))$ (nondimensional)$", fig_name="H_eff")

#tau = (H0/L**2/rho0)**(-1/3)
#pomegac = T0*R/mu[0]
#Ma2 = (H0*L/rho0)**(2/3)/pomegac

#if get_dimensions:
#    return L, tau, Ma2



##return dlogrho_field['g'], dlogP_field['g'], T_field['g'], gradS_field['g'], grav_field['g'], H_field['g'], Ma2, mu[0]
#
