"""
Turns a MESA .data file of a massive star into a .h5 file of NCCs for a d3 run.

Usage:
    make_coreCZ_nccs.py [options]

Options:
    --Re=<R>        simulation reynolds/peclet number [default: 1e4]
    --Nmax=<N>        Maximum radial coefficients (ball) [default: 63]
    --file=<f>      Path to MESA log file [default: MESA_Models_Dedalus_Full_Sphere/LOGS/6.data]
    --pre_log_folder=<f>  Folder name in which 'LOGS' sits [default: ]
"""
import os
import time

import numpy as np
import h5py
from mpi4py import MPI
import mesa_reader as mr
import matplotlib.pyplot as plt
from dedalus.core import coords, distributor, basis, field, operators, arithmetic
from docopt import docopt

from astropy import units as u
from astropy import constants
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from numpy.polynomial import Chebyshev as Pfit

args = docopt(__doc__)
plot=True

from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

def plot_ncc_figure(mesa_r, mesa_y, dedalus_rs, dedalus_ys, Ns, ylabel="", fig_name="", out_dir='.', zero_line=False, log=False, r_int=None, ylim=None):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    if zero_line:
        ax1.axhline(0, c='k', lw=0.5)

    ax1.plot(mesa_r, mesa_y, label='mesa', c='k', lw=3)
    for r, y in zip(dedalus_rs, dedalus_ys):
        ax1.plot(r, y, label='dedalus', c='red')
    plt.legend(loc='best')
    ax1.set_xlabel('radius/L', labelpad=-3)
    ax1.set_ylabel('{}'.format(ylabel))
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')
    if log:
        ax1.set_yscale('log')
    if ylim is not None:
        ax1.set_ylim(ylim)

    ax2 = fig.add_subplot(2,1,2)
    mesa_func = interp1d(mesa_r, mesa_y, bounds_error=False, fill_value='extrapolate') 
    for r, y in zip(dedalus_rs, dedalus_ys):
        diff = np.abs(1 - mesa_func(r)/y)
        ax2.plot(r, diff)
    ax2.axhline(1e-1, c='k', lw=0.5)
    ax2.axhline(1e-2, c='k', lw=0.5)
    ax2.axhline(1e-3, c='k', lw=0.5)
    ax2.set_ylabel('abs(1 - mesa/dedalus)')
    ax2.set_xlabel('radius/L')
    ax2.set_yscale('log')

    ax2.set_ylim(1e-4, 1)
    if len(Ns) == 1:
        fig.suptitle('coeff bandwidth = {}'.format(Ns[0]))
    elif len(Ns) == 2:
        fig.suptitle('coeff bandwidth = {}, {}'.format(Ns[0], Ns[1]))
    else:
        raise NotImplementedError
    if r_int is not None:
        for ax in [ax1, ax2]:
            ax.axvline(r_int, c='k')
    fig.savefig('{:s}/{}.png'.format(out_dir, fig_name), bbox_inches='tight', dpi=200)

### Read in command line args
true_Nmax = Nmax = int(args['--Nmax'])
simulation_Re = float(args['--Re'])
read_file = args['--file']
filename = read_file.split('/LOGS/')[-1]
out_dir  = read_file.replace('/LOGS/', '_').replace('.data', '_coreCZ')
if args['--pre_log_folder'] != '':
    out_dir = '{:s}_{:s}'.format(args['--pre_log_folder'], out_dir)
print('saving files to {}'.format(out_dir))
out_file = '{:s}/ballShell_nccs_B{}_Re{}.h5'.format(out_dir, Nmax, args['--Re'])
if not os.path.exists('{:s}'.format(out_dir)):
    os.mkdir('{:s}'.format(out_dir))

### Read MESA file
p = mr.MesaData(read_file)
mass           = p.mass[::-1] * u.M_sun
r              = p.radius[::-1] * u.R_sun
rho            = 10**p.logRho[::-1] * u.g / u.cm**3
P              = 10**p.logP[::-1] * u.g / u.cm / u.s**2
eps            = p.eps_nuc[::-1] * u.erg / u.g / u.s
nablaT         = p.gradT[::-1] #dlnT/dlnP
nablaT_ad      = p.grada[::-1]
chiRho         = p.chiRho[::-1]
chiT           = p.chiT[::-1]
T              = 10**p.logT[::-1] * u.K
cp             = p.cp[::-1]  * u.erg / u.K / u.g
cv             = p.cv[::-1]  * u.erg / u.K / u.g
opacity        = p.opacity[::-1] * (u.cm**2 / u.g)
mu             = p.mu[::-1]
N2             = p.brunt_N2[::-1] / u.s**2
N2_structure   = p.brunt_N2_structure_term[::-1] / u.s**2
N2_composition = p.brunt_N2_composition_term[::-1] / u.s**2
Luminosity     = p.luminosity[::-1] * u.L_sun
conv_L_div_L   = p.conv_L_div_L[::-1]
csound         = p.csound[::-1] * u.cm / u.s

#secondary MESA fields
mass            = mass.cgs
r               = r.cgs
Luminosity      = Luminosity.cgs
L_conv          = conv_L_div_L*Luminosity
rad_diff        = 16 * constants.sigma_sb.cgs * T**3 / (3 * rho**2 * cp * opacity)
rad_diff        = rad_diff.cgs
g               = constants.G.cgs*mass/r**2
gamma           = cp/cv
dlogPdr         = -rho*g/P
gamma1          = dlogPdr/(-g/csound**2)
dlogrhodr       = dlogPdr*(chiT/chiRho)*(nablaT_ad - nablaT) - g/csound**2
dlogTdr         = dlogPdr*(nablaT)
N2_therm_approx = g*(dlogPdr/gamma1 - dlogrhodr)
grad_s          = cp*N2/g #entropy gradient, for NCC, includes composition terms
grad_T          = T*dlogTdr
H_eff           = (np.gradient(L_conv,r)/(4*np.pi*r**2)) # Heating, for ncc, H = rho*eps - portion carried by radiation
H_eff[0]        = H_eff[1] #make gradient 0 at core, remove weird artifacts from gradient near r = 0.


#Find edge of core cz
cz_bool = (L_conv.value > 1)*(mass < 0.9*mass[-1]) #rudimentary but works
core_cz_mass_bound = mass[cz_bool][-1]
core_cz_r          = r[cz_bool][-1]
core_cz_bound_ind  = np.argmin(np.abs(mass - core_cz_mass_bound))

#Set things up to slice out the star appropriately
radius_MESA   = r[cz_bool][-1] #outer radius of BallBasis
sim_bool      = r <= radius_MESA

cp_surf = cp[sim_bool][-1]

#Nondimensionalization
L       = L_CZ  = r[core_cz_bound_ind]
g0      = g[core_cz_bound_ind] 
rho0    = rho[0]
P0      = P[0]
T0      = T[0]
cp0     = cp[0]
gamma0  = gamma[0]
H0      = H_eff[0]
tau     = (H0/L**2/rho0)**(-1/3)
tau     = tau.cgs
u_H     = L/tau
Ma2     = u_H**2 / ((gamma0-1)*cp0*T0)
s_c     = Ma2*(gamma0-1)*cp0
Pe_rad  = u_H*L/rad_diff
inv_Pe_rad = 1/Pe_rad
print("L CZ:", L_CZ)

#MESA radial values, in simulation units
r_sim = r[sim_bool]/L
radius = radius_MESA/L

#Get some timestepping & wave frequency info
max_dt = 0.1

print('one time unit is {:.2e}'.format(tau))
print('output cadence is {} s / {} % of a heating time'.format(max_dt*tau, max_dt*100))
#
### Make dedalus domain
c = coords.SphericalCoordinates('φ', 'θ', 'r')
d = distributor.Distributor((c,), mesh=None)
bB = basis.BallBasis(c, (4, 2, Nmax+1), radius=radius.value, dtype=np.float64)
φB, θB, rB = bB.global_grids((1, 1, 1))

radComp   = lambda A: operators.RadialComponent(A)
grad = lambda A: operators.Gradient(A, c)
dot  = lambda A, B: arithmetic.DotProduct(A, B)

def make_NCC(basis, interp_args, Nmax=32, vector=False):
    interp = np.interp(*interp_args)
    if vector:
        this_field = field.Field(dist=d, bases=(basis,), tensorsig=(c,), dtype=np.float64)
        this_field['g'][2] = interp
        this_field['c'][:, :, :, Nmax:] = 0
    else:
        this_field = field.Field(dist=d, bases=(basis,), dtype=np.float64)
        this_field['g'] = interp
        this_field['c'][:, :, Nmax:] = 0
    return this_field, interp

### Radiative diffusivity == constant in CZ
Nmax = 3
inv_Pe_rad_fieldB, inv_Pe_rad_interpB = make_NCC(bB, (rB, r_sim,  (1/simulation_Re)*np.ones_like(r_sim) ), Nmax=Nmax)
grad_inv_Pe_B = grad(inv_Pe_rad_fieldB).evaluate()
grad_inv_Pe_B['g'] = 0
grad_inv_Pe_rad = np.gradient(inv_Pe_rad, r)
if plot:
    plot_ncc_figure(r[sim_bool]/L, inv_Pe_rad[sim_bool], (rB.flatten(), ), (inv_Pe_rad_fieldB['g'][:1,:1,:].flatten(),), (Nmax,), ylabel=r"$\mathrm{Pe}^{-1}$", fig_name="inv_Pe_rad", out_dir=out_dir, log=True, r_int=radius.value)
    plot_ncc_figure(r[sim_bool]/L, np.gradient(inv_Pe_rad, r/L)[sim_bool], (rB.flatten(),), (grad_inv_Pe_B['g'][2][:1,:1,:].flatten(),), (Nmax,), ylabel=r"$\nabla\mathrm{Pe}^{-1}$", fig_name="grad_inv_Pe_rad", out_dir=out_dir, log=True, r_int=radius.value, ylim=(1e-4/simulation_Re, 1))


### Log Density 
Nmax = 32
ln_rho_fieldB, ln_rho_interpB = make_NCC(bB, (rB, r_sim, np.log(rho/rho0)[sim_bool]), Nmax=Nmax)
grad_ln_rho_fieldB, grad_ln_rho_interpB = make_NCC(bB, (rB, r_sim, dlogrhodr[sim_bool]*L), Nmax=Nmax, vector=True)

if plot:
    plot_ncc_figure(r[sim_bool]/L, np.log(rho/rho0)[sim_bool], (rB.flatten(),), (ln_rho_fieldB['g'][:1,:1,:].flatten(),), (Nmax,), ylabel=r"$\ln\rho$", fig_name="ln_rho", out_dir=out_dir, log=False, r_int=radius.value)
    plot_ncc_figure(r[sim_bool]/L, (dlogrhodr*L)[sim_bool], (rB.flatten(),), (grad_ln_rho_fieldB['g'][2][:1,:1,:].flatten(),), (Nmax,), ylabel=r"$\nabla\ln\rho$", fig_name="grad_ln_rho", out_dir=out_dir, log=False, r_int=radius.value)

### Log Temperature
Nmax = 16
ln_T_fieldB, ln_T_interpB  = make_NCC(bB, (rB, r_sim, np.log(T/T0)[sim_bool]), Nmax=Nmax)
grad_ln_T_fieldB, grad_ln_T_interpB  = make_NCC(bB, (rB, r_sim, dlogTdr[sim_bool]*L), Nmax=Nmax, vector=True)

if plot:
    plot_ncc_figure(r[sim_bool]/L, np.log(T/T0)[sim_bool], (rB.flatten(),), (ln_T_fieldB['g'][:1,:1,:].flatten(),), (Nmax,), ylabel=r"$\ln T$", fig_name="ln_T", out_dir=out_dir, log=False, r_int=radius.value)
    plot_ncc_figure(r[sim_bool]/L, (dlogTdr*L)[sim_bool], (rB.flatten(),), (grad_ln_T_fieldB['g'][2][:1,:1,:].flatten(),), (Nmax,), ylabel=r"$\nabla\ln T$", fig_name="grad_ln_T", out_dir=out_dir, log=False, r_int=radius.value)

### Temperature
Nmax = 32
T_fieldB, T_interpB = make_NCC(bB, (rB, r_sim, (T/T0)[sim_bool]), Nmax=Nmax)

if plot:
    plot_ncc_figure(r[sim_bool]/L, (T/T0)[sim_bool], (rB.flatten(),), (T_fieldB['g'][:1,:1,:].flatten(),), (Nmax,), ylabel=r"$T$", fig_name="T", out_dir=out_dir, log=True, r_int=radius.value)


Nmax = 32
grad_T_fieldB, grad_T_interpB = make_NCC(bB, (rB, r_sim,  grad_T[sim_bool] * (L/T0)), Nmax=Nmax, vector=True)

if plot:
    plot_ncc_figure(r[sim_bool]/L, -grad_T[sim_bool]*(L/T0), (rB.flatten(),), (-grad_T_fieldB['g'][2][:1,:1,:].flatten(),), (Nmax,), ylabel=r"$-\nabla T$", fig_name="grad_T", out_dir=out_dir, log=True, r_int=radius.value)



### effective heating / (rho * T)
H_NCC = ((H_eff)  / H0) * (rho0*T0/rho/T)
Nmax = 60
H_fieldB, H_interpB = make_NCC(bB, (rB, r_sim, H_NCC[sim_bool]), Nmax=Nmax)
if plot:
    plot_ncc_figure(r[sim_bool]/L, H_NCC[sim_bool], (rB.flatten(),), (H_fieldB['g'][:1,:1,:].flatten(),), (Nmax,), ylabel=r"$H$", fig_name="heating", out_dir=out_dir, log=False, r_int=radius.value)


### entropy gradient
Nmax = 3
grad_s_fieldB, grad_s_interpB = make_NCC(bB, (rB, r_sim, np.zeros_like(r_sim)), Nmax=Nmax, vector=True)

if plot:
    plot_ncc_figure(r[sim_bool]/L, (grad_s*L/s_c)[sim_bool], (rB.flatten(),), (grad_s_fieldB['g'][2][:1,:1,:].flatten(),), (Nmax,), ylabel=r"$\nabla s$", fig_name="grad_s", out_dir=out_dir, log=True, r_int=radius.value)

with h5py.File('{:s}'.format(out_file), 'w') as f:
    #slicing preserves dimensionality.
    f['rB']          = rB
    f['TB']          = T_fieldB['g'][:1,:1,:]
    f['grad_TB']     = grad_T_fieldB['g'][:,:1,:1,:]
    f['H_effB']      = H_fieldB['g'][:1,:1,:]
    f['ln_ρB']       = ln_rho_fieldB['g'][:1,:1,:]
    f['ln_TB']       = ln_T_fieldB['g'][:1,:1,:]
    f['grad_ln_TB']  = grad_ln_T_fieldB['g'][:,:1,:1,:]
    f['grad_ln_ρB']  = grad_ln_rho_fieldB['g'][:,:1,:1,:]
    f['grad_s0B']    = grad_s_fieldB['g'][:,:1,:1,:]
    f['inv_Pe_radB'] = inv_Pe_rad_fieldB['g'][:1,:1,:]
    f['grad_inv_Pe_radB'] = grad_inv_Pe_B['g'][:,:1,:1,:]

    f['radius']   = radius
    f['L']   = L
    f['g0']  = g0
    f['ρ0']  = rho0
    f['P0']  = P0
    f['T0']  = T0
    f['H0']  = H0
    f['tau'] = tau 
    f['Ma2'] = tau 
    f['max_dt'] = max_dt
    f['s_c'] = s_c
    f['cp_surf'] = cp_surf
    f['r_mesa'] = r
    f['g_mesa'] = g 
    f['cp_mesa'] = cp
print(tau, tau/60/60/24)
