"""
Turns a MESA .data file of a massive star into a .h5 file of NCCs for a d3 run.

Usage:
    make_ball_shell_nccs.py [options]

Options:
    --NB=<N>        Maximum radial coefficients (ball) [default: 63]
    --NS=<N>        Maximum radial coefficients (shell) [default: 63]
    --file=<f>      Path to MESA log file [default: MESA_Models_Dedalus_Full_Sphere/LOGS/6.data]
    --pre_log_folder=<f>  Folder name in which 'LOGS' sits [default: ]
    --halfStar      If flagged, only get the inner 50% of the star
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
plot=True

from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

def plot_ncc_figure(r, mesa_y, dedalus_y, N, ylabel="", fig_name="", out_dir='.', zero_line=False, log=False):
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
    if log:
        ax1.set_yscale('log')

    ax2 = fig.add_subplot(2,1,2)
    difference = np.abs(1 - dedalus_y/mesa_y)
    ax2.plot(r, np.abs(difference).flatten())
    ax2.set_ylabel('abs(1 - dedalus/mesa)')
    ax2.set_xlabel('radius/L')
    ax2.set_yscale('log')
    fig.suptitle('coeff bandwidth = {}'.format(N))
    fig.savefig('{:s}/{}.png'.format(out_dir, fig_name), bbox_inches='tight', dpi=200)

### Read in command line args
true_NmaxB = NmaxB = int(args['--NB'])
true_NmaxS = NmaxS = int(args['--NS'])
read_file = args['--file']
filename = read_file.split('/LOGS/')[-1]
if args['--halfStar']:
    out_dir  = read_file.replace('/LOGS/', '_').replace('.data', '_ballShell_halfStar')
else:
    out_dir  = read_file.replace('/LOGS/', '_').replace('.data', '_ballShell')
if args['--pre_log_folder'] != '':
    out_dir = '{:s}_{:s}'.format(args['--pre_log_folder'], out_dir)
print('saving files to {}'.format(out_dir))
out_file = '{:s}/ballShell_nccs_B{}_S{}.h5'.format(out_dir, NmaxB, NmaxS)
if not os.path.exists('{:s}'.format(out_dir)):
    os.mkdir('{:s}'.format(out_dir))

### Read MESA file
p = mr.MesaData(read_file)
mass = p.mass[::-1] * u.M_sun
r = p.radius[::-1] * u.R_sun
mass, r = mass.cgs, r.cgs
rho = 10**p.logRho[::-1] * u.g / u.cm**3
P = 10**p.logP[::-1] * u.g / u.cm / u.s**2
eps = p.eps_nuc[::-1] * u.erg / u.g / u.s
nablaT = p.gradT[::-1] #dlnT/dlnP
T = 10**p.logT[::-1] * u.K
cp = p.cp[::-1]  * u.erg / u.K / u.g
cv = p.cv[::-1]  * u.erg / u.K / u.g
opacity = p.opacity[::-1] * (u.cm**2 / u.g)
mu = p.mu[::-1]
N2 = p.brunt_N2[::-1] / u.s**2
N2_structure   = p.brunt_N2_structure_term[::-1] / u.s**2
N2_composition = p.brunt_N2_composition_term[::-1] / u.s**2
Luminosity = p.luminosity[::-1] * u.L_sun
Luminosity = Luminosity.cgs
L_conv = p.conv_L_div_L[::-1]*Luminosity
csound = p.csound[::-1] * u.cm / u.s
rad_diff = 16 * constants.sigma_sb.cgs * T**3 / (3 * rho**2 * cp * opacity)
rad_diff = rad_diff.cgs
g = constants.G.cgs*mass/r**2
gamma = cp/cv
chiRho  = p.chiRho[::-1]
chiT    = p.chiT[::-1]
nablaT =  p.gradT[::-1]
nablaT_ad = p.grada[::-1]
dlogPdr = -rho*g/P
gamma1  = dlogPdr/(-g/csound**2)
dlogrhodr = dlogPdr*(chiT/chiRho)*(nablaT_ad - nablaT) - g/csound**2
dlogTdr   = dlogPdr*(nablaT)
N2_therm_approx = g*(dlogPdr/gamma1 - dlogrhodr)

# Entropy gradient, for ncc
grad_s = cp*N2/g #includes composition terms
# Heating, for ncc, H = rho*eps - portion carried by radiation
H_eff = (np.gradient(L_conv,r)/(4*np.pi*r**2))
H_eff[0] = H_eff[1] #make gradient 0 at core, remove weird artifacts from gradient near r = 0.

#Find edge of core cz
cz_bool = (L_conv.value > 1)*(mass < 0.9*mass[-1])
core_cz_mass_bound = mass[cz_bool][-1]
core_cz_bound_ind = np.argmin(np.abs(mass - core_cz_mass_bound))

#Find bottom edge of FeCZ
fracStar = 0.95
fe_cz = (mass > 1.1*mass[core_cz_bound_ind])*(L_conv.value > 1)
bot_fe_cz_r = fracStar*r[fe_cz][0]
fe_cz_bound_ind = np.argmin(np.abs(r - bot_fe_cz_r))
print('fraction of FULL star simulated: {}'.format(bot_fe_cz_r/r[-1]))

#Set things up to slice out the star appropriately
halfStar_r = r[-1]/2
r_inner    = r[cz_bool][-1]*1.1
if args['--halfStar']:
    r_outer    = halfStar_r
else:
    r_outer    = bot_fe_cz_r
ball_bool  = r <= r_inner
shell_bool = (r > r_inner)*(r <= r_outer)

#Nondimensionalization
L = L_CZ  = r[core_cz_bound_ind]
g0 = g[core_cz_bound_ind] 
rho0 = rho[0]
P0 = P[0]
T0 = T[0]
cp0 = cp[0]
gamma0 = gamma[0]
H0 = H_eff[0]
tau = (H0/L**2/rho0)**(-1/3)
tau = tau.cgs
u_H = L/tau
Ma2 = u_H**2 / ((gamma0-1)*cp0*T0)
s_c = Ma2*(gamma0-1)*cp0
Pe_rad = u_H*L/rad_diff
inv_Pe_rad = 1/Pe_rad

r_ball = r[ball_bool]/L
r_shell = r[shell_bool]/L
r_inner /= L
r_outer /= L

wave_tau_ball  = (1/20)*2*np.pi/np.sqrt(N2[ball_bool].max())
wave_tau_shell = (1/20)*2*np.pi/np.sqrt(N2[shell_bool].max())
max_dt_ball    = wave_tau_ball/tau
max_dt_shell   = wave_tau_shell/tau
print('one time unit is {:.2e}'.format(tau))
print('output cadence is {} s / {} % of a heating time'.format(np.min((wave_tau_ball.value, wave_tau_shell.value)), np.min((wave_tau_ball.value, wave_tau_shell.value))/tau.value*100))

### Make dedalus domain
c = coords.SphericalCoordinates('φ', 'θ', 'r')
d = distributor.Distributor((c,), mesh=None)
bB = basis.BallBasis(c, (1, 1, NmaxB+1), radius=r_inner, dtype=np.float64, dealias=(1, 1, 1))
bS = basis.SphericalShellBasis(c, (1, 1, NmaxS+1), radii=(r_inner, r_outer), dtype=np.float64, dealias=(1, 1, 1))
φB, θB, rB = bB.global_grids((1, 1, 1))
φS, θS, rS = bS.global_grids((1, 1, 1))

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

### Radiative diffusivity
NmaxB, NmaxS = 8, 62
inv_Pe_rad_fieldB, inv_Pe_rad_interpB = make_NCC(bB, (rB, r_ball, inv_Pe_rad[ball_bool]), Nmax=NmaxB)
inv_Pe_rad_fieldS, inv_Pe_rad_interpS = make_NCC(bS, (rS, r_shell, inv_Pe_rad[shell_bool]), Nmax=NmaxS)
if plot:
    plot_ncc_figure(rB.flatten(), inv_Pe_rad_interpB.flatten(), inv_Pe_rad_fieldB['g'].flatten(), NmaxB, ylabel=r"$\mathrm{Pe}^{-1}$", fig_name="inv_Pe_radB", out_dir=out_dir, log=True)
    plot_ncc_figure(rS.flatten(), inv_Pe_rad_interpS.flatten(), inv_Pe_rad_fieldS['g'].flatten(), NmaxS, ylabel=r"$\mathrm{Pe}^{-1}$", fig_name="inv_Pe_radS", out_dir=out_dir, log=True)

### Log Density 
NmaxB, NmaxS = 8, 32
ln_rho_fieldB, ln_rho_interpB = make_NCC(bB, (rB, r_ball, np.log(rho/rho0)[ball_bool]), Nmax=NmaxB)
grad_ln_rho_fieldB, grad_ln_rho_interpB = make_NCC(bB, (rB, r_ball, dlogrhodr[ball_bool]*L), Nmax=NmaxB, vector=True)
ln_rho_fieldS, ln_rho_interpS = make_NCC(bS, (rS, r_shell, np.log(rho/rho0)[shell_bool]), Nmax=NmaxS)
grad_ln_rho_fieldS, grad_ln_rho_interpS = make_NCC(bS, (rS, r_shell, dlogrhodr[shell_bool]*L), Nmax=NmaxS, vector=True)
if plot:
    plot_ncc_figure(rB.flatten(), (-1)+ln_rho_interpB.flatten(), (-1)+ln_rho_fieldB['g'].flatten(), NmaxB, ylabel=r"$\ln\rho - 1$", fig_name="ln_rhoB", out_dir=out_dir)
    plot_ncc_figure(rB.flatten(), grad_ln_rho_interpB.flatten(), grad_ln_rho_fieldB['g'][2].flatten(), NmaxB, ylabel=r"$\nabla\ln\rho$", fig_name="grad_ln_rhoB", out_dir=out_dir)
    plot_ncc_figure(rS.flatten(), (-1)+ln_rho_interpS.flatten(), (-1)+ln_rho_fieldS['g'].flatten(), NmaxS, ylabel=r"$\ln\rho - 1$", fig_name="ln_rhoS", out_dir=out_dir)
    plot_ncc_figure(rS.flatten(), grad_ln_rho_interpS.flatten(), grad_ln_rho_fieldS['g'][2].flatten(), NmaxS, ylabel=r"$\nabla\ln\rho$", fig_name="grad_ln_rhoS", out_dir=out_dir)

### Log Temperature
NmaxB, NmaxS = 16, 32
ln_T_fieldB, ln_T_interpB  = make_NCC(bB, (rB, r_ball, np.log(T/T0)[ball_bool]), Nmax=NmaxB)
grad_ln_T_fieldB, grad_ln_T_interpB  = make_NCC(bB, (rB, r_ball, dlogTdr[ball_bool]*L), Nmax=NmaxB, vector=True)
ln_T_fieldS, ln_T_interpS  = make_NCC(bS, (rS, r_shell, np.log(T/T0)[shell_bool]), Nmax=NmaxS)
grad_ln_T_fieldS, grad_ln_T_interpS  = make_NCC(bS, (rS, r_shell, dlogTdr[shell_bool]*L), Nmax=NmaxS, vector=True)
if plot:
    plot_ncc_figure(rB.flatten(), (-1)+ln_T_interpB.flatten(), (-1)+ln_T_fieldB['g'].flatten(), NmaxB, ylabel=r"$\ln(T) - 1$", fig_name="ln_TB", out_dir=out_dir)
    plot_ncc_figure(rB.flatten(), grad_ln_T_interpB.flatten(), grad_ln_T_fieldB['g'][2].flatten(), NmaxB, ylabel=r"$\nabla\ln(T)$", fig_name="grad_ln_TB", out_dir=out_dir)
    plot_ncc_figure(rS.flatten(), (-1)+ln_T_interpS.flatten(), (-1)+ln_T_fieldS['g'].flatten(), NmaxS, ylabel=r"$\ln(T) - 1$", fig_name="ln_TS", out_dir=out_dir)
    plot_ncc_figure(rS.flatten(), grad_ln_T_interpS.flatten(), grad_ln_T_fieldS['g'][2].flatten(), NmaxS, ylabel=r"$\nabla\ln(T)$", fig_name="grad_ln_TS", out_dir=out_dir)

### Temperature
NmaxB, NmaxS = 32, 32
T_fieldB, T_interpB = make_NCC(bB, (rB, r_ball, (T/T0)[ball_bool]), Nmax=NmaxB)
T_fieldS, T_interpS = make_NCC(bS, (rS, r_shell, (T/T0)[shell_bool]), Nmax=NmaxS)
if plot:
    plot_ncc_figure(rB.flatten(), T_interpB.flatten(), T_fieldB['g'].flatten(), NmaxB, ylabel=r"$T/T_c$", fig_name="TB", out_dir=out_dir)
    plot_ncc_figure(rS.flatten(), T_interpS.flatten(), T_fieldS['g'].flatten(), NmaxS, ylabel=r"$T/T_c$", fig_name="TS", out_dir=out_dir)

### effective heating / (rho * T)
#Logic for smoothing heating profile at outer edge of CZ. Adjust outer edge of heating
#Since this is grid-locked, this isn't really needed.
#heat_erf_width  = 0.04
#heat_erf_center = 0.9891
#amount_to_adjust = 0.95
#full_lum_above = np.sum((4*np.pi*r_ball**2*H_NCC[ball_bool]*np.gradient(r_ball))[r_ball > amount_to_adjust].flatten())
#approx_H = H_NCC[ball_bool][r_ball <= amount_to_adjust].flatten()[-1]*one_to_zero(r_ball, heat_erf_center, heat_erf_width)
#full_lum_approx = np.sum((4*np.pi*r_ball**2*approx_H*np.gradient(r_ball))[r_ball > amount_to_adjust].flatten())
#H_NCC_ball = H_NCC[ball_bool]
#H_NCC_ball[r_ball > amount_to_adjust] = approx_H[r_ball > amount_to_adjust]
#print('outer', full_lum_above, full_lum_approx)
H_NCC = ((H_eff)  / H0) * (rho0*T0/rho/T)
NmaxB, NmaxS = true_NmaxB, true_NmaxS
H_fieldB, H_interpB = make_NCC(bB, (rB, r_ball, H_NCC[ball_bool]), Nmax=NmaxB)
H_fieldS, H_interpS = make_NCC(bS, (rS, r_shell, H_NCC[shell_bool]), Nmax=NmaxS)
if plot:
    plot_ncc_figure(rB.flatten(), H_interpB.flatten(), H_fieldB['g'].flatten(), NmaxB, ylabel=r"$(H_{eff}/(\rho c_p T))$ (nondimensional)", fig_name="H_effB", out_dir=out_dir, zero_line=True)
    plot_ncc_figure(rS.flatten(), H_interpS.flatten(), H_fieldS['g'].flatten(), NmaxS, ylabel=r"$(H_{eff}/(\rho c_p T))$ (nondimensional)", fig_name="H_eff", out_dir=out_dir, zero_line=True)


### entropy gradient
transition_point = 1.03
width = 0.06
N = 32
N_after = 96
center =  transition_point - 0.5*width
width *= (L_CZ/L).value
center *= (L_CZ/L).value

#Build a nice function for our basis in the ball
grad_s_smooth = np.copy(grad_s)
flat_value  = np.interp(transition_point, r/L, grad_s)
grad_s_smooth[r/L < transition_point] = flat_value

NmaxB, NmaxS = 32, 32
NmaxB_after = 96
grad_s_fieldB, grad_s_interpB = make_NCC(bB, (rB, r_ball, (grad_s_smooth*L/s_c)[ball_bool]), Nmax=NmaxB, vector=True)
grad_s_interpB = np.interp(rB, r_ball, (grad_s*L/s_c)[ball_bool])
grad_s_fieldS, grad_s_interpS = make_NCC(bS, (rS, r_shell, (grad_s*L/s_c)[shell_bool]), Nmax=NmaxS, vector=True)
grad_s_fieldB['g'][2] *= zero_to_one(rB, center, width=width).value
grad_s_fieldB['c'][:,:,:,NmaxB_after:] = 0
if plot:
    plot_ncc_figure(rB.flatten(), grad_s_interpB.flatten(), grad_s_fieldB['g'][2].flatten(), NmaxB, ylabel=r"$L(\nabla s/s_c)$", fig_name="grad_sB", out_dir=out_dir, log=True)
    plot_ncc_figure(rS.flatten(), grad_s_interpS.flatten(), grad_s_fieldS['g'][2].flatten(), NmaxS, ylabel=r"$L(\nabla s/s_c)$", fig_name="grad_sS", out_dir=out_dir, log=True)

#plt.show()
with h5py.File('{:s}'.format(out_file), 'w') as f:
    f['rB']          = rB
    f['TB']          = T_fieldB['g']
    f['H_effB']      = H_fieldB['g']
    f['ln_ρB']       = ln_rho_fieldB['g'] 
    f['ln_TB']       = ln_T_fieldB['g']
    f['grad_ln_TB']  = grad_ln_T_fieldB['g']
    f['grad_ln_ρB']  = grad_ln_rho_fieldB['g']
    f['grad_s0B']    = grad_s_fieldB['g']
    f['inv_Pe_radB'] = inv_Pe_rad_fieldB['g']

    f['rS']          = rS
    f['TS']          = T_fieldS['g']
    f['H_effS']      = H_fieldS['g']
    f['ln_ρS']       = ln_rho_fieldS['g'] 
    f['ln_TS']       = ln_T_fieldS['g']
    f['grad_ln_TS']  = grad_ln_T_fieldS['g']
    f['grad_ln_ρS']  = grad_ln_rho_fieldS['g']
    f['grad_s0S']    = grad_s_fieldS['g']
    f['inv_Pe_radS'] = inv_Pe_rad_fieldS['g']

    f['r_inner']   = r_inner
    f['r_outer']   = r_outer
    f['L']   = L
    f['g0']  = g0
    f['ρ0']  = rho0
    f['P0']  = P0
    f['T0']  = T0
    f['H0']  = H0
    f['tau'] = tau 
    f['Ma2'] = tau 
    f['max_dt_ball'] = max_dt_ball
    f['max_dt_shell'] = max_dt_shell
    f['max_dt'] = np.min((max_dt_shell, max_dt_ball))
