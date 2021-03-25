"""
Turns a MESA .data file of a massive star into a .h5 file of NCCs for a d3 run.

Usage:
    make_ball_shell_nccs.py [options]

Options:
    --Re=<R>        simulation reynolds/peclet number [default: 1e4]
    --NB=<N>        Maximum radial coefficients (ball) [default: 95]
    --NS=<N>        Maximum radial coefficients (shell) [default: 95]
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
    fig.suptitle('coeff bandwidth = {}, {}'.format(Ns[0], Ns[1]))
    if r_int is not None:
        for ax in [ax1, ax2]:
            ax.axvline(r_int, c='k')
    fig.savefig('{:s}/{}.png'.format(out_dir, fig_name), bbox_inches='tight', dpi=200)

### Read in command line args
true_NmaxB = NmaxB = int(args['--NB'])
true_NmaxS = NmaxS = int(args['--NS'])
simulation_Re = float(args['--Re'])
read_file = args['--file']
filename = read_file.split('/LOGS/')[-1]
if args['--halfStar']:
    out_dir  = read_file.replace('/LOGS/', '_').replace('.data', '_ballShell_halfStar')
else:
    out_dir  = read_file.replace('/LOGS/', '_').replace('.data', '_ballShell')
if args['--pre_log_folder'] != '':
    out_dir = '{:s}_{:s}'.format(args['--pre_log_folder'], out_dir)
print('saving files to {}'.format(out_dir))
out_file = '{:s}/ballShell_nccs_B{}_S{}_Re{}.h5'.format(out_dir, NmaxB, NmaxS, args['--Re'])
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

#plt.plot(r, L_conv)
#plt.xlabel('radius')
#plt.ylabel('L_conv')
#plt.show()


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
sim_bool = r <= r_outer

cp_surf = cp[shell_bool][-1]

lamb_freq = lambda ell : np.sqrt(ell*(ell + 1)) * csound/r
plt.figure()
plt.plot(r, np.sqrt(N2), label=r'$N$')
plt.plot(r, lamb_freq(1), label=r'$S_1$')
plt.plot(r, lamb_freq(10), label=r'$S_{10}$')
plt.plot(r, lamb_freq(100), label=r'$S_{100}$')
plt.xlim(0, r_outer.value)
plt.xlabel('r (cm)')
plt.ylabel('freq (1/s)')
plt.yscale('log')
plt.legend(loc='best')
plt.savefig('{}/propagation_diagram.png'.format(out_dir), dpi=300, bbox_inches='tight')


#plt.plot(r, cv)#gamma, label=r'$\gamma = cp/cv$')
##plt.plot(r, gamma, label=r'$\gamma = cp/cv$')
##plt.plot(r, gamma1, label=r'$\Gamma_1 = dlogPdr/(-g/c_sound^2)$')
##plt.legend(loc='best')
#plt.axvline(bot_fe_cz_r.value)
#plt.xlabel('r')
#plt.ylabel('cp')
#plt.show()

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
print("L CZ:", L_CZ)

sim_inv_Pe_rad = np.copy(inv_Pe_rad)
sim_inv_Pe_rad -= 1/simulation_Re
sim_inv_Pe_rad[sim_inv_Pe_rad < 0] = 0
r_Pe_min = r[sim_inv_Pe_rad > 0].min()
sim_inv_Pe_rad *= zero_to_one(r/L, r_Pe_min/L, width=0.05)
sim_inv_Pe_rad += 1/simulation_Re


r_ball = r[ball_bool]/L
r_shell = r[shell_bool]/L
r_inner /= L
r_outer /= L

cp_surf = cp[shell_bool][-1]
print(s_c, cp_surf)

N2max_ball = N2[ball_bool].max()
N2max_shell = N2[shell_bool].max()
N2plateau = N2[r < 4.5e11*u.cm][-1]
wave_tau_ball  = (1/20)*2*np.pi/np.sqrt(N2max_ball)
wave_tau_shell = (1/20)*2*np.pi/np.sqrt(N2max_shell)
kepler_tau     = 30*60*u.s
max_dt_ball    = wave_tau_ball/tau
max_dt_shell   = wave_tau_shell/tau
max_dt_kepler  = kepler_tau/tau
if max_dt_kepler > max_dt_ball and max_dt_kepler > max_dt_shell:
    max_dt = max_dt_kepler
else:
    max_dt = np.min((max_dt_ball, max_dt_shell))
print('one time unit is {:.2e}'.format(tau))
print('output cadence is {} s / {} % of a heating time'.format(max_dt*tau, max_dt*100))
#
### Make dedalus domain
c = coords.SphericalCoordinates('φ', 'θ', 'r')
d = distributor.Distributor((c,), mesh=None)
bB = basis.BallBasis(c, (4, 2, NmaxB+1), radius=r_inner.value, dtype=np.float64)
bS = basis.SphericalShellBasis(c, (4, 2, NmaxS+1), radii=(r_inner.value, r_outer.value), dtype=np.float64)
φB, θB, rB = bB.global_grids((1, 1, 1))
φS, θS, rS = bS.global_grids((1, 1, 1))

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

def match_boundary(fB, fS, adjust_ball=False):
    if len(fB.tensorsig) == 1:
        fB_interface = radComp(fB(r=r_inner.value)).evaluate()['g'].max()
        fS_interface = radComp(fS(r=r_inner.value)).evaluate()['g'].max()
        if adjust_ball:
            fB['g'][2,:] -= (fB_interface - fS_interface)
        else:
            fS['g'][2,:] -= (fS_interface - fB_interface)
    else:
        fB_interface = fB(r=r_inner.value).evaluate()['g'].max()
        fS_interface = fS(r=r_inner.value).evaluate()['g'].max()
        if adjust_ball:
            fB['g'][:] -= (fB_interface - fS_interface)
        else:
            fS['g'][:] -= (fS_interface - fB_interface)

### Radiative diffusivity
NmaxB, NmaxS = 8, 90#np.min((true_NmaxS - 1, 126))
gradPe_B_cutoff = 10
gradPe_S_cutoff = 93
inv_Pe_rad_fieldB, inv_Pe_rad_interpB = make_NCC(bB, (rB, r_ball,  sim_inv_Pe_rad[ball_bool]), Nmax=NmaxB)
inv_Pe_rad_fieldS, inv_Pe_rad_interpS = make_NCC(bS, (rS, r_shell, sim_inv_Pe_rad[shell_bool]), Nmax=NmaxS)

#match_boundary(inv_Pe_rad_fieldB, inv_Pe_rad_fieldS)


grad_inv_Pe_B = grad(inv_Pe_rad_fieldB).evaluate()
grad_inv_Pe_B['c'][:,:,:,gradPe_B_cutoff:] = 0

grad_inv_Pe_S = grad(inv_Pe_rad_fieldS).evaluate()
grad_inv_Pe_S['c'][:,:,:,int(NmaxS/2):] = 0
transition = (r/L)[sim_inv_Pe_rad > sim_inv_Pe_rad.min()][0].value
grad_inv_Pe_S['g'][2] *= zero_to_one(rS, transition, width=0.05)
grad_inv_Pe_S['c'][:,:,:,gradPe_S_cutoff:] = 0

#match_boundary(grad_inv_Pe_B, grad_inv_Pe_S)

#grad_inv_Pe_rad_fieldB, grad_inv_Pe_rad_interpB = make_NCC(bB, (rB, r_ball,  np.gradient(inv_Pe_rad, r/L)[ball_bool]), Nmax=NmaxB)
#grad_inv_Pe_rad_fieldS, grad_inv_Pe_rad_interpS = make_NCC(bS, (rS, r_shell, np.gradient(inv_Pe_rad, r/L)[shell_bool]), Nmax=NmaxS)
grad_inv_Pe_rad = np.gradient(inv_Pe_rad, r)
if plot:
    plot_ncc_figure(r[sim_bool]/L, sim_inv_Pe_rad[sim_bool], (rB.flatten(), rS.flatten()), (inv_Pe_rad_fieldB['g'][:1,:1,:].flatten(), inv_Pe_rad_fieldS['g'][:1,:1,:].flatten()), (NmaxB, NmaxS), ylabel=r"$\mathrm{Pe}^{-1}$", fig_name="inv_Pe_rad", out_dir=out_dir, log=True, r_int=r_inner.value)
    plot_ncc_figure(r[sim_bool]/L, np.gradient(sim_inv_Pe_rad, r/L)[sim_bool], (rB.flatten(), rS.flatten()), (grad_inv_Pe_B['g'][2][:1,:1,:].flatten(), grad_inv_Pe_S['g'][2][:1,:1,:].flatten()), (NmaxB, NmaxS), ylabel=r"$\nabla\mathrm{Pe}^{-1}$", fig_name="grad_inv_Pe_rad", out_dir=out_dir, log=True, r_int=r_inner.value, ylim=(1e-4/simulation_Re, 1))



### Log Density 
NmaxB, NmaxS = 8, 32
ln_rho_fieldB, ln_rho_interpB = make_NCC(bB, (rB, r_ball, np.log(rho/rho0)[ball_bool]), Nmax=NmaxB)
grad_ln_rho_fieldB, grad_ln_rho_interpB = make_NCC(bB, (rB, r_ball, dlogrhodr[ball_bool]*L), Nmax=NmaxB, vector=True)
ln_rho_fieldS, ln_rho_interpS = make_NCC(bS, (rS, r_shell, np.log(rho/rho0)[shell_bool]), Nmax=NmaxS)
grad_ln_rho_fieldS, grad_ln_rho_interpS = make_NCC(bS, (rS, r_shell, dlogrhodr[shell_bool]*L), Nmax=NmaxS, vector=True)

#match_boundary(ln_rho_fieldB, ln_rho_fieldS)
#match_boundary(grad_ln_rho_fieldB, grad_ln_rho_fieldS)

if plot:
    plot_ncc_figure(r[sim_bool]/L, np.log(rho/rho0)[sim_bool], (rB.flatten(), rS.flatten()), (ln_rho_fieldB['g'][:1,:1,:].flatten(), ln_rho_fieldS['g'][:1,:1,:].flatten()), (NmaxB, NmaxS), ylabel=r"$\ln\rho$", fig_name="ln_rho", out_dir=out_dir, log=False, r_int=r_inner.value)
    plot_ncc_figure(r[sim_bool]/L, (dlogrhodr*L)[sim_bool], (rB.flatten(), rS.flatten()), (grad_ln_rho_fieldB['g'][2][:1,:1,:].flatten(), grad_ln_rho_fieldS['g'][2][:1,:1,:].flatten()), (NmaxB, NmaxS), ylabel=r"$\nabla\ln\rho$", fig_name="grad_ln_rho", out_dir=out_dir, log=False, r_int=r_inner.value)

### Log Temperature
NmaxB, NmaxS = 16, 32
ln_T_fieldB, ln_T_interpB  = make_NCC(bB, (rB, r_ball, np.log(T/T0)[ball_bool]), Nmax=NmaxB)
grad_ln_T_fieldB, grad_ln_T_interpB  = make_NCC(bB, (rB, r_ball, dlogTdr[ball_bool]*L), Nmax=NmaxB, vector=True)
ln_T_fieldS, ln_T_interpS  = make_NCC(bS, (rS, r_shell, np.log(T/T0)[shell_bool]), Nmax=NmaxS)
grad_ln_T_fieldS, grad_ln_T_interpS  = make_NCC(bS, (rS, r_shell, dlogTdr[shell_bool]*L), Nmax=NmaxS, vector=True)

#match_boundary(ln_T_fieldB, ln_T_fieldS)
#match_boundary(grad_ln_T_fieldB, grad_ln_T_fieldS)

if plot:
    plot_ncc_figure(r[sim_bool]/L, np.log(T/T0)[sim_bool], (rB.flatten(), rS.flatten()), (ln_T_fieldB['g'][:1,:1,:].flatten(), ln_T_fieldS['g'][:1,:1,:].flatten()), (NmaxB, NmaxS), ylabel=r"$\ln T$", fig_name="ln_T", out_dir=out_dir, log=False, r_int=r_inner.value)
    plot_ncc_figure(r[sim_bool]/L, (dlogTdr*L)[sim_bool], (rB.flatten(), rS.flatten()), (grad_ln_T_fieldB['g'][2][:1,:1,:].flatten(), grad_ln_T_fieldS['g'][2][:1,:1,:].flatten()), (NmaxB, NmaxS), ylabel=r"$\nabla\ln T$", fig_name="grad_ln_T", out_dir=out_dir, log=False, r_int=r_inner.value)

### Temperature
NmaxB, NmaxS = 32, 32
T_fieldB, T_interpB = make_NCC(bB, (rB, r_ball, (T/T0)[ball_bool]), Nmax=NmaxB)
T_fieldS, T_interpS = make_NCC(bS, (rS, r_shell, (T/T0)[shell_bool]), Nmax=NmaxS)

#match_boundary(T_fieldB, T_fieldS)

if plot:
    plot_ncc_figure(r[sim_bool]/L, (T/T0)[sim_bool], (rB.flatten(), rS.flatten()), (T_fieldB['g'][:1,:1,:].flatten(), T_fieldS['g'][:1,:1,:].flatten()), (NmaxB, NmaxS), ylabel=r"$T$", fig_name="T", out_dir=out_dir, log=True, r_int=r_inner.value)


grad_T = (T/T0)*dlogTdr*L
NmaxB, NmaxS = 32, 32
grad_T_fieldB, grad_T_interpB = make_NCC(bB, (rB, r_ball,  grad_T[ball_bool]), Nmax=NmaxB, vector=True)
grad_T_fieldS, grad_T_interpS = make_NCC(bS, (rS, r_shell, grad_T[shell_bool]), Nmax=NmaxS, vector=True)

#match_boundary(grad_T_fieldB, grad_T_fieldS)

if plot:
    plot_ncc_figure(r[sim_bool]/L, -grad_T[sim_bool], (rB.flatten(), rS.flatten()), (-grad_T_fieldB['g'][2][:1,:1,:].flatten(), -grad_T_fieldS['g'][2][:1,:1,:].flatten()), (NmaxB, NmaxS), ylabel=r"$-\nabla T$", fig_name="grad_T", out_dir=out_dir, log=True, r_int=r_inner.value)



### effective heating / (rho * T)
#Logic for smoothing heating profile at outer edge of CZ. Adjust outer edge of heating
#Since this is grid-locked, this isn't really needed.
#heat_erf_width  = 0.04
#heat_erf_center = 0.9891
#amount_to_adjust = 0.95
#full_lum_above = np.sum((4*np.pi*r_ball**2*H_NCC[ball_bool]*np.gradient(r_ball))[r_ball > amount_to_adjust][:1,:1,:].flatten())
#approx_H = H_NCC[ball_bool][r_ball <= amount_to_adjust][:1,:1,:].flatten()[-1]*one_to_zero(r_ball, heat_erf_center, heat_erf_width)
#full_lum_approx = np.sum((4*np.pi*r_ball**2*approx_H*np.gradient(r_ball))[r_ball > amount_to_adjust][:1,:1,:].flatten())
#H_NCC_ball = H_NCC[ball_bool]
#H_NCC_ball[r_ball > amount_to_adjust] = approx_H[r_ball > amount_to_adjust]
#print('outer', full_lum_above, full_lum_approx)
H_NCC = ((H_eff)  / H0) * (rho0*T0/rho/T)
NmaxB, NmaxS = true_NmaxB, 10
H_fieldB, H_interpB = make_NCC(bB, (rB, r_ball, H_NCC[ball_bool]), Nmax=NmaxB)
H_fieldS, H_interpS = make_NCC(bS, (rS, r_shell, H_NCC[shell_bool]), Nmax=NmaxS)
if plot:
    plot_ncc_figure(r[sim_bool]/L, H_NCC[sim_bool], (rB.flatten(), rS.flatten()), (H_fieldB['g'][:1,:1,:].flatten(), H_fieldS['g'][:1,:1,:].flatten()), (NmaxB, NmaxS), ylabel=r"$H$", fig_name="heating", out_dir=out_dir, log=False, r_int=r_inner.value)


### entropy gradient
transition_point = 1.03
width = 0.04
center =  transition_point - 0.5*width
width *= (L_CZ/L).value
center *= (L_CZ/L).value

#Build a nice function for our basis in the ball
grad_s_smooth = np.copy(grad_s)
flat_value  = np.interp(transition_point, r/L, grad_s)
grad_s_smooth[r/L < transition_point] = flat_value

NmaxB, NmaxS = 31, 62
NmaxB_after = true_NmaxB - 1
grad_s_fieldB, grad_s_interpB = make_NCC(bB, (rB, r_ball, (grad_s_smooth*L/s_c)[ball_bool]), Nmax=NmaxB, vector=True)
grad_s_interpB = np.interp(rB, r_ball, (grad_s*L/s_c)[ball_bool])
grad_s_fieldS, grad_s_interpS = make_NCC(bS, (rS, r_shell, (grad_s*L/s_c)[shell_bool]), Nmax=NmaxS, vector=True)
#match_boundary(grad_s_fieldB, grad_s_fieldS, adjust_ball=True)
grad_s_fieldB['g'][2] *= zero_to_one(rB, center, width=width)
grad_s_fieldB['c'][:,:,:,NmaxB_after:] = 0


#plt.figure()
#plt.plot(rB.flatten(), grad_s_fieldB['g'][2,0,0,:])
#plt.plot(rS.flatten(), grad_s_fieldS['g'][2,0,0,:])
#plt.yscale('log')
#plt.show()

if plot:
    plot_ncc_figure(r[sim_bool]/L, (grad_s*L/s_c)[sim_bool], (rB.flatten(), rS.flatten()), (grad_s_fieldB['g'][2][:1,:1,:].flatten(), grad_s_fieldS['g'][2][:1,:1,:].flatten()), (NmaxB_after, NmaxS), ylabel=r"$\nabla s$", fig_name="grad_s", out_dir=out_dir, log=True, r_int=r_inner.value)

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

    f['rS']          = rS
    f['TS']          = T_fieldS['g'][:1,:1,:]
    f['grad_TS']     = grad_T_fieldS['g'][:,:1,:1,:]
    f['H_effS']      = H_fieldS['g'][:1,:1,:]
    f['ln_ρS']       = ln_rho_fieldS['g'][:1,:1,:]
    f['ln_TS']       = ln_T_fieldS['g'][:1,:1,:]
    f['grad_ln_TS']  = grad_ln_T_fieldS['g'][:,:1,:1,:]
    f['grad_ln_ρS']  = grad_ln_rho_fieldS['g'][:,:1,:1,:]
    f['grad_s0S']    = grad_s_fieldS['g'][:,:1,:1,:]
    f['inv_Pe_radS'] = inv_Pe_rad_fieldS['g'][:1,:1,:]
    f['grad_inv_Pe_radS'] = grad_inv_Pe_S['g'][:,:1,:1,:]

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
    f['max_dt'] = max_dt
    f['s_c'] = s_c
    f['N2max_ball'] = N2max_ball
    f['N2max_shell'] = N2max_shell
    f['N2max'] = np.max((N2max_ball.value, N2max_shell.value))
    f['N2plateau'] = N2plateau
    f['cp_surf'] = cp_surf
    f['r_mesa'] = r
    f['N2_mesa'] = N2
    f['S1_mesa'] = lamb_freq(1)
    f['g_mesa'] = g 
    f['cp_mesa'] = cp
print(tau, tau/60/60/24)
