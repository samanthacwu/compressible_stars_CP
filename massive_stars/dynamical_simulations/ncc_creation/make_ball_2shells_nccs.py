"""
Turns a MESA .data file of a massive star into a .h5 file of NCCs for a d3 run.
There is a ball basis, and two shell bases.

Usage:
    make_ball_2shells_nccs.py [options]

Options:
    --Re=<R>          simulation reynolds/peclet number [default: 4e3]
    --NB=<N>          Maximum radial degrees of freedom (ball) [default: 128]
    --NS1=<N>          Maximum radial degrees of freedom (shell) [default: 64]
    --NS2=<N>          Maximum radial degrees of freedom (shell) [default: 16]
    --file=<f>        Path to MESA log file [default: ../../mesa_models/zams_15Msol/LOGS/profile47.data]
    --out_dir=<d>     output directory [default: nccs_15msol]
    --dealias=<n>     Radial dealiasing factor of simulation [default: 1.5]
    --ncc_cutoff=<f>  NCC coefficient magnitude cutoff [default: 1e-6]

    --no_plot         If flagged, don't output plots
"""
import os, sys
from collections import OrderedDict

import numpy as np
import h5py
from mpi4py import MPI
import mesa_reader as mr
import matplotlib.pyplot as plt
import dedalus.public as d3
from dedalus.core import field
from docopt import docopt

from astropy import units as u
from astropy import constants
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import scipy.integrate as si
from numpy.polynomial import Chebyshev as Pfit

#Import parent directory and anelastic_functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anelastic_functions import make_bases

args = docopt(__doc__)

NCC_CUTOFF = float(args['--ncc_cutoff'])
PLOT = not(args['--no_plot'])
SMOOTH_H = True

### Function definitions
from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

def plot_ncc_figure(rvals, mesa_func, dedalus_vals, Ns, ylabel="", fig_name="", out_dir='.', zero_line=False, log=False, r_int=None, ylim=None, axhline=None):
    """ Plots up a figure to compare a dedalus field to the MESA field it's based on. """
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    if zero_line:
        ax1.axhline(0, c='k', lw=0.5)

    if axhline is not None:
        ax1.axhline(axhline, c='k')

    for r, y in zip(rvals, dedalus_vals):
        mesa_y = mesa_func(r)
        ax1.plot(r, mesa_y, label='mesa', c='k', lw=3)
        ax1.plot(r, y, label='dedalus', c='red')

        diff = np.abs(1 - mesa_y/y)
        ax2.plot(r, diff)

    ax1.legend(loc='best')
    ax1.set_xlabel('radius/L', labelpad=-3)
    ax1.set_ylabel('{}'.format(ylabel))
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')
    if log:
        ax1.set_yscale('log')
    if ylim is not None:
        ax1.set_ylim(ylim)

    ax2.axhline(1e-1, c='k', lw=0.5)
    ax2.axhline(1e-2, c='k', lw=0.5)
    ax2.axhline(1e-3, c='k', lw=0.5)
    ax2.set_ylabel('abs(1 - mesa/dedalus)')
    ax2.set_xlabel('radius/L')
    ax2.set_yscale('log')

    ax2.set_ylim(1e-4, 1)
    fig.suptitle('coeff bandwidth = {}, {}; cutoff = {:e}'.format(Ns[0], Ns[1], NCC_CUTOFF))
    if r_int is not None:
        for ax in [ax1, ax2]:
            for rval in r_int:
                ax.axvline(rval, c='k')
    fig.savefig('{:s}/{}.png'.format(out_dir, fig_name), bbox_inches='tight', dpi=200)

def make_NCC(basis, dist, interp_func, Nmax=32, vector=False, grid_only=False):
    scales = (1, 1, Nmax/basis.radial_basis.radial_size)
    rvals = basis.global_grid_radius(scales[2])
    if vector:
        this_field = dist.VectorField(c, bases=basis)
        this_field.change_scales(scales)
        this_field['g'][2] = interp_func(rvals)
    else:
        this_field = dist.Field(bases=basis)
        this_field.change_scales(scales)
        this_field['g'] = interp_func(rvals)
    if not grid_only:
        this_field['c'][np.abs(this_field['c']) < NCC_CUTOFF] = 0
    this_field.change_scales(basis.dealias)
    return this_field



### Read in command line args & generate output path & file
nrB = NmaxB = int(args['--NB'])
nrS1 = NmaxS = int(args['--NS1'])
nrS2 = NmaxS = int(args['--NS2'])
dealias = float(args['--dealias'])
simulation_Re = float(args['--Re'])
read_file = args['--file']
filename = read_file.split('/LOGS/')[-1]
out_dir  = args['--out_dir'] + '/'
out_file = '{:s}/ball_2shells_nccs_B-{}_S1-{}_S2-{}_Re{}_de{}_cutoff{}.h5'.format(out_dir, nrB, nrS1, nrS2, args['--Re'], args['--dealias'], NCC_CUTOFF)
print('saving output to {}'.format(out_file))
if not os.path.exists('{:s}'.format(out_dir)):
    os.mkdir('{:s}'.format(out_dir))

### Read MESA file
p = mr.MesaData(read_file)
mass           = (p.mass[::-1] * u.M_sun).cgs
r              = (p.radius[::-1] * u.R_sun).cgs
rho            = 10**p.logRho[::-1] * u.g / u.cm**3
P              = p.pressure[::-1] * u.g / u.cm / u.s**2
T              = p.temperature[::-1] * u.K
nablaT         = p.gradT[::-1] #dlnT/dlnP
nablaT_ad      = p.grada[::-1]
chiRho         = p.chiRho[::-1]
chiT           = p.chiT[::-1]
cp             = p.cp[::-1]  * u.erg / u.K / u.g
opacity        = p.opacity[::-1] * (u.cm**2 / u.g)
Luminosity     = (p.luminosity[::-1] * u.L_sun).cgs
conv_L_div_L   = p.lum_conv_div_L[::-1]
csound         = p.csound[::-1] * u.cm / u.s
N2             = p.brunt_N2[::-1] / u.s**2
N2_structure   = p.brunt_N2_structure_term[::-1] / u.s**2
N2_composition = p.brunt_N2_composition_term[::-1] / u.s**2
eps_nuc        = p.eps_nuc[::-1] * u.erg / u.g / u.s
lamb_freq = lambda ell : np.sqrt(ell*(ell + 1)) * csound/r

R_star = (p.photosphere_r * u.R_sun).cgs

#Put all MESA fields into cgs and calculate secondary MESA fields
g               = constants.G.cgs*mass/r**2
dlogPdr         = -rho*g/P
gamma1          = dlogPdr/(-g/csound**2)
dlogrhodr       = dlogPdr*(chiT/chiRho)*(nablaT_ad - nablaT) - g/csound**2
dlogTdr         = dlogPdr*(nablaT)
grad_s          = cp*N2/g #entropy gradient, for NCC, includes composition terms
L_conv          = conv_L_div_L*Luminosity
dTdr            = (T)*dlogTdr

#True calculation of rad_diff, rad_cond
#rad_diff        = (16 * constants.sigma_sb.cgs * T**3 / (3 * rho**2 * cp * opacity)).cgs
#rad_cond        = rho*cp*rad_diff

#Calculate k_rad using luminosities and smooth things.
k_rad = rad_cond = -(Luminosity - L_conv)/(4*np.pi*r**2*dTdr)
rad_diff        = k_rad / (rho * cp)


### Split up the domain
# Find edge of core cz
cz_bool = (L_conv.value > 1)*(mass < 0.9*mass[-1]) #rudimentary but works
core_index  = np.argmin(np.abs(mass - mass[cz_bool][-1]))
core_cz_radius = r[core_index]
r_ball_MESA   = r[core_index]*1.1 #outer radius of BallBasis; inner radius of SphericalShellBasis

# Specify fraction of total star to simulate
fracStar   = 0.95 #Simulate this much of the star, from r = 0 to r = R_*
r_S1_frac = 0.95
r_S2_MESA    = fracStar*R_star
r_S1_MESA    = r_S1_frac*r_S2_MESA
print('fraction of FULL star simulated: {}, up to r={:.3e}'.format(fracStar, r_S2_MESA))

#Set things up to slice out the star appropriately
ball_bool     = r <= r_ball_MESA
shell_bool    = (r > r_ball_MESA)*(r <= r_S2_MESA)
S1_bool       = (r > r_ball_MESA)*(r <= r_S1_MESA)
S2_bool       = (r > r_S1_MESA)*(r <= r_S2_MESA)
sim_bool      = r <= r_S2_MESA

# Calculate heating function
# Goal: H_eff= np.gradient(L_conv,r, edge_order=1)/(4*np.pi*r**2) # Heating, for ncc, H = rho*eps - portion carried by radiation
# (1/4pir^2) dL_conv/dr = rho * eps + (1/r^2)d/dr (r^2 k_rad dT/dr) -> chain rule
eo=2
H_eff = (1/(4*np.pi*r**2))*np.gradient(Luminosity, r, edge_order=eo) + 2*k_rad*dTdr/r + dTdr*np.gradient(k_rad, r, edge_order=eo) + k_rad*np.gradient(dTdr, r, edge_order=eo)
H_eff_secondary = rho*eps_nuc + 2*k_rad*dTdr/r + dTdr*np.gradient(k_rad, r, edge_order=eo) + k_rad*np.gradient(dTdr, r, edge_order=eo)
H_eff[:2] = H_eff_secondary[:2]

sim_H_eff = np.copy(H_eff)
L_conv_sim = np.zeros_like(L_conv)
L_eps = np.zeros_like(Luminosity)
for i in range(L_conv_sim.shape[0]):
    L_conv_sim[i] = np.trapz((4*np.pi*r**2*sim_H_eff)[:1+i], r[:1+i])
    L_eps[i] = np.trapz((4*np.pi*r**2*rho*eps_nuc)[:i+1], r[:i+1])
L_excess = L_conv_sim[-5] - Luminosity[-5]

#construct internal heating field
if SMOOTH_H:
    #smooth CZ-RZ transition
    L_conv_sim *= one_to_zero(r, 0.95*core_cz_radius, width=0.15*core_cz_radius)
    L_conv_sim *= one_to_zero(r, 0.95*core_cz_radius, width=0.05*core_cz_radius)

    transition_region = (r > 0.5*core_cz_radius)
    sim_H_eff[transition_region] = ((1/(4*np.pi*r**2))*np.gradient(L_conv_sim, r, edge_order=eo))[transition_region]

#    plt.figure()
#    plt.axhline(0, c='k')
#    plt.plot(r, L_conv)
#    plt.plot(r, L_conv_sim, c='k', ls='--')
#    plt.figure()
#    plt.plot(r, H_eff)
#    plt.plot(r, sim_H_eff, ls='--', c='k')
#    plt.show()
else:
    sim_H_eff = H_eff

#Nondimensionalization
L_CZ    = core_cz_radius
L_nd    = L_CZ
#L_nd    = r_S2_MESA - r_ball_MESA
T_nd    = T[0]
m_nd    = rho[0] * L_nd**3
H0      = (rho*eps_nuc)[0]
tau_nd  = ((H0*L_nd/m_nd)**(-1/3)).cgs
rho_nd  = m_nd/L_nd**3
u_nd    = L_nd/tau_nd
s_nd    = L_nd**2 / tau_nd**2 / T_nd
rad_diff_nd = inv_Pe_rad = rad_diff * (tau_nd / L_nd**2)
print('Nondimensionalization: L_nd = {:.2e}, T_nd = {:.2e}, m_nd = {:.2e}, tau_nd = {:.2e}'.format(L_nd, T_nd, m_nd, tau_nd))
print('m_nd/M_\odot: {:.3f}'.format((m_nd/constants.M_sun).cgs))

#Central values
rho_r0    = rho[0]
P_r0      = P[0]
T_r0      = T[0]
cp_r0     = cp[0]
gamma1_r0  = gamma1[0]
Ma2_r0 = (u_nd**2 / ((gamma1_r0-1)*cp_r0*T_r0)).cgs
print('estimated mach number: {:.3e}'.format(np.sqrt(Ma2_r0)))

cp_surf = cp[S2_bool][-1]

#MESA radial values, in simulation units
r_ball = (r_ball_MESA/L_nd).value
r_S1 = (r_S1_MESA/L_nd).value
r_S2 = (r_S2_MESA/L_nd).value
r_nd = (r/L_nd).cgs

### entropy gradient
grad_s_transition_point = 1.02
grad_s_width = 0.02
grad_s_center =  grad_s_transition_point - 0.5*grad_s_width
grad_s_width *= (L_CZ/L_nd).value
grad_s_center *= (L_CZ/L_nd).value

#Build a nice function for our basis in the ball
grad_s_smooth = np.copy(grad_s)
flat_value  = np.interp(grad_s_transition_point, r/L_nd, grad_s)
grad_s_smooth[r/L_nd < grad_s_transition_point] = flat_value




# Get some timestepping & wave frequency info
N2max_ball = N2[ball_bool].max()
N2max_shell = N2[shell_bool].max()
shell_points = len(N2[shell_bool])
N2plateau = np.median(N2[int(shell_points*0.25):int(shell_points*0.75)])
f_nyq_ball  = np.sqrt(N2max_ball)/(2*np.pi)
f_nyq_shell = np.sqrt(N2max_shell)/(2*np.pi)
f_nyq    = 2*np.max((f_nyq_ball*tau_nd, f_nyq_shell*tau_nd))
nyq_dt   = (1/f_nyq) 

kepler_tau     = 30*60*u.s
max_dt_kepler  = kepler_tau/tau_nd

max_dt = max_dt_kepler
print('needed nyq_dt is {} s / {} % of a heating time (Kepler 30 min is {} %) '.format(nyq_dt*tau_nd, nyq_dt*100, max_dt_kepler*100))

### Make dedalus domain and bases
resolutions = ((8, 4, nrB), (8, 4, nrS1), (8, 4, nrS2))
stitch_radii = (r_ball, r_S1)
dtype=np.float64
mesh=None
c, d, bases, bases_keys = make_bases(resolutions, stitch_radii, r_S2, dealias=(1,1,dealias), dtype=dtype, mesh=mesh)
dedalus_r = OrderedDict()
for bn in bases.keys():
    phi, theta, r_vals = bases[bn].global_grids((1, 1, dealias))
    dedalus_r[bn] = r_vals

rvals_B = dedalus_r['B']
rvals_S1 = dedalus_r['S1']
rvals_S2 = dedalus_r['S2']

#construct rad_diff_profile
#sim_rad_diff = np.copy(rad_diff_nd)
#diff_transition = r_nd[sim_rad_diff > 1/simulation_Re][0]
#sim_rad_diff[:] = (1/simulation_Re)*one_to_zero(r_nd, diff_transition*1.05, width=0.02*diff_transition)\
#                + rad_diff_nd*zero_to_one(r_nd, diff_transition*0.95, width=0.1*diff_transition)
sim_rad_diff = np.copy(rad_diff_nd) + 1/simulation_Re

ncc_dict = OrderedDict()
for ncc in ['ln_rho', 'grad_ln_rho', 'ln_T', 'grad_ln_T', 'T', 'grad_T', 'H', 'grad_s', 'chi_rad', 'grad_chi_rad']:
    ncc_dict[ncc] = OrderedDict()
    for bn in bases.keys():
        ncc_dict[ncc]['Nmax_{}'.format(bn)] = 32
        ncc_dict[ncc]['field_{}'.format(bn)] = None
    ncc_dict[ncc]['Nmax_S2'.format(bn)] = 10
    ncc_dict[ncc]['vector'] = False
    ncc_dict[ncc]['grid_only'] = False 

ncc_dict['ln_rho']['interp_func'] = interp1d(r_nd, np.log(rho/rho_nd))
ncc_dict['ln_T']['interp_func'] = interp1d(r_nd, np.log(T/T_nd))
ncc_dict['grad_ln_rho']['interp_func'] = interp1d(r_nd, dlogTdr*L_nd)
ncc_dict['grad_ln_T']['interp_func'] = interp1d(r_nd, dlogTdr*L_nd)
ncc_dict['T']['interp_func'] = interp1d(r_nd, T/T_nd)
ncc_dict['grad_T']['interp_func'] = interp1d(r_nd, (L_nd/T_nd)*dTdr)
ncc_dict['H']['interp_func'] = interp1d(r_nd, ( sim_H_eff/(rho*T) ) * (rho_nd*T_nd/H0))
ncc_dict['grad_s']['interp_func'] = interp1d(r_nd, (L_nd/s_nd) * grad_s_smooth)

ncc_dict['chi_rad']['interp_func'] = interp1d(r_nd, sim_rad_diff)
ncc_dict['grad_chi_rad']['interp_func'] = interp1d(r_nd, np.gradient(rad_diff_nd, r_nd))

ncc_dict['grad_ln_rho']['vector'] = True
ncc_dict['grad_ln_T']['vector'] = True
ncc_dict['grad_T']['vector'] = True
ncc_dict['grad_s']['vector'] = True
ncc_dict['grad_chi_rad']['vector'] = True

ncc_dict['ln_T']['Nmax_B'] = 16
ncc_dict['grad_ln_T']['Nmax_B'] = 17
ncc_dict['H']['Nmax_B'] = 60
ncc_dict['H']['Nmax_S1'] = 2

ncc_dict['chi_rad']['Nmax_B'] = 1
ncc_dict['chi_rad']['Nmax_S1'] = 20
ncc_dict['chi_rad']['Nmax_S2'] = 7

ncc_dict['H']['grid_only'] = True


for bn, basis in bases.items():
    rvals = dedalus_r[bn]
    for ncc in ncc_dict.keys():
        interp_func = ncc_dict[ncc]['interp_func']
        Nmax = ncc_dict[ncc]['Nmax_{}'.format(bn)]
        vector = ncc_dict[ncc]['vector']
        grid_only = ncc_dict[ncc]['grid_only']
        ncc_dict[ncc]['field_{}'.format(bn)] = make_NCC(basis, d, interp_func, Nmax=Nmax, vector=vector, grid_only=grid_only)
        if ncc == 'grad_T':
            print(ncc_dict[ncc]['field_{}'.format(bn)]['g'][2,0,0,:])

    #Evaluate for grad chi rad
    ncc_dict['grad_chi_rad']['field_{}'.format(bn)]['g'] = d3.grad(ncc_dict['chi_rad']['field_{}'.format(bn)]).evaluate()['g']

#Further post-process work to make grad_S nice in the ball
NmaxB_after = resolutions[0][-1] - 1
ncc_dict['grad_s']['field_B']['g'][2] *= zero_to_one(rvals_B, grad_s_center, width=grad_s_width)
ncc_dict['grad_s']['field_B']['c'][:,:,:,NmaxB_after:] = 0

#Post-processing for grad chi rad - doesn't work great...
diff_transition = r_nd[sim_rad_diff > 1/simulation_Re][0].value
gradPe_S1_cutoff = 32
gradPe_S2_cutoff = 32
ncc_dict['grad_chi_rad']['field_S1']['g'][2,] *= zero_to_one(rvals_S1, diff_transition, width=(r_S2-r_ball)/10)
ncc_dict['grad_chi_rad']['field_S1']['c'][:,:,:,gradPe_S1_cutoff:] = 0
ncc_dict['grad_chi_rad']['field_S2']['g'][2,] *= zero_to_one(rvals_S2, diff_transition, width=(r_S2-r_ball)/10)
ncc_dict['grad_chi_rad']['field_S2']['c'][:,:,:,gradPe_S2_cutoff:] = 0

#plt.plot(rvals_S1.ravel(), ncc_dict['grad_chi_rad']['field_S1']['g'][2,0,0,:])
#plt.plot(rvals_S2.ravel(), ncc_dict['grad_chi_rad']['field_S2']['g'][2,0,0,:])
#plt.plot(r_nd, rad_diff_nd)
#plt.axhline(1/simulation_Re)
#plt.yscale('log')
#plt.show()

if PLOT:
    for ncc in ncc_dict.keys():
        axhline = None
        log = False
        ylim = None
        rvals = []
        dedalus_yvals = []
        nvals = []
        for bn, basis in bases.items():
            rvals.append(dedalus_r[bn].ravel())
            nvals.append(ncc_dict[ncc]['Nmax_{}'.format(bn)])
            if ncc_dict[ncc]['vector']:
                dedalus_yvals.append(ncc_dict[ncc]['field_{}'.format(bn)]['g'][2,0,0,:])
            else:
                dedalus_yvals.append(ncc_dict[ncc]['field_{}'.format(bn)]['g'][0,0,:])

        if ncc in ['T', 'grad_T', 'chi_rad', 'grad_chi_rad', 'grad_s']:
            print('log scale for ncc {}'.format(ncc))
            log = True
        if ncc == 'grad_s': 
            axhline = 1
        elif ncc in ['chi_rad', 'grad_chi_rad']:
            axhline = 1/simulation_Re

        interp_func = ncc_dict[ncc]['interp_func']
        if ncc == 'H':
            interp_func = interp1d(r_nd, ( one_to_zero(r_nd, 1.5*r_ball, width=0.05*r_ball)*H_eff/(rho*T) ) * (rho_nd*T_nd/H0) )
        elif ncc == 'grad_s':
            interp_func = interp1d(r_nd, (L_nd/s_nd) * grad_s)

        if ncc == 'grad_T':
            interp_func = lambda r: -ncc_dict[ncc]['interp_func'](r)
            ylabel='-{}'.format(ncc)
            for i in range(len(dedalus_yvals)):
                dedalus_yvals[i] *= -1
        else:
            ylabel = ncc

        plot_ncc_figure(rvals, interp_func, dedalus_yvals, nvals, \
                    ylabel=ylabel, fig_name=ncc, out_dir=out_dir, log=log, ylim=ylim, \
                    r_int=stitch_radii, axhline=axhline)


with h5py.File('{:s}'.format(out_file), 'w') as f:
    # Save output fields.
    # slicing preserves dimensionality
    for bn, basis in bases.items():
        f['r_{}'.format(bn)] = dedalus_r[bn]
        for ncc in ncc_dict.keys():
            this_field = ncc_dict[ncc]['field_{}'.format(bn)]
            if ncc_dict[ncc]['vector']:
                f['{}_{}'.format(ncc, bn)] = this_field['g'][:, :1,:1,:]
            else:
                f['{}_{}'.format(ncc, bn)] = this_field['g'][:1,:1,:]

    #Save properties of the star, with units.
    f['L_nd']   = L_nd
    f['L_nd'].attrs['units'] = str(L_nd.unit)
    f['rho_nd']  = rho_nd
    f['rho_nd'].attrs['units']  = str(rho_nd.unit)
    f['T_nd']  = T_nd
    f['T_nd'].attrs['units']  = str(T_nd.unit)
    f['tau_nd'] = tau_nd 
    f['tau_nd'].attrs['units'] = str(tau_nd.unit)
    f['m_nd'] = m_nd 
    f['m_nd'].attrs['units'] = str(m_nd.unit)
    f['s_nd'] = s_nd
    f['s_nd'].attrs['units'] = str(s_nd.unit)
    f['P_r0']  = P_r0
    f['P_r0'].attrs['units']  = str(P_r0.unit)
    f['H0']  = H0
    f['H0'].attrs['units']  = str(H0.unit)
    f['N2max_ball'] = N2max_ball
    f['N2max_ball'].attrs['units'] = str(N2max_ball.unit)
    f['N2max_shell'] = N2max_shell
    f['N2max_shell'].attrs['units'] = str(N2max_shell.unit)
    f['N2max'] = np.max((N2max_ball.value, N2max_shell.value))
    f['N2max'].attrs['units'] = str(N2max_ball.unit)
    f['N2plateau'] = N2plateau
    f['N2plateau'].attrs['units'] = str(N2plateau.unit)
    f['cp_surf'] = cp_surf
    f['cp_surf'].attrs['units'] = str(cp_surf.unit)
    f['r_mesa'] = r
    f['r_mesa'].attrs['units'] = str(r.unit)
    f['N2_mesa'] = N2
    f['N2_mesa'].attrs['units'] = str(N2.unit)
    f['S1_mesa'] = lamb_freq(1)
    f['S1_mesa'].attrs['units'] = str(lamb_freq(1).unit)
    f['g_mesa'] = g 
    f['g_mesa'].attrs['units'] = str(g.unit)
    f['cp_mesa'] = cp
    f['cp_mesa'].attrs['units'] = str(cp.unit)

    f['r_ball']   = r_ball
    f['r_outer']   = r_S2
    f['max_dt'] = max_dt
    f['Ma2_r0'] = Ma2_r0
    for k in ['r_ball', 'r_outer', 'max_dt', 'Ma2_r0']:
        f[k].attrs['units'] = 'dimensionless'
print('finished saving NCCs to {}'.format(out_file))


#fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8,4))
#
#for i in range(2):
#    for j in range(3):
#        axs[i][j].axvline(r_ball_MESA.value, c='k', lw=0.5)
#        axs[i][j].axvline(r_S2_MESA.value, c='k', lw=0.5)
#
#
#axs[0][0].plot(r, T)
#axs[0][0].plot(rvals_B.flatten()*L_nd, T_nd*T_fieldB['g'][0,0,:], c='k')
#axs[0][0].plot(rS.flatten()*L_nd, T_nd*T_fieldS['g'][0,0,:], c='k')
#axs[0][0].set_ylabel('T (K)')
#
#axs[0][1].plot(r, np.log(rho/rho_nd))
#axs[0][1].plot(rvals_B.flatten()*L_nd, ln_rho_fieldB['g'][0,0,:], c='k')
#axs[0][1].plot(rS.flatten()*L_nd, ln_rho_fieldS['g'][0,0,:], c='k')
#axs[0][1].set_ylabel(r'$\ln(\rho/\rho_{\rm{nd}})$')
#
#axs[1][0].plot(r, inv_Pe_rad)
#axs[1][0].plot(rvals_B.flatten()*L_nd, inv_Pe_rad_fieldB['g'][0,0,:], c='k')
#axs[1][0].plot(rS.flatten()*L_nd, inv_Pe_rad_fieldS['g'][0,0,:], c='k')
#axs[1][0].set_yscale('log')
#axs[1][0].set_ylabel(r'$\chi_{\rm{rad}}\,L_{\rm{nd}}^{-2}\,\tau_{\rm{nd}}$')
#
##axs[0][1].plot(r, np.gradient(inv_Pe_rad, r))
##axs[0][1].plot(rvals_B.flatten()*L_nd, grad_inv_Pe_B['g'][2,0,0,:]/L_nd, c='k')
##axs[0][1].plot(rS.flatten()*L_nd, grad_inv_Pe_S['g'][2,0,0,:]/L_nd, c='k')
##axs[0][1].set_yscale('log')
#
#
##axs[0][3].plot(r, np.log(T/T_nd))
##axs[0][3].plot(rvals_B.flatten()*L_nd, ln_T_fieldB['g'][0,0,:], c='k')
##axs[0][3].plot(rS.flatten()*L_nd, ln_T_fieldS['g'][0,0,:], c='k')
#
##axs[1][0].plot(r, grad_T*T_nd/L_nd)
##axs[1][0].plot(rvals_B.flatten()*L_nd, T_nd*grad_T_fieldB['g'][2,0,0,:]/L_nd, c='k')
##axs[1][0].plot(rS.flatten()*L_nd, T_nd*grad_T_fieldS['g'][2,0,0,:]/L_nd, c='k')
#
#
#
#axs[1][1].plot(r, H_eff/(rho*T), c='b')
#axs[1][1].plot(r, eps_nuc / T, c='r')
#axs[1][1].plot(rvals_B.flatten()*L_nd, (H0 / rho_nd / T_nd)*H_fieldB['g'][0,0,:], c='k')
#axs[1][1].plot(rS.flatten()*L_nd, (H0 / rho_nd / T_nd)*H_fieldS['g'][0,0,:], c='k')
#axs[1][1].set_ylim(-5e-4, 2e-3)
##axs[1][1].plot(rvals_B.flatten()*L_nd, -(H0 / rho_nd / T_nd)*H_fieldB['g'][0,0,:], c='k', ls='--')
##axs[1][1].plot(rS.flatten()*L_nd, -(H0 / rho_nd / T_nd)*H_fieldS['g'][0,0,:], c='k', ls='--')
##axs[1][1].set_yscale('log')
#axs[1][1].set_ylabel(r'$H/(\rho T)$ (units)')
#
#axs[1][2].plot(r, grad_s)
#axs[1][2].plot(rvals_B.flatten()*L_nd, (s_nd/L_nd)*grad_s_fieldB['g'][2,0,0,:], c='k')
#axs[1][2].plot(rS.flatten()*L_nd, (s_nd/L_nd)*grad_s_fieldS['g'][2,0,0,:], c='k')
#axs[1][2].set_yscale('log')
#axs[1][2].set_ylim(1e-6, 1e0)
#axs[1][2].set_ylabel(r'$\nabla s$ (erg$\,\rm{g}^{-1}\rm{K}^{-1}\rm{cm}^{-1}$)')
#
#
#
#plt.subplots_adjust(wspace=0.4, hspace=0.4)
#fig.savefig('dedalus_mesa_figure.png', dpi=200, bbox_inches='tight')
#
##Plot a propagation diagram
#plt.figure()
#plt.plot(r, np.sqrt(N2), label=r'$N$')
#plt.plot(r, lamb_freq(1), label=r'$S_1$')
#plt.plot(r, lamb_freq(10), label=r'$S_{10}$')
#plt.plot(r, lamb_freq(100), label=r'$S_{100}$')
#plt.xlim(0, r_S2_MESA.value)
#plt.xlabel('r (cm)')
#plt.ylabel('freq (1/s)')
#plt.yscale('log')
#plt.legend(loc='best')
#plt.savefig('{}/propagation_diagram.png'.format(out_dir), dpi=300, bbox_inches='tight')
#
#
