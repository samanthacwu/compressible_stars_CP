"""
Turns a MESA .data file of a massive star into a .h5 file of NCCs for a d3 run.

Usage:
    make_coreCZ_nccs.py [options]

Options:
    --NB=<N>        Maximum radial coefficients (ball) [default: 63]
    --NS=<N>        Maximum radial coefficients (shell) [default: 31]
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
NmaxB = int(args['--NB'])
NmaxS = int(args['--NS'])
read_file = args['--file']
filename = read_file.split('/LOGS/')[-1]
out_dir  = read_file.replace('/LOGS/', '_').replace('.data', '_ballShell_halfStar')
if args['--pre_log_folder'] != '':
    out_dir = '{:s}_{:s}'.format(args['--pre_log_folder'], out_dir)
print('saving files to {}'.format(out_dir))
out_file = '{:s}/ballShell_nccs_B{}_S{}.h5'.format(out_dir, NmaxB, NmaxS)
if not os.path.exists('{:s}'.format(out_dir)):
    os.mkdir('{:s}'.format(out_dir))


p = mr.MesaData(read_file)


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
opacity = p.opacity[::-1] * (u.cm**2 / u.g)
mu = p.mu[::-1]
N2 = p.brunt_N2[::-1] / u.s**2
N2_structure   = p.brunt_N2_structure_term[::-1] / u.s**2
N2_composition = p.brunt_N2_composition_term[::-1] / u.s**2
Luminosity = p.luminosity[::-1] * u.L_sun
Luminosity = Luminosity.to('erg/s')
L_conv = p.conv_L_div_L[::-1]*Luminosity
csound = p.csound[::-1] * u.cm / u.s

rad_diff = 16 * constants.sigma_sb.cgs * T**3 / (3 * rho**2 * cp * opacity)
rad_diff = rad_diff.cgs


cgs_G = constants.G.to('cm^3/(g*s^2)')
g = cgs_G*mass/r**2
gamma = cp/cv

#Thermo gradients
chiRho  = p.chiRho[::-1]
chiT    = p.chiT[::-1]
nablaT =  p.gradT[::-1]
nablaT_ad = p.grada[::-1]
dlogPdr = -rho*g/P
gamma1  = dlogPdr/(-g/csound**2)
dlogrhodr = dlogPdr*(chiT/chiRho)*(nablaT_ad - nablaT) - g/csound**2
dlogTdr   = dlogPdr*(nablaT)
N2_therm_approx = g*(dlogPdr/gamma1 - dlogrhodr)
grad_s = cp*N2/g #includes composition terms
print(chiRho, chiT)



# Heating
#H = rho * eps
#C = (np.gradient(Luminosity-L_conv,r)/(4*np.pi*r**2))
#H_eff = H - C
H_eff = (np.gradient(L_conv,r)/(4*np.pi*r**2))
H_eff[0] = H_eff[1] #make gradient 0 at core, remove weird artifacts.

H0 = H_eff[0]
H_NCC = ((H_eff)  / H0)


#Find edge of core cz
cz_bool = (L_conv.value > 1)*(mass < 0.9*mass[-1])
core_cz_bound = mass[cz_bool][-1] # 0.9 to avoid some of the cz->rz transition region.
bound_ind = np.argmin(np.abs(mass - core_cz_bound))


#Nondimensionalization
halfStar_r = r[-1]/2
L = L_CZ  = r[bound_ind]
g0 = g[bound_ind] 
rho0 = rho[0]
P0 = P[0]
T0 = T[0]
cp0 = cp[0]
gamma0 = gamma[0]
tau = (H0/L**2/rho0)**(-1/3)
tau = tau.cgs
print('one time unit is {:.2e}'.format(tau))
u_H = L/tau

Pe = u_H*L/rad_diff

#fig = plt.figure()
#ax1 = fig.add_subplot(2,1,1)
#plt.plot(r/L, np.abs(N2), label='brunt2_full', c='k', lw=3)
#plt.plot(r/L, np.abs(N2_structure), label='brunt2_structure', c='r')
#plt.plot(r/L, np.abs(N2_composition), label='brunt2_comp', c='orange')
#plt.yscale('log')
#plt.ylabel(r'$N^2$')
#plt.legend(loc='best')
#ax2 = fig.add_subplot(2,1,2)
#plt.yscale('log')
#plt.plot(r/L, np.abs(N2), label='brunt2_full', c='k', lw=3)
#plt.plot(r/L, np.abs(N2_structure), label='brunt2_structure', c='r')
#plt.plot(r/L, np.abs(N2_composition), label='brunt2_comp', c='orange')
##plt.plot(r, np.abs(N2_therm_approx), label='brunt2_structure', c='r')
##plt.plot(r, np.abs(N2 - N2_therm_approx), label='brunt2_comp', c='orange', lw=0.5)
#plt.xlim(3/4, 5/4)
#plt.suptitle(filename)
#plt.ylabel(r'$N^2$')
#plt.xlabel(r'radius (cm)')
#plt.savefig('{:s}/{:s}_star_brunt_fig.png'.format(out_dir, filename.split('.data')[0]), dpi=200, bbox_inches='tight')
#
#
#
#
#fig = plt.figure()
#ax1 = fig.add_subplot(2,1,1)
#plt.plot(r, N2, label='brunt2_full', c='k', lw=3)
#plt.plot(r, N2_structure, label='brunt2_structure', c='r')
#plt.plot(r, N2_composition, label='brunt2_comp', c='orange')
#plt.yscale('log')
#plt.ylabel(r'$N^2$')
#plt.legend(loc='best')
#ax2 = fig.add_subplot(2,1,2)
#plt.yscale('log')
#plt.axhline(1e3, c='k', lw=0.5)
#plt.axhline(1e4, c='k', lw=0.5)
#plt.axhline(1e5, c='k', lw=0.5)
#plt.plot(r, Pe, c='k', lw=3)
#plt.suptitle(filename)
#plt.ylabel(r'Pe')
#plt.xlabel(r'radius (cm)')
#plt.savefig('{:s}/{:s}_brunt_and_Pe_fig.png'.format(out_dir, filename.split('.data')[0]), dpi=200, bbox_inches='tight')



Ma2 = u_H**2 / ((gamma0-1)*cp0*T0)
s_c = Ma2*(gamma0-1)*cp0


r_inner    = r[cz_bool][-1]*1.1/L
r_outer    = halfStar_r/L
ball_bool  = r <= r_inner*L
shell_bool = (r > r_inner*L)*(r <= r_outer*L)

#rz_bool = (r > r[cz_bool][-1])*(r <= halfStar_r)

r_ball = r[ball_bool]/L
r_shell = r[shell_bool]/L

c = coords.SphericalCoordinates('φ', 'θ', 'r')
d = distributor.Distributor((c,), mesh=None)
bB = basis.BallBasis(c, (1, 1, NmaxB+1), radius=r_inner, dtype=np.float64, dealias=(1, 1, 1))
bS = basis.SphericalShellBasis(c, (1, 1, NmaxS+1), radii=(r_inner, r_outer), dtype=np.float64, dealias=(1, 1, 1))
φB, θB, rB = bB.global_grids((1, 1, 1))
φS, θS, rS = bS.global_grids((1, 1, 1))

grad = lambda A: operators.Gradient(A, c)
dot  = lambda A, B: arithmetic.DotProduct(A, B)


r_vec  = field.Field(dist=d, bases=(bB,), dtype=np.float64, tensorsig=(c,))
r_vec['g'][2,:] = 1


### Log Density (Ball)
N = 8
ln_rho_fieldB  = field.Field(dist=d, bases=(bB,), dtype=np.float64)
ln_rho = np.log(rho/rho0)[ball_bool]
ln_rho_interp = np.interp(rB, r_ball, ln_rho)
ln_rho_fieldB['g'] = ln_rho_interp
ln_rho_fieldB['c'][:, :, N:] = 0
if plot:
    plot_ncc_figure(rB.flatten(), (-1)+ln_rho_interp.flatten(), (-1)+ln_rho_fieldB['g'].flatten(), N, ylabel=r"$\ln\rho - 1$", fig_name="ln_rhoB", out_dir=out_dir)

N = 8
grad_ln_rho_fieldB  = field.Field(dist=d, bases=(bB,), tensorsig=(c,), dtype=np.float64)
grad_ln_rho_interp = np.interp(rB, r_ball, dlogrhodr[ball_bool]*L)
grad_ln_rho_interp2 = np.interp(rB, r_ball, np.gradient(ln_rho, r_ball))
grad_ln_rho_fieldB['g'][2] = grad_ln_rho_interp
grad_ln_rho_fieldB['c'][:,:,:,N:] = 0
if plot:
    plot_ncc_figure(rB.flatten(), grad_ln_rho_interp2.flatten(), grad_ln_rho_fieldB['g'][2].flatten(), N, ylabel=r"$\nabla\ln\rho$", fig_name="grad_ln_rhoB", out_dir=out_dir)


### Log Density (Shell)
N = 8
ln_rho_fieldS  = field.Field(dist=d, bases=(bS,), dtype=np.float64)
ln_rho = np.log(rho/rho0)[shell_bool]
ln_rho_interp = np.interp(rS, r_shell, ln_rho)
ln_rho_fieldS['g'] = ln_rho_interp
ln_rho_fieldS['c'][:, :, N:] = 0
if plot:
    plot_ncc_figure(rS.flatten(), (-1)+ln_rho_interp.flatten(), (-1)+ln_rho_fieldS['g'].flatten(), N, ylabel=r"$\ln\rho - 1$", fig_name="ln_rhoS", out_dir=out_dir)

N = 8
grad_ln_rho_fieldS  = field.Field(dist=d, bases=(bS,), tensorsig=(c,), dtype=np.float64)
grad_ln_rho_interp = np.interp(rS, r_shell, dlogrhodr[shell_bool]*L)
grad_ln_rho_fieldS['g'][2] = grad_ln_rho_interp
grad_ln_rho_fieldS['c'][:,:,:,N:] = 0
if plot:
    plot_ncc_figure(rS.flatten(), grad_ln_rho_interp.flatten(), grad_ln_rho_fieldS['g'][2].flatten(), N, ylabel=r"$\nabla\ln\rho$", fig_name="grad_ln_rhoS", out_dir=out_dir)



### Log Temperature (Ball) 
N = 8
ln_T_fieldB  = field.Field(dist=d, bases=(bB,), dtype=np.float64)
ln_T = np.log((T)/T0)
ln_T_interp = np.interp(rB, r_ball, ln_T[ball_bool])
ln_T_fieldB['g'] = ln_T_interp
ln_T_fieldB['c'][:, :, N:] = 0
if plot:
    plot_ncc_figure(rB.flatten(), (-1)+ln_T_interp.flatten(), (-1)+ln_T_fieldB['g'].flatten(), N, ylabel=r"$\ln(T) - 1$", fig_name="ln_TB", out_dir=out_dir)

N = 8
grad_ln_T_fieldB  = field.Field(dist=d, bases=(bB,), tensorsig=(c,), dtype=np.float64)
grad_ln_T_interp = np.interp(rB, r_ball, dlogTdr[ball_bool]*L)
grad_ln_T_fieldB['g'][2] = grad_ln_T_interp 
grad_ln_T_fieldB['c'][:, :, :, N:] = 0
if plot:
    plot_ncc_figure(rB.flatten(), grad_ln_T_interp.flatten(), grad_ln_T_fieldB['g'][2].flatten(), N, ylabel=r"$\nabla\ln(T)$", fig_name="grad_ln_TB", out_dir=out_dir)

### Log Temperature (Shell)
N = 8
ln_T_fieldS  = field.Field(dist=d, bases=(bS,), dtype=np.float64)
ln_T = np.log((T)/T0)
ln_T_interp = np.interp(rS, r_shell, ln_T[shell_bool])
ln_T_fieldS['g'] = ln_T_interp
ln_T_fieldS['c'][:, :, N:] = 0
if plot:
    plot_ncc_figure(rS.flatten(), (-1)+ln_T_interp.flatten(), (-1)+ln_T_fieldS['g'].flatten(), N, ylabel=r"$\ln(T) - 1$", fig_name="ln_TS", out_dir=out_dir)

N = 8
grad_ln_T_fieldS  = field.Field(dist=d, bases=(bS,), tensorsig=(c,), dtype=np.float64)
grad_ln_T_interp = np.interp(rS, r_shell, dlogTdr[shell_bool]*L)
grad_ln_T_fieldS['g'][2] = grad_ln_T_interp 
grad_ln_T_fieldS['c'][:, :, :, N:] = 0
if plot:
    plot_ncc_figure(rS.flatten(), grad_ln_T_interp.flatten(), grad_ln_T_fieldS['g'][2].flatten(), N, ylabel=r"$\nabla\ln(T)$", fig_name="grad_ln_TS", out_dir=out_dir)

### Temperature (Ball) 
T_nondim = (T) / T0
plt.figure()
likeT = np.copy(T_nondim)[ball_bool]
#likeT[r_ball < 0.05] = 1
width = 0.05
Toutside   = likeT*zero_to_one(r_ball.flatten(), 0.1, width=width)
TnearCore  = one_to_zero(r_ball.flatten(), 0.1, width=width)
niceT      = TnearCore + Toutside

plt.plot(r_ball.flatten(), T_nondim[ball_bool].flatten())
plt.plot(r_ball.flatten(), Toutside)
plt.plot(r_ball.flatten(), TnearCore)
plt.plot(r_ball.flatten(), niceT.flatten())
plt.show()




N = int((NmaxB+1))-10
T_fieldB = field.Field(dist=d, bases=(bB,), dtype=np.float64)
T_interp = np.interp(rB, r_ball, niceT)
T_interpReal = np.interp(rB, r_ball, T_nondim[ball_bool])
T_fieldB['g'] = T_interp
T_fieldB['c'][:, :, N:] = 0
if plot:
    plot_ncc_figure(rB.flatten(), T_interpReal.flatten(), T_fieldB['g'].flatten(), N, ylabel=r"$T/T_c$", fig_name="TB", out_dir=out_dir)

### Temperature (Shell)
N = 63
T_fieldS = field.Field(dist=d, bases=(bS,), dtype=np.float64)
T_interp = np.interp(rS, r_shell, T_nondim[shell_bool])
T_fieldS['g'] = T_interp
T_fieldS['c'][:, :, N:] = 0
if plot:
    plot_ncc_figure(rS.flatten(), T_interp.flatten(), T_fieldS['g'].flatten(), N, ylabel=r"$T/T_c$", fig_name="TS", out_dir=out_dir)


### effective heating / (rho * T)

#Adjust outer edge of heating
heat_erf_width  = 0.04
heat_erf_center = 0.9891
amount_to_adjust = 0.95
full_lum_above = np.sum((4*np.pi*r_ball**2*H_NCC[ball_bool]*np.gradient(r_ball))[r_ball > amount_to_adjust].flatten())
approx_H = H_NCC[ball_bool][r_ball <= amount_to_adjust].flatten()[-1]*one_to_zero(r_ball, heat_erf_center, heat_erf_width)
full_lum_approx = np.sum((4*np.pi*r_ball**2*approx_H*np.gradient(r_ball))[r_ball > amount_to_adjust].flatten())
H_NCC_ball = H_NCC[ball_bool]
H_NCC_ball[r_ball > amount_to_adjust] = approx_H[r_ball > amount_to_adjust]
print('outer', full_lum_above, full_lum_approx)

#Adjust so gradient is zero near center
#heat_erf_width  = 0.01
#heat_erf_center = 0.03
amount_to_adjust = 0.05
full_lum_above = np.sum((4*np.pi*r_ball**2*H_NCC[ball_bool]*np.gradient(r_ball))[r_ball <= amount_to_adjust].flatten())
approx_H = H_NCC[ball_bool][r_ball > amount_to_adjust].flatten()[0]*np.ones_like(r_ball)
full_lum_approx = np.sum((4*np.pi*r_ball**2*approx_H*np.gradient(r_ball))[r_ball <= amount_to_adjust].flatten())
approx_H *= full_lum_above/full_lum_approx
full_lum_approx = np.sum((4*np.pi*r_ball**2*approx_H*np.gradient(r_ball))[r_ball <= amount_to_adjust].flatten())
H_NCC_ball[r_ball <= amount_to_adjust] = approx_H[r_ball <= amount_to_adjust]
print('inner', full_lum_above, full_lum_approx)





N = int((NmaxB+1)/2)
H_fieldB = field.Field(dist=d, bases=(bB,), dtype=np.float64)
H_interp = np.interp(rB, r_ball, H_NCC_ball)
H_interp_plot = np.interp(rB, r_ball, H_NCC_ball * (rho0*T0/rho/T)[ball_bool])
H_fieldB['g'] = H_interp / T_fieldB['g'] / np.exp(ln_rho_fieldB['g'])
H_fieldB['c'][:,:,N:] = 0
if plot:
    plot_ncc_figure(rB.flatten(), H_interp_plot.flatten(), H_fieldB['g'].flatten(), N, ylabel=r"$(H_{eff}/(\rho c_p T))$ (nondimensional)", fig_name="H_effB", out_dir=out_dir, zero_line=True)

plt.figure()
#plt.plot(rB.flatten(), (H_NCC_ball[5] * (rho0*T0/rho/T)[5]*one_to_zero(rB, 0.05, width=0.01)).flatten())
plt.plot(r_ball.flatten(), (H_NCC_ball * (rho0*T0/rho/T)[ball_bool]).flatten())
plt.plot(rB.flatten(), H_fieldB['g'][0,0,:])
#plt.plot(r_ball.flatten(), (approx_H*(rho0*T0/rho/T)[ball_bool].flatten()))


N = 1
H_fieldS = field.Field(dist=d, bases=(bS,), dtype=np.float64)
H_interp = np.interp(rS, r_shell, H_NCC[shell_bool])
H_interp_plot = np.interp(rS, r_shell, (H_NCC * rho0*T0/rho/T)[shell_bool])
H_fieldS['g'] = 0#H_interp / T_fieldS['g'] / np.exp(ln_rho_fieldS['g'])
H_fieldS['c'][:,:,N:] = 0
if plot:
    plot_ncc_figure(rS.flatten(), H_interp_plot.flatten(), H_fieldS['g'].flatten(), N, ylabel=r"$(H_{eff}/(\rho c_p T))$ (nondimensional)", fig_name="H_eff", out_dir=out_dir, zero_line=True)


transition_point = 1.03
if NmaxB == 63:
    width = 0.06
    N = 32
    N_after = -1
elif NmaxB == 127:
    width = 0.06
    N = 36
    N_after = -1
elif NmaxB == 255:
    width = 0.015
    N = 47
    N_after = -128
    transition_point=1.02
elif NmaxB == 383:
    width = 0.01
    N = 47
    N_after = -192
    transition_point=1.015
elif NmaxB == 511:
    width = 0.01
    N = 47
    N_after = -255
    transition_point=1.0125
center =  transition_point - 1.5*width
width *= (L_CZ/L).value
center *= (L_CZ/L).value

### entropy gradient (Ball)
N = 32
grad_s_fieldB  = field.Field(dist=d, bases=(bB,), dtype=np.float64, tensorsig=(c,))
grad_s_interp = np.interp(rB, r_ball, grad_s[ball_bool]*L/s_c)
grad_s_base = np.copy(grad_s_interp)
flat_value  = np.interp(transition_point, r/L, grad_s*L/s_c)
grad_s_base[:,:,(rB).flatten() < transition_point] = flat_value
grad_s_fieldB['g'][2] = grad_s_base
grad_s_fieldB['c'][:,:,:,N:] = 0
grad_s_fieldB['g'][2] *= zero_to_one(rB, center, width=width).value
grad_s_fieldB['c'][:,:,:,N_after:] = 0
grad_s_fieldB['g'][2] *= zero_to_one(rB, 0.4, width=0.1).value
grad_s_fieldB['c'][:,:,:,-1:] = 0
if plot:
    plot_ncc_figure(rB.flatten(), grad_s_interp.flatten(), grad_s_fieldB['g'][2].flatten(), N, ylabel=r"$L(\nabla s/s_c)$", fig_name="grad_sB", out_dir=out_dir, zero_line=True)


### entropy gradient (Shell)
N = 48
grad_s_fieldS  = field.Field(dist=d, bases=(bS,), dtype=np.float64, tensorsig=(c,))
grad_s_interp = np.interp(rS, r_shell, grad_s[shell_bool]*L/s_c)
grad_s_fieldS['g'][2] = grad_s_interp
grad_s_fieldS['c'][:,:,:,N:] = 0
#grad_s_field['g'][2] *= zero_to_one(rB, 0.995*(L_CZ/L).value, width=width*(L_CZ/L).value)
#grad_s_field['c'][:,:,:,N_after:] = 0
if plot:
    plot_ncc_figure(rS.flatten(), grad_s_interp.flatten(), grad_s_fieldS['g'][2].flatten(), N, ylabel=r"$L(\nabla s/s_c)$", fig_name="grad_sS", out_dir=out_dir, zero_line=True)



plt.figure()
plt.plot(r_ball, grad_s[ball_bool]*L/s_c, c='k')
plt.plot(r_shell, grad_s[shell_bool]*L/s_c, c='k')
plt.plot(rS.flatten(), np.abs(grad_s_fieldS['g'][2].flatten()))
plt.plot(rB.flatten(), np.abs(grad_s_fieldB['g'][2].flatten()))
plt.yscale('log')
plt.show()

with h5py.File('{:s}'.format(out_file), 'w') as f:
    f['rB']          = rB
    f['TB']          = T_fieldB['g']
    f['H_effB']      = H_fieldB['g']
    f['ln_ρB']       = ln_rho_fieldB['g'] 
    f['ln_TB']       = ln_T_fieldB['g']
    f['grad_ln_TB']  = grad_ln_T_fieldB['g']
    f['grad_ln_ρB']  = grad_ln_rho_fieldB['g']
    f['grad_s0B']    = grad_s_fieldB['g']

    f['rS']          = rS
    f['TS']          = T_fieldS['g']
    f['H_effS']      = H_fieldS['g']
    f['ln_ρS']       = ln_rho_fieldS['g'] 
    f['ln_TS']       = ln_T_fieldS['g']
    f['grad_ln_TS']  = grad_ln_T_fieldS['g']
    f['grad_ln_ρS']  = grad_ln_rho_fieldS['g']
    f['grad_s0S']    = grad_s_fieldS['g']

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
