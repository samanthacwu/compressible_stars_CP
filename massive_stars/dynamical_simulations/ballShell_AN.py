"""
d3 script for anelastic convection in a stitched BallBasis and ShellBasis domain.

A config file can be provided which overwrites any number of the options below, but command line flags can also be used if preferred.
Note that options specified in a cnofig file override command line arguments.

Usage:
    ballShell_AN.py [options]
    ballShell_AN.py <config> [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 2e2]
    --Pr=<Prandtl>       The Prandtl number  of the numerical diffusivities [default: 1]
    --ntheta=<res>       Number of theta grid points (Lmax+1)   [default: 4]
    --nrB=<res>          Number of radial grid points in ball (Nmax+1)   [default: 24]
    --nrS=<res>          Number of radial grid points in shell (Nmax+1)   [default: 8]
    --sponge             If flagged, add a damping layer in the shell that damps out waves.

    --wall_hours=<t>     Max number of wall hours to run simulation for [default: 24]
    --buoy_end_time=<t>  Max number of buoyancy time units to simulate [default: 1e5]

    --mesh=<n,m>         The processor mesh over which to distribute the cores

    --RK222              Use RK222 (default is SBDF2)
    --SBDF4              Use SBDF4 (default is SBDF2)
    --safety=<s>         Timestep CFL safety factor [default: 0.2]
    --CFL_max_r=<r>      zero out velocities above this radius value for CFL

    --ncc_file=<f>      path to a .h5 file of ICCs, curated from a MESA model
    --restart=<chk_f>    path to a checkpoint file to restart from
    --A0=<A>             Amplitude of random noise initial conditions [default: 1e-6]

    --label=<label>      A label to add to the end of the output directory

    --rotation_time=<t>  Rotation timescale, in days (if ncc_file is not None) or sim units (for polytrope)
"""
import os
import time
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
from docopt import docopt
from configparser import ConfigParser
import dedalus.public as d3
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)

# Define smooth Heaviside functions
from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

# Read options
args   = docopt(__doc__)
if args['<config>'] is not None: 
    config_file = Path(args['<config>'])
    config = ConfigParser()
    config.read(str(config_file))
    for n, v in config.items('parameters'):
        for k in args.keys():
            if k.split('--')[-1].lower() == n:
                if v == 'true': v = True
                args[k] = v

# Parameters
ntheta = int(args['--ntheta'])
nphi = int(2*ntheta)
nrB = int(args['--nrB'])
nrS = int(args['--nrS'])
resolutionB = (nphi, ntheta, nrB)
resolutionS = (nphi, ntheta, nrS)
L_dealias = N_dealias = dealias = 1.5
dtype = np.float64
Re  = float(args['--Re'])
Pr  = 1
Pe  = Pr*Re
ncc_file = args['--ncc_file']
wall_hours = float(args['--wall_hours'])
buoy_end_time = float(args['--buoy_end_time'])
sponge = args['--sponge']

# rotation
do_rotation = False
rotation_time = args['--rotation_time']
if rotation_time is not None:
    do_rotation = True
    rotation_time = float(rotation_time)
    dimensional_Ω = 2*np.pi / rotation_time  #radians / day [in MESA units]

# Initial conditions
restart = args['--restart']
A0 = float(args['--A0'])

# Timestepper
if args['--SBDF4']:
    ts = d3.SBDF4
    timestepper_history = [0, 1, 2, 3]
elif args['--RK222']:
    ts = d3.RK222
    timestepper_history = [0, ]
else:
    ts = d3.SBDF2
    timestepper_history = [0, 1,]
hermitian_cadence = 100
safety = float(args['--safety'])
CFL_max_r = args['--CFL_max_r']
if CFL_max_r is not None:
    CFL_max_r = float(CFL_max_r)
else:
    CFL_max_r = np.inf

# Processor mesh
mesh = args['--mesh']
ncpu = MPI.COMM_WORLD.size
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
else:
    log2 = np.log2(ncpu)
    if log2 == int(log2):
        mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
    logger.info("running on processor mesh={}".format(mesh))

# Output directory
out_dir = './' + sys.argv[0].split('.py')[0]
if sponge:
    out_dir += '_sponge'
if ncc_file is None:
    out_dir += '_polytrope'
if do_rotation:
    out_dir += '_rotation{}'.format(rotation_time)

out_dir += '_Re{}_{}x{}x{}+{}'.format(args['--Re'], nphi, ntheta, nrB, nrS)
if args['--label'] is not None:
    out_dir += '_{:s}'.format(args['--label'])
logger.info('saving data to {:s}'.format(out_dir))
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(out_dir)):
        os.makedirs('{:s}/'.format(out_dir))

# Read in domain bound values
if ncc_file is not None:
    with h5py.File(args['--ncc_file'], 'r') as f:
        Ri = f['r_inner'][()]
        Ro = f['r_outer'][()]
else:
    Ri = 1.1
    Ro = 1.5
logger.info('Ri: {:.2f} / Ro: {:.2f}'.format(Ri, Ro))
L_shell = Ro - Ri
sponge_function = lambda r: zero_to_one(r, Ri + 2*L_shell/3, 0.1*L_shell)

max_dt = None
t_buoy = None
t_rot = None

bases = OrderedDict()

# Bases
coords  = d3.SphericalCoordinates('phi', 'theta', 'r')
dist    = d3.Distributor((coords,), mesh=mesh, dtype=dtype)
bases['B']   = d3.BallBasis(coords, resolutionB, radius=Ri, dtype=dtype, dealias=(L_dealias, L_dealias, N_dealias)) 
bases['S'] = d3.ShellBasis(coords, resolutionS, radii=(Ri, Ro), dtype=dtype, dealias=(L_dealias, L_dealias, N_dealias))


tau_p = dist.Field(name='tau_p')

vec_fields = ['u',]
unit_vectors = ['ephi', 'etheta', 'er', 'ex', 'ey', 'ez']
scalar_fields = ['p', 's1', 'inv_T', 'H', 'ρ', 'T']
vec_taus = ['tau_u']
scalar_taus = ['tau_s']
tensor_nccs = ['I_matrix']
vec_nccs = ['grad_ln_ρ', 'grad_ln_T', 'grad_s0', 'grad_T', 'grad_inv_Pe_rad']
scalar_nccs = ['ln_ρ', 'ln_T', 'inv_Pe_rad', 'sponge']

variables = OrderedDict()
for bn, basis in bases.items():
    phi, theta, r = basis.local_grids(basis.dealias)
    phi1, theta1, r1 = basis.local_grids((1,1,1))
    variables['phi_'+bn], variables['theta_'+bn], variables['r_'+bn] = phi, theta, r
    variables['phi1_'+bn], variables['theta1_'+bn], variables['r1_'+bn] = phi1, theta1, r1

    #Define problem fields
    for fn in vec_fields:
        key = '{}_{}'.format(fn, bn)
        logger.info('creating vector field {}'.format(key))
        variables[key] = dist.VectorField(coords, name=key, bases=basis)
    for fn in scalar_fields:
        key = '{}_{}'.format(fn, bn)
        logger.info('creating scalar field {}'.format(key))
        variables[key] = dist.Field(name=key, bases=basis)

    #Taus
    S2_basis = basis.S2_basis()
    tau_names = []
    if type(basis) == d3.BallBasis:
        tau_names += ['',]
    else:
        tau_names += ['1', '2']
    for name in tau_names:
        for fn in vec_taus:
            key = '{}{}_{}'.format(fn, name, bn)
            logger.info('creating vector tau {}'.format(key))
            variables[key] = dist.VectorField(coords, name=key, bases=S2_basis)
        for fn in scalar_taus:
            key = '{}{}_{}'.format(fn, name, bn)
            logger.info('creating scalar tau {}'.format(key))
            variables[key] = dist.Field(name=key, bases=S2_basis)
    
    #Define problem NCCs
    for fn in tensor_nccs:
        key = '{}_{}'.format(fn, bn)
        logger.info('creating tensor NCC {}'.format(key))
        variables[key] = dist.TensorField(coords, name=key, bases=basis.radial_basis)
        if fn == 'I_matrix':
            for i in range(3):
                variables[key]['g'][i,i,:] = 1
    for fn in vec_nccs + ['er_LHS', 'rvec_LHS']:
        key = '{}_{}'.format(fn, bn)
        logger.info('creating vector NCC {}'.format(key))
        variables[key] = dist.VectorField(coords, name=key, bases=basis.radial_basis)
        if fn == 'er_LHS':
            variables[key]['g'][2] = 1
        elif fn == 'rvec_LHS':
            variables[key]['g'][2] = r1
    for fn in scalar_nccs:
        key = '{}_{}'.format(fn, bn)
        logger.info('creating scalar NCC {}'.format(key))
        variables[key] = dist.Field(name=key, bases=basis.radial_basis)

    #Define unit vectors
    for fn in unit_vectors:
        key = '{}_{}'.format(fn, bn)
        logger.info('creating unit vector field {}'.format(key))
        variables[key] = dist.VectorField(coords, name=key, bases=basis)

    if sponge:
        variables['sponge_S']['g'] = sponge_function(r)

    variables['ephi_{}'.format(bn)]['g'][0,:] = 1 
    variables['etheta_{}'.format(bn)]['g'][1,:] = 1 
    variables['er_{}'.format(bn)]['g'][2,:] = 1 

    variables['ex_{}'.format(bn)]['g'][0] = -np.sin(phi1)
    variables['ex_{}'.format(bn)]['g'][1] = np.cos(theta1)*np.cos(phi1)
    variables['ex_{}'.format(bn)]['g'][2] = np.sin(theta1)*np.cos(phi1)

    variables['ey_{}'.format(bn)]['g'][0] = np.cos(phi1)
    variables['ey_{}'.format(bn)]['g'][1] = np.cos(theta1)*np.sin(phi1)
    variables['ey_{}'.format(bn)]['g'][2] = np.sin(theta1)*np.sin(phi1)

    variables['ez_{}'.format(bn)]['g'][0] = 0
    variables['ez_{}'.format(bn)]['g'][1] = -np.sin(theta1)
    variables['ez_{}'.format(bn)]['g'][2] =  np.cos(theta1)
    
    for k in ['ex', 'ey', 'ez']:
        variables['{}_{}'.format(k, bn)] = d3.Grid(variables['{}_{}'.format(k, bn)]).evaluate()

    # Load MESA NCC file or setup NCCs using polytrope
    a_vector = variables['{}_{}'.format(vec_fields[0], bn)]
    grid_slices  = dist.layouts[-1].slices(a_vector.domain, N_dealias)
    a_vector.change_scales(basis.dealias)
    local_vncc_size = variables['{}_{}'.format(vec_nccs[0], bn)]['g'].size
    if ncc_file is not None:
        logger.info('reading NCCs from {}'.format(ncc_file))
        for k in vec_nccs + scalar_nccs + ['H', 'ρ', 'T', 'inv_T']:
            variables['{}_{}'.format(k, bn)].change_scales(basis.dealias)
        with h5py.File(ncc_file, 'r') as f:
            for k in vec_nccs:
                if '{}{}'.format(k, bn) not in f.keys():
                    logger.info('skipping {}{}, not in file'.format(k, bn))
                    continue
                if local_vncc_size > 0:
                    logger.info('reading {}{}'.format(k, bn))
                    variables['{}_{}'.format(k, bn)]['g'] = f['{}{}'.format(k, bn)][:,0,0,grid_slices[-1]][:,None,None,:]
            for k in scalar_nccs:
                if '{}{}'.format(k, bn) not in f.keys():
                    logger.info('skipping {}{}, not in file'.format(k, bn))
                    continue
                logger.info('reading {}{}'.format(k, bn))
                variables['{}_{}'.format(k, bn)]['g'] = f['{}{}'.format(k, bn)][:,:,grid_slices[-1]]
            variables['H_{}'.format(bn)]['g']         = f['H_eff{}'.format(bn)][:,:,grid_slices[-1]]
            variables['ρ_{}'.format(bn)]['g']         = np.exp(f['ln_ρ{}'.format(bn)][:,:,grid_slices[-1]])[None,None,:]
            variables['T_{}'.format(bn)]['g']         = f['T{}'.format(bn)][:,:,grid_slices[-1]][None,None,:]
            variables['inv_T_{}'.format(bn)]['g']     = 1/variables['T_{}'.format(bn)]['g']

            if max_dt is None:
                max_dt = f['max_dt'][()]

            if t_buoy is None:
                t_buoy = 1 #assume nondimensionalization on heating ~ buoyancy time

            if t_rot is None:
                if do_rotation:
                    sim_tau_sec = f['tau_nd'][()]
                    sim_tau_day = sim_tau_sec / (60*60*24)
                    Ω = sim_tau_day * dimensional_Ω 
                    t_rot = 1/(2*Ω)
                else:
                    t_rot = np.inf

            if sponge:
                f_brunt = f['tau_nd'][()]*np.sqrt(f['N2max_shell'][()])/(2*np.pi)
                variables['sponge_{}'.format(bn)]['g'] *= f_brunt

    else:
        logger.info("Using polytropic initial conditions")
        from scipy.interpolate import interp1d
        with h5py.File('benchmark/poly_nOuter1.6.h5', 'r') as f:
            T_func = interp1d(f['r'][()], f['T'][()])
            ρ_func = interp1d(f['r'][()], f['ρ'][()])
            grad_s0_func = interp1d(f['r'][()], f['grad_s0'][()])
            H_func   = interp1d(f['r'][()], f['H_eff'][()])
        max_grad_s0 = grad_s0_func(Ro)
        if max_dt is None:
            max_dt = 2/np.sqrt(max_grad_s0)
        if t_buoy is None:
            t_buoy      = 1
        if t_rot is None:
            if do_rotation:
                Ω = dimensional_Ω 
                t_rot = 1/(2*Ω)
            else:
                t_rot = np.inf

        variables['T_{}'.format(bn)]['g'] = T_func(r1)
        variables['ρ_{}'.format(bn)]['g'] = ρ_func(r1)
        variables['H_{}'.format(bn)]['g'] = H_func(r1)
        variables['inv_T_{}'.format(bn)]['g'] = 1/T_func(r1)
        
        grad_ln_ρ_full = (d3.grad(variables['ρ_{}'.format(bn)])/variables['ρ_{}'.format(bn)]).evaluate()
        grad_T_full = d3.grad(variables['T_{}'.format(bn)]).evaluate()
        grad_ln_T_full = (grad_T_full/variables['T_{}'.format(bn)]).evaluate()
        if local_vncc_size > 0:
            variables['grad_s0_{}'.format(bn)].change_scales(1)
            print(variables['grad_s0_{}'.format(bn)]['g'].shape, 'grad_s0_{}'.format(bn))
            variables['grad_s0_{}'.format(bn)]['g'][2]  = grad_s0_func(r1)
            for f in ['grad_ln_ρ', 'grad_ln_T', 'grad_T']: variables['{}_{}'.format(f, bn)].change_scales(basis.dealias)
            variables['grad_ln_ρ_{}'.format(bn)]['g']   = grad_ln_ρ_full['g'][:,0,0,None,None,:]
            variables['grad_ln_T_{}'.format(bn)]['g']   = grad_ln_T_full['g'][:,0,0,None,None,:]
            variables['grad_T_{}'.format(bn)]['g']      = grad_T_full['g'][:,0,0,None,None,:]
            variables['grad_inv_Pe_rad_{}'.format(bn)]['g'] = 0
        variables['ln_T_{}'.format(bn)]['g']   = np.log(T_func(r1))
        variables['ln_ρ_{}'.format(bn)]['g']   = np.log(ρ_func(r1))
        variables['inv_Pe_rad_{}'.format(bn)]['g'] = 1/Pe

    if do_rotation:
        logger.info("Running with Coriolis Omega = {:.3e}".format(Ω))

    u = variables['u_{}'.format(bn)]
    s1 = variables['s1_{}'.format(bn)]
    I_mat = variables['I_matrix_{}'.format(bn)]
    grad_ln_rho = variables['grad_ln_ρ_{}'.format(bn)]
    grad_ln_T = variables['grad_ln_T_{}'.format(bn)]
    er_LHS = variables['er_LHS_{}'.format(bn)]

    # Lift operators for boundary conditions
    if type(basis) == d3.BallBasis:
        lift_basis = basis.clone_with(k=0)
        variables['lift_{}'.format(bn)] = lift_fn = lambda A: d3.Lift(A, lift_basis, -1)
        variables['taus_u_{}'.format(bn)] = lift_fn(variables['tau_u_{}'.format(bn)])
        variables['taus_s_{}'.format(bn)] = lift_fn(variables['tau_s_{}'.format(bn)])
        variables['grad_u_{}'.format(bn)] = grad_u = d3.grad(u)
        variables['grad_s1_{}'.format(bn)] = grad_s = d3.grad(s1)
        variables['div_u_RHS_{}'.format(bn)] = div_u_RHS = d3.div(u)
        variables['div_u_{}'.format(bn)] = div_u = div_u_RHS
    else:
        lift_basis = basis.clone_with(k=2)
        variables['lift_{}'.format(bn)] = lift_fn = lambda A, n: d3.Lift(A, lift_basis, n)
        variables['taus_u_{}'.format(bn)] = lift_fn(variables['tau_u1_{}'.format(bn)], -1) + lift_fn(variables['tau_u2_{}'.format(bn)], -2)
        variables['taus_s_{}'.format(bn)] = lift_fn(variables['tau_s1_{}'.format(bn)], -1) + lift_fn(variables['tau_s2_{}'.format(bn)], -2)
        variables['grad_u_{}'.format(bn)] = grad_u = d3.grad(u) 
        variables['grad_s1_{}'.format(bn)] = grad_s = d3.grad(s1)
        variables['div_u_RHS_{}'.format(bn)] = div_u_RHS = d3.div(u)
        variables['div_u_{}'.format(bn)] = div_u = div_u_RHS

    #Stress matrices & viscous terms (assumes uniform kinematic viscosity; so dynamic viscosity mu = const * rho)
    variables['E_{}'.format(bn)] = E = 0.5*(grad_u + d3.transpose(grad_u))
    variables['E_RHS_{}'.format(bn)] = E_RHS = 0.5*(d3.grad(u) + d3.transpose(d3.grad(u)))
    variables['sigma_{}'.format(bn)] = sigma = 2*(E - (1/3)*div_u*I_mat)
    variables['sigma_RHS_{}'.format(bn)] = sigma_RHS = 2*(E_RHS - (1/3)*d3.div(u)*I_mat)
    variables['visc_div_stress_{}'.format(bn)] = d3.div(sigma) + d3.dot(sigma, grad_ln_rho)
    variables['VH_{}'.format(bn)] = 2*(d3.trace(d3.dot(E_RHS, E_RHS)) - (1/3)*div_u_RHS*div_u_RHS)


    # Grid-lock some operators / define grad's
    for field in ['H', 'inv_T']:
        variables['{}_{}'.format(field, bn)] = d3.Grid(variables['{}_{}'.format(field, bn)]).evaluate()

    #variables['div_rad_flux_{}'.format(bn)] = (1/Re)*d3.div(grad_s)
    chi_rad = variables['inv_Pe_rad_{}'.format(bn)]
    grad_chi_rad = variables['grad_inv_Pe_rad_{}'.format(bn)]
    variables['div_rad_flux_{}'.format(bn)] = chi_rad*(d3.div(grad_s) + d3.dot(grad_s, (grad_ln_rho + grad_ln_T))) + d3.dot(grad_s, grad_chi_rad)

    # Rotation and damping terms
    if do_rotation:
        ez = variables['ez_{}'.format(bn)]
        variables['rotation_term_{}'.format(bn)] = -2*Ω*d3.cross(ez, u)
    else:
        variables['rotation_term_{}'.format(bn)] = 0

    if args['--sponge']:
        variables['sponge_term_{}'.format(bn)] = u*variables['sponge_{}'.format(bn)]
    else:
        variables['sponge_term_{}'.format(bn)] = 0

# Put nccs and fields into locals()
locals().update(variables)

# Problem
problem = d3.IVP([p_B, p_S, u_B, u_S, s1_B, s1_S, tau_u_B, tau_u1_S, tau_u2_S, tau_s_B, tau_s1_S, tau_s2_S], namespace=locals())
#problem = d3.IVP([p_B, p_S, u_B, u_S, s1_B, s1_S, tau_p, tau_u_B, tau_u1_S, tau_u2_S, tau_s_B, tau_s1_S, tau_s2_S], namespace=locals())

problem.add_equation("div_u_B + dot(u_B, grad_ln_ρ_B) = 0", condition="ntheta != 0")
problem.add_equation("div_u_S + dot(u_S, grad_ln_ρ_S) = 0", condition="ntheta != 0")
problem.add_equation("dt(u_B) + grad(p_B) + grad_T_B*s1_B - (1/Re)*visc_div_stress_B + sponge_term_B + taus_u_B = cross(u_B, curl(u_B)) + rotation_term_B", condition="ntheta != 0")
problem.add_equation("dt(u_S) + grad(p_S) + grad_T_S*s1_S - (1/Re)*visc_div_stress_S + sponge_term_S + taus_u_S = cross(u_S, curl(u_S)) + rotation_term_S", condition="ntheta != 0")
problem.add_equation("u_B = 0", condition = "ntheta == 0")
problem.add_equation("u_S = 0", condition = "ntheta == 0")
problem.add_equation("p_B = 0", condition = "ntheta == 0")
problem.add_equation("p_S = 0", condition = "ntheta == 0")
problem.add_equation("dt(s1_B) + dot(u_B, grad_s0_B) - div_rad_flux_B + taus_s_B = - dot(u_B, grad_s1_B) + H_B + (1/Re)*inv_T_B*VH_B ")
problem.add_equation("dt(s1_S) + dot(u_S, grad_s0_S) - div_rad_flux_S + taus_s_S = - dot(u_S, grad(s1_S)) + H_S + (1/Re)*inv_T_S*VH_S ")

problem.add_equation("u_B(r=Ri) - u_S(r=Ri) = 0", condition="ntheta != 0")
problem.add_equation("p_B(r=Ri) - p_S(r=Ri) = 0", condition="ntheta != 0")
problem.add_equation("angular(radial(sigma_B(r=Ri) - sigma_S(r=Ri)), index=0) = 0", condition="ntheta != 0")
problem.add_equation("radial(u_S(r=Ro)) = 0", condition="ntheta != 0")
problem.add_equation("angular(radial(E_S(r=Ro))) = 0", condition="ntheta != 0")
problem.add_equation("tau_u_B = 0", condition="ntheta == 0")
problem.add_equation("tau_u1_S = 0", condition="ntheta == 0")
problem.add_equation("tau_u2_S = 0", condition="ntheta == 0")

## Entropy BCs
problem.add_equation("s1_B(r=Ri) - s1_S(r=Ri) = 0")
problem.add_equation("radial(grad_s1_B(r=Ri) - grad(s1_S)(r=Ri)) = 0")
problem.add_equation("radial(grad_s1_S(r=Ro)) = 0")


#problem.add_equation("div_u_B + dot(u_B, grad_ln_ρ_B) + tau_p = 0")
#problem.add_equation("div_u_S + dot(u_S, grad_ln_ρ_S) + tau_p = 0")
#problem.add_equation("dt(u_B) + grad(p_B) + grad_T_B*s1_B - (1/Re)*visc_div_stress_B + sponge_term_B + taus_u_B = cross(u_B, curl(u_B)) + rotation_term_B")
#problem.add_equation("dt(u_S) + grad(p_S) + grad_T_S*s1_S - (1/Re)*visc_div_stress_S + sponge_term_S + taus_u_S = cross(u_S, curl(u_S)) + rotation_term_S")
#problem.add_equation("dt(s1_B) + dot(u_B, grad_s0_B) - div_rad_flux_B + taus_s_B = - dot(u_B, grad_s1_B) + H_B + (1/Re)*inv_T_B*VH_B ")
#problem.add_equation("dt(s1_S) + dot(u_S, grad_s0_S) - div_rad_flux_S + taus_s_S = - dot(u_S, grad(s1_S)) + H_S + (1/Re)*inv_T_S*VH_S ")
#
#problem.add_equation("u_B(r=Ri) - u_S(r=Ri) = 0")
#problem.add_equation("p_B(r=Ri) - p_S(r=Ri) = 0")
#problem.add_equation("angular(radial(sigma_B(r=Ri) - sigma_S(r=Ri)), index=0) = 0")
#problem.add_equation("radial(u_S(r=Ro)) = 0")
#problem.add_equation("angular(radial(E_S(r=Ro))) = 0")
##
### Entropy BCs
#problem.add_equation("s1_B(r=Ri) - s1_S(r=Ri) = 0")
#problem.add_equation("radial(grad_s1_B(r=Ri) - grad(s1_S)(r=Ri)) = 0")
#problem.add_equation("radial(grad_s1_S(r=Ro)) = 0")
#
#problem.add_equation("integ(p_B) + integ(p_S) = 0")

logger.info("Problem built")
# Solver
solver = problem.build_solver(ts)
solver.stop_sim_time = buoy_end_time*t_buoy
solver.stop_wall_time = wall_hours * 60 * 60
logger.info("solver built")

# Initial conditions / Checkpoint
write_mode = 'overwrite'
timestep = None
if restart is not None:
    write, timestep = solver.load_state(restart)
    write_mode = 'append'
else:
    # Initial conditions
    s1_B.fill_random(layout='g', seed=42, distribution='normal', scale=A0)
    s1_B.low_pass_filter(scales=0.25)
    s1_S.fill_random(layout='g', seed=42, distribution='normal', scale=A0)
    s1_S.low_pass_filter(scales=0.25)
    s1_B['g'] *= np.sin(theta1_B)
    s1_S['g'] *= np.sin(theta1_S)
    s1_B['g'] *= np.cos(np.pi*r1_B/Ro)
    s1_S['g'] *= np.cos(np.pi*r1_S/Ro)

## Analysis Setup
# Cadence
scalar_dt = 0.25*t_buoy
lum_dt   = 0.5*t_buoy
visual_dt = 0.05*t_buoy
outer_shell_dt = max_dt

##TODO: revisit outputs
## Operators, extra fields
#r_vals_B = dist.Field(name='r_vals_B', bases=ball_basis)
#r_vals_S = dist.Field(name='r_vals_S', bases=shell_basis)
#r_vals_B['g'] = r1B
#r_vals_S['g'] = r1S
#er_B = d3.Grid(er_B).evaluate()
#er_S = d3.Grid(er_S).evaluate()
#r_vals_B = d3.Grid(r_vals_B).evaluate()
#r_vals_S = d3.Grid(r_vals_S).evaluate()
u_B_squared = d3.dot(u_B, u_B)
u_S_squared = d3.dot(u_S, u_S)
#ur_B = d3.dot(er_B, u_B)
#ur_S = d3.dot(er_S, u_S)
#h_B = p_B - 0.5*u_B_squared + T_B*s1_B
#h_S = p_S - 0.5*u_S_squared + T_S*s1_S
#pomega_hat_B = p_B - 0.5*u_B_squared
#pomega_hat_S = p_S - 0.5*u_S_squared
#
#visc_flux_B_r = 2*d3.dot(er_B, d3.dot(u_B, E_B) - (1/3) * u_B * div_u_B)
#visc_flux_S_r = 2*d3.dot(er_S, d3.dot(u_S, E_S) - (1/3) * u_S * div_u_S)
#
## Angular momentum
#r_B_vec_post = dist.VectorField(coords, name='rB_vec_post', bases=ball_basis)
#r_B_vec_post['g'][2] = r1B
#L_AM_B = d3.cross(r_B_vec_post, ρ_B*u_B)
#Lx_AM_B = d3.dot(ex_B, L_AM_B)
#Ly_AM_B = d3.dot(ey_B, L_AM_B)
#Lz_AM_B = d3.dot(ez_B, L_AM_B)
#
#r_S_vec_post = dist.VectorField(coords, name='rS_vec_post', bases=shell_basis)
#r_S_vec_post['g'][2] = r1S
#L_AM_S = d3.cross(r_S_vec_post, ρ_S*u_S)
#Lx_AM_S = d3.dot(ex_S, L_AM_S)
#Ly_AM_S = d3.dot(ey_S, L_AM_S)
#Lz_AM_S = d3.dot(ez_S, L_AM_S)
#
# Averaging Operations
volume  = (4/3)*np.pi*Ro**3
volume_B = (4/3)*np.pi*Ri**3
volume_S = volume - volume_B

az_avg = lambda A: d3.Average(A, coords.coords[0])
s2_avg = lambda A: d3.Average(A, coords.S2coordsys)
vol_avg_B = lambda A: d3.Integrate(A/volume_B, coords)
vol_avg_S = lambda A: d3.Integrate(A/volume_S, coords)
luminosity_B = lambda A: s2_avg((4*np.pi*r_vals_B**2) * A)
luminosity_S = lambda A: s2_avg((4*np.pi*r_vals_S**2) * A)

analysis_tasks = []

slices = solver.evaluator.add_file_handler('{:s}/slices'.format(out_dir), sim_dt=visual_dt, max_writes=40)
slices.add_task(u_B(theta=np.pi/2), name='u_B_eq', layout='g')
slices.add_task(u_S(theta=np.pi/2), name='u_S_eq', layout='g')
slices.add_task(s1_B(theta=np.pi/2), name='s1_B_eq', layout='g')
slices.add_task(s1_S(theta=np.pi/2), name='s1_S_eq', layout='g')
for fd, name in zip((u_B, s1_B), ('u_B', 's1_B')):
    for radius, r_str in zip((0.5, 1), ('0.5', '1')):
        operation = fd(r=radius)
        slices.add_task(operation, name=name+'(r={})'.format(r_str), layout='g')
for fd, name in zip((u_S, s1_S), ('u_S', 's1_S')):
    for radius, r_str in zip((0.95*Ro,), ('0.95R',)):
        operation = fd(r=radius)
        slices.add_task(operation, name=name+'(r={})'.format(r_str), layout='g')
for az_val, name in zip([0, np.pi/2, np.pi, 3*np.pi/2], ['(phi=0)', '(phi=0.5*pi)', '(phi=pi)', '(phi=1.5*pi)',]):
    slices.add_task(u_B(phi=az_val),  name='u_B' +name, layout='g')
    slices.add_task(s1_B(phi=az_val), name='s1_B'+name, layout='g')
for az_val, name in zip([0, np.pi/2, np.pi, 3*np.pi/2], ['(phi=0)', '(phi=0.5*pi)', '(phi=pi)', '(phi=1.5*pi)',]):
    slices.add_task(u_S(phi=az_val),  name='u_S' +name, layout='g')
    slices.add_task(s1_S(phi=az_val), name='s1_S'+name, layout='g')
analysis_tasks.append(slices)

#scalars = solver.evaluator.add_file_handler('{:s}/scalars'.format(out_dir), sim_dt=scalar_dt, max_writes=np.inf)
#scalars.add_task(vol_avg_B(Re*(u_B_squared)**(1/2)), name='Re_avg_ball',  layout='g')
#scalars.add_task(vol_avg_S(Re*(u_S_squared)**(1/2)), name='Re_avg_shell', layout='g')
#scalars.add_task(vol_avg_B(ρ_B*u_B_squared/2), name='KE_ball',   layout='g')
#scalars.add_task(vol_avg_S(ρ_S*u_S_squared/2), name='KE_shell',  layout='g')
#scalars.add_task(vol_avg_B(ρ_B*T_B*s1_B), name='TE_ball',  layout='g')
#scalars.add_task(vol_avg_S(ρ_S*T_S*s1_S), name='TE_shell', layout='g')
#scalars.add_task(vol_avg_B(Lx_AM_B), name='Lx_AM_ball', layout='g')
#scalars.add_task(vol_avg_B(Ly_AM_B), name='Ly_AM_ball', layout='g')
#scalars.add_task(vol_avg_B(Lz_AM_B), name='Lz_AM_ball', layout='g')
#scalars.add_task(vol_avg_S(Lx_AM_S), name='Lx_AM_shell', layout='g')
#scalars.add_task(vol_avg_S(Ly_AM_S), name='Ly_AM_shell', layout='g')
#scalars.add_task(vol_avg_S(Lz_AM_S), name='Lz_AM_shell', layout='g')
#analysis_tasks.append(scalars)
#
#profiles = solver.evaluator.add_file_handler('{:s}/profiles'.format(out_dir), sim_dt=visual_dt, max_writes=100)
#profiles.add_task(luminosity_B(ρ_B*ur_B*pomega_hat_B),         name='wave_lumB', layout='g')
#profiles.add_task(luminosity_S(ρ_S*ur_S*pomega_hat_S),         name='wave_lumS', layout='g')
#profiles.add_task(luminosity_B(ρ_B*ur_B*h_B),                   name='enth_lumB', layout='g')
#profiles.add_task(luminosity_S(ρ_S*ur_S*h_S),                   name='enth_lumS', layout='g')
#profiles.add_task(luminosity_B(-ρ_B*visc_flux_B_r/Re),         name='visc_lumB', layout='g')
#profiles.add_task(luminosity_S(-ρ_S*visc_flux_S_r/Re),         name='visc_lumS', layout='g')
#profiles.add_task(luminosity_B(-ρ_B*T_B*d3.dot(er_B, grad_s1_B)/Pe), name='cond_lumB', layout='g')
#profiles.add_task(luminosity_S(-ρ_S*T_S*d3.dot(er_S, grad_s1_S)/Pe), name='cond_lumS', layout='g')
#profiles.add_task(luminosity_B(0.5*ρ_B*ur_B*u_B_squared),       name='KE_lumB',   layout='g')
#profiles.add_task(luminosity_S(0.5*ρ_S*ur_S*u_S_squared),       name='KE_lumS',   layout='g')
#analysis_tasks.append(profiles)

#if args['--sponge']:
#    surface_shell_slices = solver.evaluator.add_file_handler('{:s}/wave_shell_slices'.format(out_dir), sim_dt=100000*max_dt, max_writes=20)
#    for rval in [0.90, 1.05]:
#        surface_shell_slices.add_task(d3.radial(u_B(r=rval)), name='u(r={})'.format(rval), layout='g')
#        surface_shell_slices.add_task(pomega_hat_B(r=rval), name='pomega(r={})'.format(rval),    layout='g')
#    for rval in [1.5, 2.0, 2.5, 3.0, 3.5]:
#        surface_shell_slices.add_task(d3.radial(u_S(r=rval)), name='u(r={})'.format(rval), layout='g')
#        surface_shell_slices.add_task(pomega_hat_S(r=rval), name='pomega(r={})'.format(rval),    layout='g')
#    analysis_tasks.append(surface_shell_slices)
#else:
#    surface_shell_slices = solver.evaluator.add_file_handler('{:s}/wave_shell_slices'.format(out_dir), sim_dt=100000*max_dt, max_writes=20)
#    surface_shell_slices.add_task(s1_S(r=Ro),         name='s1_surf',    layout='g')
#    analysis_tasks.append(surface_shell_slices)
#
if Re > 1e4:
    chk_time = 2*t_buoy
else:
    chk_time = 10*t_buoy
checkpoint = solver.evaluator.add_file_handler('{:s}/checkpoint'.format(out_dir), max_writes=1, sim_dt=chk_time)
checkpoint.add_tasks(solver.state, layout='g')

re_ball = solver.evaluator.add_dictionary_handler(iter=10)
re_ball.add_task(vol_avg_B(Re*(u_B_squared)**(1/2)), name='Re_avg_ball', layout='g')
re_ball.add_task(d3.integ(ρ_S*(u_S_squared)/2), name='KE_shell', layout='g')

#CFL setup
heaviside_cfl = dist.Field(name='heaviside_cfl', bases=bases['B'])
heaviside_cfl['g'] = 1
if np.sum(r1_B > CFL_max_r) > 0:
    heaviside_cfl['g'][:,:, r1_B.flatten() > CFL_max_r] = 0
heaviside_cfl = d3.Grid(heaviside_cfl).evaluate()

#initial_max_dt = max_dt
initial_max_dt = np.min((visual_dt, t_rot*0.5))
while initial_max_dt < max_dt:
    max_dt /= 2
if timestep is None:
    timestep = initial_max_dt
my_cfl = d3.CFL(solver, timestep, safety=safety, cadence=1, max_dt=initial_max_dt, min_change=0.1, max_change=1.5, threshold=0.1)
my_cfl.add_velocity(heaviside_cfl*u_B)
my_cfl.add_velocity(u_S)

#startup iterations
for i in range(10):
    solver.step(timestep)
    Re_avg = re_ball.fields['Re_avg_ball']
    KE_shell = re_ball.fields['KE_shell']
    if dist.comm_cart.rank == 0:
        KE0 = KE_shell['g'].min()
        Re0 = Re_avg['g'].min()
        taus = tau_p['g'].squeeze()
    else:
        KE0 = None 
        Re0 = None
        taus = -1
    Re0 = dist.comm_cart.bcast(Re0, root=0)
    KE0 = dist.comm_cart.bcast(KE0, root=0)
    this_str = "startup iteration {}, t = {:f}, timestep = {:f}, Re = {:.4e}".format(i, solver.sim_time, timestep, Re0)
    this_str += ", KE = {:.4e}".format(KE0)
#    this_str += ", tau_p = {:.4e}".format(taus)
    logger.info(this_str)
    timestep = my_cfl.compute_timestep()

# Main loop
start_time = time.time()
start_iter = solver.iteration
max_dt_check = True
current_max_dt = my_cfl.max_dt
slice_process = False
just_wrote    = False
slice_time = np.inf
Re0 = 0
try:
    while solver.proceed:
        timestep = my_cfl.compute_timestep()

#        if just_wrote:
#            just_wrote = False
#            num_steps = np.ceil(outer_shell_dt / timestep)
#            timestep = current_max_dt = my_cfl.stored_dt = outer_shell_dt/num_steps
#        elif max_dt_check:
#            timestep = np.min((timestep, current_max_dt))
#        else:
#            my_cfl.stored_dt = timestep = current_max_dt
#
#        t_future = solver.sim_time + timestep
#        if t_future >= slice_time*(1-1e-8):
#           slice_process = True
#
#        if solver.iteration % hermitian_cadence in timestepper_history:
#            for f in solver.state:
#                f.require_grid_space()
#
        solver.step(timestep)

        if solver.iteration % 10 == 0:
            Re_avg = re_ball.fields['Re_avg_ball']
            KE_shell = re_ball.fields['KE_shell']
            if dist.comm_cart.rank == 0:
                KE0 = KE_shell['g'].min()
                Re0 = Re_avg['g'].min()
                taus = tau_p['g'].squeeze()
            else:
                KE0 = None
                Re0 = None
                taus = -1
            Re0 = dist.comm_cart.bcast(Re0, root=0)
            KE0 = dist.comm_cart.bcast(KE0, root=0)
            this_str = "t = {:f}, timestep = {:f}, Re = {:.4e}".format(solver.sim_time, timestep, Re0)
            this_str += ", KE = {:.4e}".format(KE0)
#            this_str += ", tau_ps = {:.4e}".format(taus)
            logger.info(this_str)
        if max_dt_check and timestep < outer_shell_dt:
            my_cfl.max_dt = max_dt
            max_dt_check = False
            just_wrote = True
            slice_time = solver.sim_time + outer_shell_dt

#        if slice_process:
#            slice_process = False
#            wall_time = time.time() - solver.start_time
#            solver.evaluator.evaluate_handlers([surface_shell_slices],wall_time=wall_time, sim_time=solver.sim_time, iteration=solver.iteration,world_time = time.time(),timestep=timestep)
#            slice_time = solver.sim_time + outer_shell_dt
#            just_wrote = True

        if np.isnan(Re0):
            logger.info('exiting with NaN')
            break

except:
    logger.info('something went wrong in main loop, making final checkpoint')
    raise
finally:
    end_time = time.time()
    end_iter = solver.iteration
    cpu_sec  = end_time - start_time
    n_iter   = end_iter - start_iter


    fcheckpoint = solver.evaluator.add_file_handler('{:s}/final_checkpoint'.format(out_dir), max_writes=1, sim_dt=10*t_buoy)
    fcheckpoint.add_tasks(solver.state, layout='g')
    solver.step(timestep)

    if dist.comm_cart.rank == 0:
        for handler in analysis_tasks:
            handler.process_virtual_file()

    #TODO: Make the end-of-sim report better
    n_coeffs = np.prod(resolutionB) + np.prod(resolutionS)
    n_cpu    = dist.comm_cart.size
    dof_cycles_per_cpusec = n_coeffs*n_iter/(cpu_sec*n_cpu)
    logger.info('DOF-cycles/cpu-sec : {:e}'.format(dof_cycles_per_cpusec))
    logger.info('Run iterations: {:e}'.format(n_iter))
    logger.info('Sim end time: {:e}'.format(solver.sim_time))
    logger.info('Run time: {}'.format(cpu_sec))
