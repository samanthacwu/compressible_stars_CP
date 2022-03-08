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
from operator import itemgetter
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

def make_bases(resolutions, stitch_radii, radius, dealias=3/2, dtype=np.float64, mesh=None):
    bases = OrderedDict()
    coords  = d3.SphericalCoordinates('phi', 'theta', 'r')
    dist    = d3.Distributor((coords,), mesh=mesh, dtype=dtype)
    bases_keys = ['B']
    for i, resolution in enumerate(resolutions):
        if i == 0:
            if len(resolutions) == 1:
                ball_radius = radius
            else:
                ball_radius = stitch_radii[i]
            bases['B']   = d3.BallBasis(coords, resolution, radius=ball_radius, dtype=dtype, dealias=dealias)
        else:
            if len(resolutions) == i+1:
                shell_radii = (stitch_radii[i-1], radius)
            else:
                shell_radii = (stitch_radii[i-1], stitch_radii[i])
            bases['S{}'.format(i)] = d3.ShellBasis(coords, resolutionS, radii=shell_radii, dtype=dtype, dealias=dealias)
            bases_keys += ['S{}'.format(i)]
    return coords, dist, bases, bases_keys

def make_fields(bases, coords, dist, vec_fields=[], scalar_fields=[], vec_taus=[], scalar_taus=[], vec_nccs=[], scalar_nccs=[], sponge=False, do_rotation=False):
    variables = OrderedDict()
    for basis_number, bn in enumerate(bases.keys()):
        unit_vectors = ['ephi', 'etheta', 'er', 'ex', 'ey', 'ez']
        basis = bases[bn]
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

        #Define identity matrix
        key = 'I_matrix_{}'.format(bn)
        logger.info('creating identity matrix NCC {}'.format(key))
        variables[key] = dist.TensorField(coords, name=key, bases=basis.radial_basis)
        for i in range(3):
            variables[key]['g'][i,i,:] = 1

        #Define unit vectors
        for fn in unit_vectors:
            key = '{}_{}'.format(fn, bn)
            logger.info('creating unit vector field {}'.format(key))
            variables[key] = dist.VectorField(coords, name=key, bases=basis)

        if sponge:
            variables['sponge_{}'.format(bn)]['g'] = sponge_function(r1)

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
            variables['rotation_term_{}'.format(bn)] = -2*Omega*d3.cross(ez, u)
        else:
            variables['rotation_term_{}'.format(bn)] = 0

        if args['--sponge']:
            variables['sponge_term_{}'.format(bn)] = u*variables['sponge_{}'.format(bn)]
        else:
            variables['sponge_term_{}'.format(bn)] = 0
    return variables

def fill_structure(bases, dist, variables, ncc_file, radius, Pe, vec_fields=[], vec_nccs=[], scalar_nccs=[], sponge=False, do_rotation=False):
    max_dt = None
    t_buoy = None
    t_rot = None
    for basis_number, bn in enumerate(bases.keys()):
        basis = bases[bn]
        phi, theta, r = basis.local_grids(basis.dealias)
        phi1, theta1, r1 = basis.local_grids((1,1,1))
        # Load MESA NCC file or setup NCCs using polytrope
        a_vector = variables['{}_{}'.format(vec_fields[0], bn)]
        grid_slices  = dist.layouts[-1].slices(a_vector.domain, basis.dealias[-1])
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
                        Omega = sim_tau_day * dimensional_Omega 
                        t_rot = 1/(2*Omega)
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
            max_grad_s0 = grad_s0_func(radius)
            if max_dt is None:
                max_dt = 2/np.sqrt(max_grad_s0)
            if t_buoy is None:
                t_buoy      = 1
            if t_rot is None:
                if do_rotation:
                    Omega = dimensional_Omega 
                    t_rot = 1/(2*Omega)
                else:
                    t_rot = np.inf

            variables['T_{}'.format(bn)]['g'] = T_func(r1)
            variables['ρ_{}'.format(bn)]['g'] = ρ_func(r1)
            variables['H_{}'.format(bn)]['g'] = H_func(r)
            variables['inv_T_{}'.format(bn)]['g'] = 1/T_func(r)
            
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
            logger.info("Running with Coriolis Omega = {:.3e}".format(Omega))

        return variables, (max_dt, t_buoy, t_rot)

def get_anelastic_variables(bases, bases_keys, variables):
    problem_variables = []
    problem_taus = []
    equations = OrderedDict()
    u_BCs = OrderedDict()
    s_BCs = OrderedDict()
    for basis_number, bn in enumerate(bases_keys):
        basis = bases[bn]
        phi, theta, r = basis.local_grids(basis.dealias)
        phi1, theta1, r1 = basis.local_grids((1,1,1))

        for field in ['p', 'u', 's1']:
            problem_variables.append(variables['{}_{}'.format(field, bn)])
        for tau in ['tau_u', 'tau_s']:
            if type(basis) == d3.BallBasis:
                problem_taus.append(variables['{}_{}'.format(tau, bn)])
            else:
                problem_taus.append(variables['{}1_{}'.format(tau, bn)])
                problem_taus.append(variables['{}2_{}'.format(tau, bn)])

    return problem_variables + problem_taus

def set_anelastic_problem(problem, bases, bases_keys):
    equations = OrderedDict()
    u_BCs = OrderedDict()
    s_BCs = OrderedDict()
    for basis_number, bn in enumerate(bases_keys):
        basis = bases[bn]

        #Standard Equations
        equations['continuity_{}'.format(bn)] = "div_u_{0} + dot(u_{0}, grad_ln_ρ_{0}) = 0".format(bn)
        equations['momentum_{}'.format(bn)] = "dt(u_{0}) + grad(p_{0}) + grad_T_{0}*s1_{0} - (1/Re)*visc_div_stress_{0} + sponge_term_{0} + taus_u_{0} = cross(u_{0}, curl(u_{0})) + rotation_term_{0}".format(bn)
        equations['energy_{}'.format(bn)] = "dt(s1_{0}) + dot(u_{0}, grad_s0_{0}) - div_rad_flux_{0} + taus_s_{0} = - dot(u_{0}, grad_s1_{0}) + H_{0} + (1/Re)*inv_T_{0}*VH_{0}".format(bn)

        #Boundary conditions
        if type(basis) == d3.BallBasis:
            if basis_number == len(bases_keys) - 1:
                #No shell bases
                u_BCs['BC_u1_{}'.format(bn)] = "radial(u_{0}(r={1})) = 0".format(bn, 'radius')
                u_BCs['BC_u2_{}'.format(bn)] = "angular(radial(E_{0}(r={1}))) = 0".format(bn, 'radius')
                s_BCs['BC_s_{}'.format(bn)] = "radial(grad_s1_{0}(r={1})) = 0".format(bn, 'radius')
            else:
                shell_name = bases_keys[basis_number+1] 
                rval = stitch_radii[basis_number]
                u_BCs['BC_u_{}'.format(bn)] = "u_{0}(r={2}) - u_{1}(r={2}) = 0".format(bn, shell_name, rval)
                s_BCs['BC_s_{}'.format(bn)] = "s1_{0}(r={2}) - s1_{1}(r={2}) = 0".format(bn, shell_name, rval)
        else:
            #Stitch to basis below
            below_name = bases_keys[basis_number - 1]
            rval = stitch_radii[basis_number - 1]
            u_BCs['BC_u1_{}'.format(bn)] = "p_{0}(r={2}) - p_{1}(r={2}) = 0".format(bn, below_name, rval)
            u_BCs['BC_u2_{}'.format(bn)] = "angular(radial(sigma_{0}(r={2}) - sigma_{1}(r={2}))) = 0".format(bn, below_name, rval)
            s_BCs['BC_s1_{}'.format(bn)] = "radial(grad(s1_{0})(r={2}) - grad(s1_{1})(r={2})) = 0".format(bn, below_name, rval)

            #Add upper BCs
            if basis_number == len(bases_keys) - 1:
                #top of domain
                u_BCs['BC_u3_{}'.format(bn)] = "radial(u_{0}(r={1})) = 0".format(bn, 'radius')
                u_BCs['BC_u4_{}'.format(bn)] = "angular(radial(E_{0}(r={1}))) = 0".format(bn, 'radius')
                s_BCs['BC_s2_{}'.format(bn)] = "radial(grad_s1_{0}(r={1})) = 0".format(bn, 'radius')
            else:
                shn = bases_keys[basis_number+1] 
                rval = stitch_radii[basis_number]
                u_BCs['BC_u3_{}'.format(bn)] = "u_{0}(r={2}) - u_{1}(r={2}) = 0".format(bn, shn, rval)
                s_BCs['BC_s2_{}'.format(bn)] = "s1_{0}(r={2}) - s1_{1}(r={2}) = 0".format(bn, shn, rval)


    for bn, basis in bases.items():
        continuity = equations['continuity_{}'.format(bn)]
        continuity_ell0 = "p_{} = 0".format(bn)
        logger.info('adding eqn "{}" for ntheta != 0'.format(continuity))
        logger.info('adding eqn "{}" for ntheta == 0'.format(continuity_ell0))
        problem.add_equation(continuity, condition="ntheta != 0")
        problem.add_equation(continuity_ell0, condition="ntheta == 0")

        momentum = equations['momentum_{}'.format(bn)]
        momentum_ell0 = "u_{} = 0".format(bn)
        logger.info('adding eqn "{}" for ntheta != 0'.format(momentum))
        logger.info('adding eqn "{}" for ntheta == 0'.format(momentum_ell0))
        problem.add_equation(momentum, condition="ntheta != 0")
        problem.add_equation(momentum_ell0, condition="ntheta == 0")

        energy = equations['energy_{}'.format(bn)]
        logger.info('adding eqn "{}"'.format(energy))
        problem.add_equation(energy)

    for BC in s_BCs.values():
        logger.info('adding BC "{}"'.format(BC))
        problem.add_equation(BC)

    for BC in u_BCs.values():
        logger.info('adding BC "{}" for ntheta != 0'.format(BC))
        problem.add_equation(BC, condition="ntheta != 0")

    for bn, basis in bases.items():
        if type(basis) == d3.BallBasis:
            BC = 'tau_u_{} = 0'.format(bn)
            logger.info('adding BC "{}" for ntheta == 0'.format(BC))
            problem.add_equation(BC, condition="ntheta == 0")
        else:
            for i in [1, 2]:
                BC = 'tau_u{}_{} = 0'.format(i, bn)
                logger.info('adding BC "{}" for ntheta == 0'.format(BC))
                problem.add_equation(BC, condition="ntheta == 0")
    return problem

if __name__ == '__main__':

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
        dimensional_Omega = 2*np.pi / rotation_time  #radians / day [in MESA units]

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

    resolutions = (resolutionB, resolutionS)
    stitch_radii = (Ri,)
    radius = Ro
    coords, dist, bases, bases_keys = make_bases(resolutions, stitch_radii, radius, dealias=(L_dealias, L_dealias, N_dealias), dtype=dtype, mesh=mesh)

    vec_fields = ['u',]
    scalar_fields = ['p', 's1', 'inv_T', 'H', 'ρ', 'T']
    vec_taus = ['tau_u']
    scalar_taus = ['tau_s']
    vec_nccs = ['grad_ln_ρ', 'grad_ln_T', 'grad_s0', 'grad_T', 'grad_inv_Pe_rad']
    scalar_nccs = ['ln_ρ', 'ln_T', 'inv_Pe_rad', 'sponge']
    variables = make_fields(bases, coords, dist, 
                            vec_fields=vec_fields, scalar_fields=scalar_fields, 
                            vec_taus=vec_taus, scalar_taus=scalar_taus, 
                            vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                            sponge=sponge, do_rotation=do_rotation)


    variables, timescales = fill_structure(bases, dist, variables, ncc_file, Ro, Pe,
                                            vec_fields=vec_fields, vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                                            sponge=sponge, do_rotation=do_rotation)
    max_dt, t_buoy, t_rot = timescales

    # Put nccs and fields into locals()
    locals().update(variables)


    # Problem
    prob_variables = get_anelastic_variables(bases, bases_keys, variables)
    problem = d3.IVP(prob_variables, namespace=locals())

    problem = set_anelastic_problem(problem, bases, bases_keys)

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
        s1_S1.fill_random(layout='g', seed=42, distribution='normal', scale=A0)
        s1_S1.low_pass_filter(scales=0.25)
        s1_B['g'] *= np.sin(theta1_B)
        s1_S1['g'] *= np.sin(theta1_S1)
        s1_B['g'] *= np.cos(np.pi*r1_B/Ro)
        s1_S1['g'] *= np.cos(np.pi*r1_S1/Ro)

    ## Analysis Setup
    # Cadence
    scalar_dt = 0.25*t_buoy
    lum_dt   = 0.5*t_buoy
    visual_dt = 0.05*t_buoy
    outer_shell_dt = max_dt
    if Re > 1e4:
        checkpoint_time = 2*t_buoy
    else:
        checkpoint_time = 10*t_buoy

    analysis_tasks = []
    slices = solver.evaluator.add_file_handler('{:s}/slices'.format(out_dir), sim_dt=visual_dt, max_writes=40)
    scalars = solver.evaluator.add_file_handler('{:s}/scalars'.format(out_dir), sim_dt=scalar_dt, max_writes=np.inf)
    profiles = solver.evaluator.add_file_handler('{:s}/profiles'.format(out_dir), sim_dt=visual_dt, max_writes=100)
    if args['--sponge']:
        surface_shell_slices = solver.evaluator.add_file_handler('{:s}/wave_shell_slices'.format(out_dir), sim_dt=100000*max_dt, max_writes=20)
    else:
        surface_shell_slices = solver.evaluator.add_file_handler('{:s}/wave_shell_slices'.format(out_dir), sim_dt=100000*max_dt, max_writes=20)
    analysis_tasks.append(slices)
    analysis_tasks.append(scalars)
    analysis_tasks.append(profiles)
    analysis_tasks.append(surface_shell_slices)

    checkpoint = solver.evaluator.add_file_handler('{:s}/checkpoint'.format(out_dir), max_writes=1, sim_dt=checkpoint_time)
    checkpoint.add_tasks(solver.state, layout='g')
    analysis_tasks.append(checkpoint)


    logger_handler = solver.evaluator.add_dictionary_handler(iter=10)

    az_avg = lambda A: d3.Average(A, coords.coords[0])
    s2_avg = lambda A: d3.Average(A, coords.S2coordsys)

    ##TODO: revisit outputs
    for bn, basis in bases.items():
        phi, theta, r = itemgetter('phi_'+bn, 'theta_'+bn, 'r_'+bn)(variables)
        phi1, theta1, r1 = itemgetter('phi1_'+bn, 'theta1_'+bn, 'r1_'+bn)(variables)
        ex, ey, ez = itemgetter('ex_'+bn, 'ey_'+bn, 'ez_'+bn)(variables)
        T, ρ = itemgetter('T_{}'.format(bn), 'ρ_{}'.format(bn))(variables)
        div_u, E = itemgetter('div_u_RHS_{}'.format(bn), 'E_RHS_{}'.format(bn))(variables)
        u = variables['u_{}'.format(bn)]
        p = variables['p_{}'.format(bn)]
        s1 = variables['s1_{}'.format(bn)]

        variables['r_vec_{}'.format(bn)] = r_vec = dist.VectorField(coords, name='r_vec_{}'.format(bn), bases=basis)
        variables['r_vals_{}'.format(bn)] = r_vals = dist.Field(name='r_vals_{}'.format(bn), bases=basis)
        r_vals['g'] = r1
        r_vec['g'][2] = r1
        r_vals = d3.Grid(r_vals).evaluate()
        er = d3.Grid(variables['er_{}'.format(bn)]).evaluate()

        u_squared = d3.dot(u, u)
        ur = d3.dot(er, u)
        pomega_hat = p - 0.5*u_squared
        h = pomega_hat + T*s1
        visc_flux = 2*(d3.dot(u, E) - (1/3) * u * div_u)
        visc_flux_r = d3.dot(er, visc_flux)

        angular_momentum = d3.cross(r_vec, ρ*u)
        am_Lx = d3.dot(ex, angular_momentum)
        am_Ly = d3.dot(ey, angular_momentum)
        am_Lz = d3.dot(ez, angular_momentum)

        if type(basis) == d3.BallBasis:
            volume  = (4/3)*np.pi*Ri**3
        else:
            volume  = (4/3)*np.pi*(Ro**3-Ri**3)

        vol_avg = variables['vol_avg_{}'.format(bn)] = lambda A: d3.Integrate(A/volume, coords)
        lum_prof = variables['lum_prof_{}'.format(bn)] = lambda A: s2_avg((4*np.pi*r_vals**2) * A)

        # Add slices for making movies
        slices.add_task(u(theta=np.pi/2), name='u_eq_{}'.format(bn), layout='g')
        slices.add_task(s1(theta=np.pi/2), name='s1_eq_{}'.format(bn), layout='g')

        if type(basis) == d3.BallBasis:
            radius_vals = (0.5, 1)
            radius_strs = ('0.5', '1')
        else:
            radius_vals = (0.95*Ro,)
            radius_strs = ('0.95R',)
        for r_val, r_str in zip(radius_vals, radius_strs):
                slices.add_task(u(r=r_val), name='u_{}(r={})'.format(bn, r_str), layout='g')
                slices.add_task(s1(r=r_val), name='s1_{}(r={})'.format(bn, r_str), layout='g')

        for az_val, phi_str in zip([0, np.pi/2, np.pi, 3*np.pi/2], ['0', '0.5*pi', 'pi', '1.5*pi',]):
            slices.add_task(u(phi=az_val),  name='u_{}(phi={})'.format(bn, phi_str), layout='g')
            slices.add_task(s1(phi=az_val), name='s1_{}(phi={})'.format(bn, phi_str), layout='g')

        # Add scalars for simple evolution tracking
        scalars.add_task(vol_avg(Re*(u_squared)**(1/2)), name='Re_avg_{}'.format(bn),  layout='g')
        scalars.add_task(vol_avg(ρ*u_squared/2), name='KE_{}'.format(bn),   layout='g')
        scalars.add_task(vol_avg(ρ*T*s1), name='TE_{}'.format(bn),  layout='g')
        scalars.add_task(vol_avg(am_Lx), name='angular_momentum_x_{}'.format(bn), layout='g')
        scalars.add_task(vol_avg(am_Ly), name='angular_momentum_y_{}'.format(bn), layout='g')
        scalars.add_task(vol_avg(am_Lz), name='angular_momentum_z_{}'.format(bn), layout='g')
        scalars.add_task(vol_avg(d3.dot(angular_momentum, angular_momentum)), name='square_angular_momentum_{}'.format(bn), layout='g')

        # Add profiles to track structure and fluxes
        profiles.add_task(s2_avg(s1), name='s1_profile_{}'.format(bn), layout='g')
        profiles.add_task(lum_prof(ρ*ur*pomega_hat),   name='wave_lum_{}'.format(bn), layout='g')
        profiles.add_task(lum_prof(ρ*ur*h),            name='enth_lum_{}'.format(bn), layout='g')
        profiles.add_task(lum_prof(-ρ*visc_flux_r/Re), name='visc_lum_{}'.format(bn), layout='g')
        profiles.add_task(lum_prof(-ρ*T*d3.dot(er, d3.grad(s1)/Pe)), name='cond_lum_{}'.format(bn), layout='g')
        profiles.add_task(lum_prof(0.5*ρ*ur*u_squared), name='KE_lum_{}'.format(bn),   layout='g')

        # Output high-cadence S2 shells for wave output tasks
        if args['--sponge']:
            if type(basis) == d3.BallBasis:
                radius_vals = (0.90, 1.05)
                radius_strs = ('0.90', '1.05')
            else:
                radius_vals = (1.5, 2.0, 2.5, 3.0, 3.5)
                radius_strs = ('1.50', '2.00', '2.50', '3.00', '3.50')

            for r_val, r_str in zip(radius_vals, radius_strs):
                    surface_shell_slices.add_task(ur(r=r_val),         name='ur_{}(r={})'.format(bn, r_str), layout='g')
                    surface_shell_slices.add_task(pomega_hat(r=r_val), name='pomega_{}(r={})'.format(bn, r_str), layout='g')
        else:
            if type(basis) == d3.BallBasis:
                pass
            surface_shell_slices.add_task(s1(r=Ro), name='s1_{}(r=Ro)'.format(bn), layout='g')

        logger_handler.add_task(vol_avg(Re*(u_squared)**(1/2)), name='Re_avg_{}'.format(bn), layout='g')
        logger_handler.add_task(d3.integ(ρ*(u_squared)/2), name='KE_{}'.format(bn), layout='g')

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
            if max_dt_check and timestep < outer_shell_dt:
                #throttle max_dt timestep CFL early in simulation once timestep is below the output cadence.
                my_cfl.max_dt = max_dt
                max_dt_check = False
                just_wrote = True
                slice_time = solver.sim_time + outer_shell_dt

            timestep = my_cfl.compute_timestep()

            if just_wrote:
                just_wrote = False
                num_steps = np.ceil(outer_shell_dt / timestep)
                timestep = current_max_dt = my_cfl.stored_dt = outer_shell_dt/num_steps
            elif max_dt_check:
                timestep = np.min((timestep, current_max_dt))
            else:
                my_cfl.stored_dt = timestep = current_max_dt

            t_future = solver.sim_time + timestep
            if t_future >= slice_time*(1-1e-8):
               slice_process = True

            if solver.iteration % hermitian_cadence in timestepper_history:
                for f in solver.state:
                    f.require_grid_space()

            solver.step(timestep)

            if solver.iteration % 10 == 0 or solver.iteration <= 10:
                Re_avg = logger_handler.fields['Re_avg_B']
                KE_shell = logger_handler.fields['KE_S1']
                if dist.comm_cart.rank == 0:
                    KE0 = KE_shell['g'].min()
                    Re0 = Re_avg['g'].min()
                else:
                    KE0 = None
                    Re0 = None
                Re0 = dist.comm_cart.bcast(Re0, root=0)
                KE0 = dist.comm_cart.bcast(KE0, root=0)
                this_str = "iteration = {:08d}, t = {:f}, timestep = {:f}, Re = {:.4e}".format(solver.iteration, solver.sim_time, timestep, Re0)
                this_str += ", KE = {:.4e}".format(KE0)
                logger.info(this_str)


            if slice_process:
                slice_process = False
                wall_time = time.time() - solver.start_time
                solver.evaluator.evaluate_handlers([surface_shell_slices],wall_time=wall_time, sim_time=solver.sim_time, iteration=solver.iteration,world_time = time.time(),timestep=timestep)
                slice_time = solver.sim_time + outer_shell_dt
                just_wrote = True

            if np.isnan(Re0):
                logger.info('exiting with NaN')
                break

    except:
        logger.info('something went wrong in main loop.')
        raise
    finally:
        solver.log_stats()

        logger.info('making final checkpoint')
        fcheckpoint = solver.evaluator.add_file_handler('{:s}/final_checkpoint'.format(out_dir), max_writes=1, sim_dt=10*t_buoy)
        fcheckpoint.add_tasks(solver.state, layout='g')
        solver.step(timestep)

        logger.info('Stitching open virtual files...')
        for handler in analysis_tasks:
            if not handler.check_file_limits():
                file = handler.get_file()
                file.close()
                if dist.comm_cart.rank == 0:
                    handler.process_virtual_file()

