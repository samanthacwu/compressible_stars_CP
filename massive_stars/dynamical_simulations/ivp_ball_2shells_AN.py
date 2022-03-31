"""
d3 script for anelastic convection in a stitched BallBasis and ShellBasis domain.

A config file can be provided which overwrites any number of the options below, but command line flags can also be used if preferred.
Note that options specified in a cnofig file override command line arguments.

Usage:
    ivp_ball_2shells_AN.py [options]
    ivp_ball_2shells_AN.py <config> [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 2e2]
    --Pr=<Prandtl>       The Prandtl number  of the numerical diffusivities [default: 1]
    --ntheta=<res>       Number of theta grid points (Lmax+1)   [default: 4]
    --nrB=<res>          Number of radial grid points in ball (Nmax+1)   [default: 24]
    --nrS1=<res>          Number of radial grid points in first shell (Nmax+1)   [default: 8]
    --nrS2=<res>          Number of radial grid points in second shell (Nmax+1)   [default: 8]
    --sponge             If flagged, add a damping layer in the shell that damps out waves.
    --tau_factor=<f>     Multiplication factor on sponge term [default: 1]

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
    --cutoff=<c>         NCC cutoff magnitude [default: 1e-8]

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

from anelastic_functions import make_bases, make_fields, fill_structure, get_anelastic_variables, set_anelastic_problem

# Define smooth Heaviside functions
from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

if __name__ == '__main__':

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
    nrS1 = int(args['--nrS1'])
    nrS2 = int(args['--nrS2'])
    resolutionB = (nphi, ntheta, nrB)
    resolutionS1 = (nphi, ntheta, nrS1)
    resolutionS2 = (nphi, ntheta, nrS2)
    L_dealias = N_dealias = dealias = 1.5
    dtype = np.float64
    Re  = float(args['--Re'])
    Pr  = 1
    Pe  = Pr*Re
    ncc_file = args['--ncc_file']
    wall_hours = float(args['--wall_hours'])
    buoy_end_time = float(args['--buoy_end_time'])
    sponge = args['--sponge']
    tau_factor = float(args['--tau_factor'])

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
        out_dir += '_sponge_tauF{}'.format(args['--tau_factor'])
    if ncc_file is None:
        out_dir += '_polytrope'
    if do_rotation:
        out_dir += '_rotation{}'.format(rotation_time)

    out_dir += '_Re{}_{}x{}x{}+{}+{}'.format(args['--Re'], nphi, ntheta, nrB, nrS1, nrS2)
    if args['--label'] is not None:
        out_dir += '_{:s}'.format(args['--label'])
    logger.info('saving data to {:s}'.format(out_dir))
    if MPI.COMM_WORLD.rank == 0:
        if not os.path.exists('{:s}/'.format(out_dir)):
            os.makedirs('{:s}/'.format(out_dir))

    # Read in domain bound values
    if ncc_file is not None:
        with h5py.File(args['--ncc_file'], 'r') as f:
            r_stitch = f['r_stitch'][()]
            r_outer = f['r_outer'][()]
            Re_shift = f['Re_shift'][()]
        Re *= Re_shift
        Pe *= Re_shift
    else:
        r_stitch = (1.1, 1.4)
        r_outer = 1.5
    logger.info('r_stitch: {} / r_outer: {:.2f}'.format(r_stitch, r_outer))
    L_shell = r_outer - r_stitch[0]
    sponge_function = lambda r: zero_to_one(r, r_stitch[0] + 2*L_shell/3, 0.1*L_shell)

    resolutions = (resolutionB, resolutionS1, resolutionS2)
    stitch_radii = r_stitch
    radius = r_outer
    coords, dist, bases, bases_keys = make_bases(resolutions, stitch_radii, radius, dealias=(L_dealias, L_dealias, N_dealias), dtype=dtype, mesh=mesh)

    vec_fields = ['u',]
    scalar_fields = ['p', 's1', 'inv_T', 'H', 'rho', 'T']
    vec_taus = ['tau_u']
    scalar_taus = ['tau_s']
    vec_nccs = ['grad_ln_rho', 'grad_ln_T', 'grad_s0', 'grad_T', 'grad_chi_rad']
    scalar_nccs = ['ln_rho', 'ln_T', 'chi_rad', 'sponge']
    variables = make_fields(bases, coords, dist, 
                            vec_fields=vec_fields, scalar_fields=scalar_fields, 
                            vec_taus=vec_taus, scalar_taus=scalar_taus, 
                            vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                            sponge=sponge, do_rotation=do_rotation, sponge_function=sponge_function)


    variables, timescales = fill_structure(bases, dist, variables, ncc_file, r_outer, Pe,
                                            vec_fields=vec_fields, vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                                            sponge=sponge, do_rotation=do_rotation)

    for i, bn in enumerate(bases.keys()):
        variables['sponge_{}'.format(bn)]['g'] *= tau_factor
    max_dt, t_buoy, t_rot = timescales

    # Put nccs and fields into locals()
    locals().update(variables)


    # Problem
    prob_variables = get_anelastic_variables(bases, bases_keys, variables)
    problem = d3.IVP(prob_variables, namespace=locals())

    problem = set_anelastic_problem(problem, bases, bases_keys, stitch_radii=stitch_radii)

    logger.info("Problem built")
    # Solver
    solver = problem.build_solver(ts, ncc_cutoff=float(args['--cutoff']))
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
        s1_S2.fill_random(layout='g', seed=42, distribution='normal', scale=A0)
        s1_S2.low_pass_filter(scales=0.25)
        s1_B['g'] *= np.sin(theta1_B)
        s1_B['g'] *= np.sin(theta1_B)
        s1_S1['g'] *= np.sin(theta1_S1)
        s1_S2['g'] *= np.sin(theta1_S2)
        s1_B['g'] *= np.cos(np.pi*r1_B/r_outer)
        s1_S1['g'] *= np.cos(np.pi*r1_S1/r_outer)
        s1_S2['g'] *= np.cos(np.pi*r1_S2/r_outer)

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

    for bn, basis in bases.items():
        phi, theta, r = itemgetter('phi_'+bn, 'theta_'+bn, 'r_'+bn)(variables)
        phi1, theta1, r1 = itemgetter('phi1_'+bn, 'theta1_'+bn, 'r1_'+bn)(variables)
        ex, ey, ez = itemgetter('ex_'+bn, 'ey_'+bn, 'ez_'+bn)(variables)
        T, rho = itemgetter('T_{}'.format(bn), 'rho_{}'.format(bn))(variables)
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

        angular_momentum = d3.cross(r_vec, rho*u)
        am_Lx = d3.dot(ex, angular_momentum)
        am_Ly = d3.dot(ey, angular_momentum)
        am_Lz = d3.dot(ez, angular_momentum)

        if type(basis) == d3.BallBasis:
            volume  = (4/3)*np.pi*r_stitch[0]**3
        else:
            index = int(bn.split('S')[-1])-1
            Ri = r_stitch[index]
            volume  = (4/3)*np.pi*(r_outer**3-Ri**3)

        vol_avg = variables['vol_avg_{}'.format(bn)] = lambda A: d3.Integrate(A/volume, coords)
        lum_prof = variables['lum_prof_{}'.format(bn)] = lambda A: s2_avg((4*np.pi*r_vals**2) * A)

        # Add slices for making movies
        slices.add_task(u(theta=np.pi/2), name='u_eq_{}'.format(bn), layout='g')
        slices.add_task(s1(theta=np.pi/2), name='s1_eq_{}'.format(bn), layout='g')

        if type(basis) == d3.BallBasis:
            radius_vals = (0.5, 1)
            radius_strs = ('0.5', '1')
        else:
            value = 0.95*r_outer
            if basis.radii[0] <= value and basis.radii[1] >= value:
                radius_vals = (value,)
                radius_strs = ('0.95R',)
        for r_val, r_str in zip(radius_vals, radius_strs):
                slices.add_task(u(r=r_val), name='u_{}(r={})'.format(bn, r_str), layout='g')
                slices.add_task(s1(r=r_val), name='s1_{}(r={})'.format(bn, r_str), layout='g')

        for az_val, phi_str in zip([0, np.pi/2, np.pi, 3*np.pi/2], ['0', '0.5*pi', 'pi', '1.5*pi',]):
            slices.add_task(u(phi=az_val),  name='u_{}(phi={})'.format(bn, phi_str), layout='g')
            slices.add_task(s1(phi=az_val), name='s1_{}(phi={})'.format(bn, phi_str), layout='g')

        # Add scalars for simple evolution tracking
        scalars.add_task(vol_avg(Re*(u_squared)**(1/2)), name='Re_avg_{}'.format(bn),  layout='g')
        scalars.add_task(vol_avg(rho*u_squared/2), name='KE_{}'.format(bn),   layout='g')
        scalars.add_task(vol_avg(rho*T*s1), name='TE_{}'.format(bn),  layout='g')
        scalars.add_task(vol_avg(am_Lx), name='angular_momentum_x_{}'.format(bn), layout='g')
        scalars.add_task(vol_avg(am_Ly), name='angular_momentum_y_{}'.format(bn), layout='g')
        scalars.add_task(vol_avg(am_Lz), name='angular_momentum_z_{}'.format(bn), layout='g')
        scalars.add_task(vol_avg(d3.dot(angular_momentum, angular_momentum)), name='square_angular_momentum_{}'.format(bn), layout='g')

        # Add profiles to track structure and fluxes
        profiles.add_task(s2_avg(s1), name='s1_profile_{}'.format(bn), layout='g')
        profiles.add_task(lum_prof(rho*ur*pomega_hat),   name='wave_lum_{}'.format(bn), layout='g')
        profiles.add_task(lum_prof(rho*ur*h),            name='enth_lum_{}'.format(bn), layout='g')
        profiles.add_task(lum_prof(-rho*visc_flux_r/Re), name='visc_lum_{}'.format(bn), layout='g')
        profiles.add_task(lum_prof(-rho*T*d3.dot(er, d3.grad(s1)/Pe)), name='cond_lum_{}'.format(bn), layout='g')
        profiles.add_task(lum_prof(0.5*rho*ur*u_squared), name='KE_lum_{}'.format(bn),   layout='g')

        # Output high-cadence S2 shells for wave output tasks
        if args['--sponge']:
            if type(basis) == d3.BallBasis:
                radius_vals = (0.90, 1.05)
                radius_strs = ('0.90', '1.05')
            else:
                global_radius_vals = (1.5, 2.0, 2.5, 3.0, 3.5)
                global_radius_strs = ('1.50', '2.00', '2.50', '3.00', '3.50')
                radius_vals = []
                radius_strs = []
                for i, rv in enumerate(global_radius_vals):
                    if basis.radii[0] <= rv and basis.radii[1] >= rv:
                        radius_vals.append(rv)
                        radius_strs.append(global_radius_strs[i])

            for r_val, r_str in zip(radius_vals, radius_strs):
                    surface_shell_slices.add_task(ur(r=r_val),         name='ur_{}(r={})'.format(bn, r_str), layout='g')
                    surface_shell_slices.add_task(pomega_hat(r=r_val), name='pomega_{}(r={})'.format(bn, r_str), layout='g')
        else:
            if type(basis) != d3.BallBasis:
                if basis.radii[1] == r_outer:
                    surface_shell_slices.add_task(s1(r=r_outer), name='s1_{}(r=r_outer)'.format(bn), layout='g')

        logger_handler.add_task(vol_avg(Re*(u_squared)**(1/2)), name='Re_avg_{}'.format(bn), layout='g')
        logger_handler.add_task(d3.integ(rho*(u_squared)/2), name='KE_{}'.format(bn), layout='g')

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

