"""
d3 script for anelastic convection in a massive star's core CZ.

A config file can be provided which overwrites any number of the options below, but command line flags can also be used if preferred.

Usage:
    bootstrap_rrbc.py [options]
    bootstrap_rrbc.py <config> [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 1e2]
    --Pr=<Prandtl>       The Prandtl number  of the numerical diffusivities [default: 1]
    --nz=<nz>            Chebyshev modes  [default: 64]
    --nx=<nx>            Fourier modes    [default: 64]
    --aspect=<a>         Asepect Ratio    [default: 4]

    --wall_hours=<t>     The number of hours to run for [default: 24]
    --buoy_end_time=<t>  Number of buoyancy times to run [default: 1e5]
    --safety=<s>         Timestep CFL safety factor [default: 0.4]

    --mesh=<n,m>         The processor mesh over which to distribute the cores
    --A0=<A>             Amplitude of initial noise [default: 1e-6]

    --label=<label>      A label to add to the end of the output directory

    --SBDF2              Use SBDF2 (default)
    --SBDF4              Use SBDF4

    --mesa_file=<f>      path to a .h5 file of ICCs, curated from a MESA model
    --restart=<chk_f>    path to a checkpoint file to restart from
    --seed=<s>           rng seed [default: 42]
"""
import logging
import os
import sys
import time

import numpy as np
from docopt import docopt
from mpi4py import MPI

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post
from dedalus.tools.config import config
logger = logging.getLogger(__name__)


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


def filter_field(field, frac=0.25):
    """
    Filter a field in coefficient space by cutting off all coefficient above
    a given threshold.  This is accomplished by changing the scale of a field,
    forcing it into coefficient space at that small scale, then coming back to
    the original scale.

    Inputs:
        field   - The dedalus field to filter
        frac    - The fraction of coefficients to KEEP POWER IN.  If frac=0.25,
                    The upper 75% of coefficients are set to 0.
    """
    dom = field.domain
    logger.info("filtering field {} with frac={} using a set-scales approach".format(field.name,frac))
    orig_scale = field.scales
    field.set_scales(frac, keep_data=True)
    field['c']
    field['g']
    field.set_scales(orig_scale, keep_data=True)


def global_noise(domain, seed=42, n_modes=None, **kwargs):
    """
    Create a field filled with random noise of order 1.  

    Arguments:
    ----------
    seed : int, optional
        The seed for the random number generator; change it to get a different noise field.
    n_modes : int, optional
        The number of chebyshev modes to fill in the noise field.
    """
    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=1)
    slices = domain.dist.grid_layout.slices(scales=1)
    rand  = np.random.RandomState(seed=seed)
    noise_field = domain.new_field()

    if n_modes is None:
        noise = rand.standard_normal(gshape)[slices]

        # filter in k-space
        noise_field.set_scales(1, keep_data=False)
        noise_field['g'] = noise
        filter_field(noise_field, **kwargs)
    else:
        n_modes = int(n_modes)
        scale   = n_modes/gshape[-1]
        gshape_small = domain.dist.grid_layout.global_shape(scales=scale)
        slices_small = domain.dist.grid_layout.slices(scales=scale)
        noise = rand.standard_normal(gshape_small)[slices_small]

        noise_field.set_scales(scale, keep_data=False)
        noise_field['g'] = noise

    noise_field.set_scales(domain.dealias, keep_data=True)
        
    return noise_field





# Parameters
radius = Lz = 1
nx     = int(args['--nx'])
nz     = int(args['--nz'])
aspect = Lx = float(args['--aspect'])

out_dir = './' + sys.argv[0].split('.py')[0]
out_dir += '_Re{}_{}x{}/'.format(args['--Re'], args['--nx'], args['--nz'])
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(out_dir)):
        os.makedirs('{:s}/'.format(out_dir))



if args['--SBDF4']:
    ts = de.timesteppers.SBDF4
    timestepper_history = [0, 1, 2, 3]
else:
    ts = de.timesteppers.SBDF2
    timestepper_history = [0, 1,]

mesh = args['--mesh']
if mesh is not None:
    mesh = [int(m) for m in mesh.split(',')]

Re  = float(args['--Re'])
Pr  = 1
Pe  = Pr*Re

# Bases
x_basis = de.Fourier( 'x', nx, interval = [0, Lx], dealias=3/2)
z_basis = de.Chebyshev('z', nz, interval = [0, Lz], dealias=3/2)

bases = [x_basis, z_basis]
domain = de.Domain(bases, grid_dtype=np.float64, mesh=mesh)
r = z = domain.grid(-1)



#nccs
grad_ln_ρ  = domain.new_field()
ln_ρ  = domain.new_field()
ln_T  = domain.new_field()
inv_T = domain.new_field()
H_eff = domain.new_field()
g_eff = domain.new_field()

gslices = domain.dist.grid_layout.slices(scales=1)


if args['--mesa_file'] is not None:
    import h5py
    r_slice = gslices[-1]
    with h5py.File(args['--mesa_file'], 'r') as f:
        ln_ρ['g']      = f['ln_ρ'][()][r_slice]
        ln_T['g']      = f['ln_T'][()][r_slice]
        H_eff['g']     = 4*np.pi*r**2*f['H_eff'][()][r_slice]
        inv_T['g']     = f['inv_T'][()][r_slice]
        g_eff['g']     = f['g_eff'][()][r_slice]
        ln_ρ.differentiate('z', out=grad_ln_ρ)

        t_buoy = np.sqrt(1/f['g_eff'][()].max())
else:
    logger.error("Must specify an initial condition file")
    import sys
    sys.exit()



variables = ['s1', 's1_z', 'p', 'u', 'u_z', 'w', 'w_z']
problem = de.IVP(domain, variables=variables, ncc_cutoff=1e-10)

logger.info('buoyancy time is {}'.format(t_buoy))
max_dt = 0.5*t_buoy


for f in [grad_ln_ρ, ln_ρ, ln_T, inv_T, H_eff, g_eff]:
    f.meta['x']['constant'] = True

problem.parameters['ln_ρ']   = ln_ρ
problem.parameters['grad_ln_ρ']   = grad_ln_ρ
problem.parameters['ln_T']   = ln_T
problem.parameters['inv_T']  = inv_T
problem.parameters['H_eff']  = H_eff
problem.parameters['g_eff']  = g_eff
problem.parameters['Pe'] = Pe
problem.parameters['Re'] = Re
problem.parameters['Lx'] = Lx
problem.parameters['Lz'] = Lz

problem.substitutions['Lap(A, A_z)']=       '(dx(dx(A)) + dz(A_z))'
problem.substitutions['UdotGrad(A, A_z)'] = '(u*dx(A) + w*A_z)'

problem.substitutions['dy(A)'] = '0'
problem.substitutions['Ox'] = '0'
problem.substitutions['Oz'] = '0'
problem.substitutions['v'] = '0'
problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'
problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'
problem.substitutions['DivU'] = '(dx(u) + w_z)'

problem.substitutions['vel_rms'] = 'sqrt(u**2 + v**2 + w**2)'

problem.substitutions['Re_rms'] = '(vel_rms * Re)'
problem.substitutions['Pe_rms'] = '(vel_rms * Pe)'

problem.substitutions['σ_xx'] = '(2*dx(u))'
problem.substitutions['σ_xz'] = '(dx(w) + u_z)'
problem.substitutions['σ_zz'] = '(2*w_z)'

problem.substitutions['tr_σ'] = '(σ_xx + σ_xz)'
problem.substitutions['tr_σ_dot_σ'] = '(σ_xx**2 + σ_xz**2)'

problem.substitutions['viscous_x'] = '(dx(σ_xx) + dz(σ_xz) + dz(ln_ρ)*σ_xz - (2/3)*(dx(DivU)))'
problem.substitutions['viscous_z'] = '(dx(σ_xz) + dz(σ_zz) + dz(ln_ρ)*σ_zz - (2/3)*(dz(DivU) + DivU*dz(ln_ρ)))'

problem.substitutions['VH'] = 'tr_σ_dot_σ - (1/3)*(tr_σ)**2'


### 4.Setup equations and Boundary Conditions
problem.add_equation("DivU + w*grad_ln_ρ = 0")
problem.add_equation("dt(u) + dx(p)             - (1/Re)*viscous_x = - UdotGrad(u, u_z)")
problem.add_equation("dt(w) + dz(p) - g_eff*s1  - (1/Re)*viscous_z = - UdotGrad(w, w_z)")
problem.add_equation("dt(s1) - (1/Pe)*(dx(dx(s1)) + dz(s1_z) + s1_z*dz(ln_ρ + ln_T)) = -UdotGrad(s1, s1_z) + H_eff + (1/Re)*inv_T*VH")
problem.add_equation("dz(u) - u_z = 0")
problem.add_equation("dz(w) - w_z = 0")
problem.add_equation("dz(s1) - s1_z = 0")

problem.add_bc(" left(w) = 0")
problem.add_bc("right(w) = 0", condition="nx != 0")
problem.add_bc("right(p) = 0", condition="nx == 0")
problem.add_bc(" left(u_z) = 0")
problem.add_bc("right(u_z) = 0")
problem.add_bc(" left(s1) = 0")
problem.add_bc("right(s1) = 0")

print("Problem built")

# Solver
cfl_safety = float(args['--safety'])
solver = problem.build_solver(ts)
logger.info('Solver built')



### 6. Set initial conditions: noise or loaded checkpoint
restart = args['--restart']
if restart is None:
    p = solver.state['p']
    s1 = solver.state['s1']
    s1_z = solver.state['s1_z']
    p.set_scales(domain.dealias)
    s1.set_scales(domain.dealias)
    s1_z.set_scales(domain.dealias)
    z_de = domain.grid(-1, scales=domain.dealias)

    A0 = 1e-6

       

    #Add noise kick
    noise = global_noise(domain, int(args['--seed']))
    s1['g'] += A0*np.cos(np.pi*z_de)*noise['g']
    s1.differentiate('z', out=s1_z)


    dt = None
    mode = 'overwrite'
else:
    logger.info("not implemented")
   

### 7. Set simulation stop parameters, output, and CFL
t_end = float(args['--buoy_end_time'])*t_buoy
solver.stop_sim_time = t_end
solver.stop_wall_time = 24*3600.

max_dt    = np.min((0.5*t_buoy, 1))
if dt is None: dt = max_dt

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=cfl_safety,
                     max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.1)
CFL.add_velocities(('u', 'w'))


# Output
output_dt = t_buoy/5
out_iter = np.inf
from collections import OrderedDict
analysis_tasks = OrderedDict()
profiles = solver.evaluator.add_file_handler(out_dir+'profiles', sim_dt=output_dt, max_writes=200, mode=mode, iter=out_iter)
profiles.add_task("plane_avg(s1)", name="s1")
profiles.add_task("plane_avg(u)", name="u")
profiles.add_task("plane_avg(w)", name="w")
profiles.add_task("plane_avg(exp(ln_ρ)*w*s1)", name="enth_flux")
profiles.add_task("plane_avg(exp(ln_ρ)*exp(ln_T)*dz(s1)/Pe)", name="kappa_flux")

analysis_tasks['profiles'] = profiles

scalar = solver.evaluator.add_file_handler(out_dir+'scalar', sim_dt=output_dt, max_writes=np.inf, mode=mode, iter=out_iter)
scalar.add_task("vol_avg(s1)", name="s1")
scalar.add_task("vol_avg(0.5*exp(ln_ρ)*vel_rms**2)", name="KE")
scalar.add_task("vol_avg(Re_rms)", name="Re_rms")
scalar.add_task("vol_avg(Pe_rms)", name="Pe_rms")
scalar.add_task("vol_avg(u)",  name="u")
scalar.add_task("vol_avg(w)",  name="w")
analysis_tasks['scalar'] = scalar

slices = solver.evaluator.add_file_handler(out_dir+'slices', sim_dt=output_dt, max_writes=40, mode=mode, iter=out_iter)
slices.add_task('s1')
slices.add_task('u')
slices.add_task('w')

checkpoint = solver.evaluator.add_file_handler(out_dir+'checkpoint', sim_dt=100*t_buoy, max_writes=1, mode=mode)
checkpoint.add_system(solver.state, layout='c')




### 8. Setup flow tracking for terminal output, including rolling averages
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("Re_rms", name='Re_rms')
flow.add_property("s1", name='s1')

Hermitian_cadence = 100
# Main loop
try:
    count = Re_avg = 0
    logger.info('Starting loop')
    init_time = last_time = solver.sim_time
    start_iter = solver.iteration
    start_time = time.time()
    while (solver.ok and np.isfinite(Re_avg)) or first_step:
        dt = CFL.compute_dt()
        solver.step(dt) #, trim=True)

        if solver.iteration % 10 == 0:
            Re_avg = flow.grid_average('Re_rms')
            log_string =  'Iteration: {:5d}, '.format(solver.iteration)
            log_string += 'Time: {:8.3e}, dt: {:8.3e}, '.format(solver.sim_time,  dt)
            log_string += 'Re: {:8.3e}/{:8.3e}, '.format(Re_avg, flow.max('Re_rms'))
            log_string += 's1: {:8.3e}, '.format(flow.grid_average('s1'))
            logger.info(log_string)
except:
    raise
    logger.error('Exception raised, triggering end of main loop.')
finally:
    end_time = time.time()
    main_loop_time = end_time-start_time
    n_iter_loop = solver.iteration-1
    logger.info('Iterations: {:d}'.format(n_iter_loop))
    logger.info('Sim end time: {:f}'.format(solver.sim_time))
    logger.info('Run time: {:f} sec'.format(main_loop_time))
    logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*domain.dist.comm_cart.size))
    logger.info('iter/sec: {:f} (main loop only)'.format(n_iter_loop/main_loop_time))

    logger.info('beginning join operation')
    post.merge_analysis(out_dir+'checkpoint')

    for key, task in analysis_tasks.items():
        logger.info(task.base_path)
        post.merge_analysis(task.base_path)

    logger.info(40*"=")
    logger.info('Iterations: {:d}'.format(n_iter_loop))
    logger.info('Sim end time: {:f}'.format(solver.sim_time))
    logger.info('Run time: {:f} sec'.format(main_loop_time))
    logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*domain.dist.comm_cart.size))
    logger.info('iter/sec: {:f} (main loop only)'.format(n_iter_loop/main_loop_time))
