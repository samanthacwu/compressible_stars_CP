"""
d3 script for anelastic convection in a massive star's core CZ.

A config file can be provided which overwrites any number of the options below, but command line flags can also be used if preferred.

Usage:
    annulus_coreCZ_AN.py [options]
    annulus_coreCZ_AN.py <config> [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 1e2]
    --Pr=<Prandtl>       The Prandtl number  of the numerical diffusivities [default: 1]
    --nr=<nr>            Chebyshev modes  [default: 64]
    --nphi=<nphi>        Fourier modes    [default: 64]
    --Lbot=<Lbot>        r position at bottom of domain [default: 0.5]

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
from pathlib import Path
from configparser import ConfigParser

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
Lbot   = float(args['--Lbot'])
radius = Lr = 1
nφ     = int(args['--nphi'])
nr     = int(args['--nr'])

out_dir = './' + sys.argv[0].split('.py')[0]
if args['--mesa_file'] is not None and 'polytrope' in args['--mesa_file']:
    out_dir += '_polytrope'
out_dir += '_Re{}_{}x{}'.format(args['--Re'], args['--nphi'], args['--nr'])
if args['--label'] is not None:
    out_dir += '_{}'.format(args['--label'])
out_dir += '/'
logger.info('saving to {}'.format(out_dir))
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
φ_basis = de.Fourier( 'φ', nφ, interval = [0, 2*np.pi], dealias=3/2)
r_basis = de.Chebyshev('r', nr, interval = [Lbot, Lbot+Lr], dealias=3/2)

bases = [φ_basis, r_basis]
domain = de.Domain(bases, grid_dtype=np.float64, mesh=mesh)
r = z = domain.grid(-1)



#nccs
grad_ln_ρ  = domain.new_field()
grad_ln_T  = domain.new_field()
ln_ρ  = domain.new_field()
ln_T  = domain.new_field()
T = domain.new_field()
H_eff = domain.new_field()

gamma  = domain.new_field()
gamma['g'] = 5./3

gslices = domain.dist.grid_layout.slices(scales=1)


if args['--mesa_file'] is not None:
    import h5py
    r_slice = gslices[-1]
    with h5py.File(args['--mesa_file'], 'r') as f:
        ln_ρ['g']      = f['ln_ρ'][()][r_slice]
        ln_T['g']      = f['ln_T'][()][r_slice]
        if 'polytrope' in args['--mesa_file']:
            H_factor = 1
        else:
            logger.error("Only polytrope implemented")
            import sys
            sys.exit()
        H_eff['g']     = H_factor*f['H_eff'][()][r_slice]
        T['g']         = f['T'][()][r_slice]
        ln_ρ.differentiate('r', out=grad_ln_ρ)
        ln_T.differentiate('r', out=grad_ln_T)

        t_buoy = 1
else:
    logger.error("Must specify an initial condition file")
    import sys
    sys.exit()

#import matplotlib
#import matplotlib.pyplot as plt
#for f in [T, ln_ρ, H_eff]:
#    f.set_scales(1, keep_data=True)
#plt.plot(r[0,:], T['g'][0,:])
#plt.show()
#plt.plot(r[0,:], np.exp(ln_ρ['g'][0,:]))
#plt.show()
#plt.plot(r[0,:], H_eff['g'][0,:])
#plt.show()
#
#import sys
#sys.exit()

variables = ['s1', 's1_r', 'p', 'uφ', 'uφ_r', 'ur', 'ur_r']
problem = de.IVP(domain, variables=variables, ncc_cutoff=1e-10)

logger.info('buoyancy time is {}'.format(t_buoy))
max_dt = 0.5*t_buoy


for f in [grad_ln_ρ, ln_ρ, ln_T, grad_ln_T, T, H_eff, gamma]:
    f.meta['φ']['constant'] = True

problem.parameters['grad_ln_ρ']   = grad_ln_ρ
problem.parameters['grad_ln_T']   = grad_ln_T
problem.parameters['ln_ρ']        = ln_ρ
problem.parameters['ln_T']        = ln_T
problem.parameters['T']           = T
problem.parameters['H_eff']       = H_eff
problem.parameters['Pe']          = Pe
problem.parameters['Re']          = Re
problem.parameters['Lr']          = Lr
problem.parameters['Lbot']        = Lbot
problem.parameters['pi']          = np.pi

problem.parameters['gamma'] = gamma
problem.parameters['Cv']    = 1/(gamma-1)
problem.parameters['Cp']    = gamma/(gamma-1)

problem.substitutions['ρ'] = 'exp(ln_ρ)'

problem.substitutions['Lap_scalar(A, A_r)']       = 'dr(r*A_r)/r + dφ(dφ(A))/r**2'
problem.substitutions['Lap_r(Ar, Ar_r, Aφ)']      = 'Lap_scalar(Ar, Ar_r) - 2*(dφ(Aφ))/r**2 - Ar/r**2'
problem.substitutions['Lap_φ(Aφ, Aφ_r, Ar)']      = 'Lap_scalar(Aφ, Aφ_r) + 2*(dφ(Ar))/r**2 - Aφ/r**2'

problem.substitutions['UdotGrad(A, A_r)']         = '(ur*A_r + uφ*dφ(A)/r)'
problem.substitutions['UdotGrad_r(Ar, Ar_r, Aφ)'] = '(UdotGrad(Ar, Ar_r) - uφ*Aφ/r)'
problem.substitutions['UdotGrad_φ(Aφ, Aφ_r, Ar)'] = '(UdotGrad(Aφ, Aφ_r) + uφ*Ar/r)'

problem.substitutions['plane_avg(A)'] = 'integ(A, "φ")/(2*pi)'
problem.substitutions['vol_avg(A)']   = 'integ(r*A)/(pi)/((Lr+Lbot)**2 - Lbot**2)'
problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'
problem.substitutions['DivU']         = '(ur/r + ur_r + dφ(uφ)/r)'

problem.substitutions['vel_rms'] = 'sqrt(ur**2 + uφ**2)'

problem.substitutions['Re_rms'] = '(vel_rms * Re)'
problem.substitutions['Pe_rms'] = '(vel_rms * Pe)'

problem.substitutions['grad_u_rr'] = '(ur_r)'
problem.substitutions['grad_u_rφ'] = '(uφ_r)'
problem.substitutions['grad_u_φr'] = '(dφ(ur)/r - uφ/r)'
problem.substitutions['grad_u_φφ'] = '(dφ(uφ)/r + ur/r)'

problem.substitutions['E_rr'] = '0.5*(grad_u_rr + grad_u_rr)'
problem.substitutions['E_rφ'] = '0.5*(grad_u_rφ + grad_u_φr)'
problem.substitutions['E_φφ'] = '0.5*(grad_u_φφ + grad_u_φφ)'
problem.substitutions['σ_rr'] = '2*(E_rr - (1/3)*DivU)'
problem.substitutions['σ_rφ'] = '2*(E_rφ)'
problem.substitutions['σ_φφ'] = '2*(E_φφ - (1/3)*DivU)'

problem.substitutions['tr_E']       = '(E_rr + E_φφ)'
problem.substitutions['tr_E_dot_E'] = '(E_rr**2 + 2*E_rφ**2 + E_φφ**2)'

#viscous in momentum: tensor div(σ) + grad(ln_ρ) dot σ.
problem.substitutions['div_σ_r']   = '(dr(r*σ_rr)/r + dφ(σ_rφ)/r - σ_φφ/r)'
problem.substitutions['div_σ_φ']   = '(dr(r*σ_rφ)/r + dφ(σ_φφ)/r + σ_rφ/r)'
problem.substitutions['viscous_r'] = '(div_σ_r + grad_ln_ρ*σ_rr)'
problem.substitutions['viscous_φ'] = '(div_σ_φ + grad_ln_ρ*σ_rφ)'

problem.substitutions['VH'] = '2*(tr_E_dot_E - (1/3)*DivU**2)'

problem.substitutions['enth_flux'] = 'ρ*ur*(p)'
problem.substitutions['visc_flux'] = '-ρ*(σ_rr*ur + σ_rφ*uφ)/Re'
problem.substitutions['cond_flux'] = '-ρ*T*s1_r/Pe'
problem.substitutions['KE_flux']   = '0.5*ρ*ur*vel_rms**2'

### 4.Setup equations and Boundary Conditions
problem.add_equation("r*(DivU + ur*grad_ln_ρ) = 0", condition="nφ != 0")
problem.add_equation("p = 0", condition="nφ == 0")
problem.add_equation("r**2*(dt(ur) + dr(p)   - T*s1_r      - (1/Re)*viscous_r) = - r**2*UdotGrad_r(ur, ur_r, uφ)", condition="nφ != 0")
problem.add_equation("r**2*(dt(uφ) + dφ(p)/r - T*dφ(s1)/r  - (1/Re)*viscous_φ) = - r**2*UdotGrad_φ(uφ, uφ_r, ur)", condition="nφ != 0")
problem.add_equation("ur = 0", condition="nφ == 0")
problem.add_equation("uφ = 0", condition="nφ == 0")
problem.add_equation("r**2*(dt(s1) - (1/Pe)*(Lap_scalar(s1, s1_r) + s1_r*dr(ln_ρ + ln_T))) = r**2*(-UdotGrad(s1, s1_r) + H_eff + (1/(T*Re))*VH)")
problem.add_equation("dr(uφ) - uφ_r = 0", condition="nφ != 0")
problem.add_equation("dr(ur) - ur_r = 0", condition="nφ != 0")
problem.add_equation("uφ_r = 0", condition="nφ == 0")
problem.add_equation("ur_r = 0", condition="nφ == 0")
problem.add_equation("dr(s1) - s1_r = 0")

problem.add_bc(" left(ur) = 0", condition="nφ != 0")
problem.add_bc("right(ur) = 0", condition="nφ != 0")
#problem.add_bc("right(p) = 0", condition="nφ == 0")
problem.add_bc(" left(E_rφ) = 0", condition="nφ != 0")
problem.add_bc("right(E_rφ) = 0", condition="nφ != 0")
problem.add_bc(" left(s1_r) = 0")
problem.add_bc("right(s1)   = 0")

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
    s1_r = solver.state['s1_r']
    p.set_scales(domain.dealias)
    s1.set_scales(domain.dealias)
    s1_r.set_scales(domain.dealias)
    r_de = domain.grid(-1, scales=domain.dealias)

    A0 = 1e-6

       

    #Add noise kick
    noise = global_noise(domain, int(args['--seed']))
    s1['g'] += A0*np.cos(np.pi*(r_de-Lbot))*noise['g']
    s1.differentiate('r', out=s1_r)


    dt = None
    mode = 'overwrite'
else:
    logger.info("checkpoint restart not implemented")
    import sys
    sys.exit()
   

### 7. Set simulation stop parameters, output, and CFL
t_end = float(args['--buoy_end_time'])*t_buoy
solver.stop_sim_time = t_end
solver.stop_wall_time = 24*3600.

max_dt    = np.min((0.5*t_buoy, 1))
if dt is None: dt = max_dt

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=cfl_safety,
                     max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.1)
CFL.add_velocities(('uφ/r', 'ur'))


# Output
output_dt = t_buoy/2
out_iter = np.inf
from collections import OrderedDict
analysis_tasks = OrderedDict()
profiles = solver.evaluator.add_file_handler(out_dir+'profiles', sim_dt=output_dt, max_writes=200, mode=mode, iter=out_iter)
profiles.add_task("plane_avg(s1)", name="s1")
profiles.add_task("plane_avg(ur)", name="ur")
profiles.add_task("plane_avg(uφ)", name="uφ")
profiles.add_task("plane_avg(enth_flux)", name="enth_flux")
profiles.add_task("plane_avg(cond_flux)", name="cond_flux")
profiles.add_task("plane_avg(visc_flux)", name="visc_flux")
profiles.add_task("plane_avg(KE_flux)",   name="KE_flux")

analysis_tasks['profiles'] = profiles

scalar = solver.evaluator.add_file_handler(out_dir+'scalar', sim_dt=output_dt, max_writes=np.inf, mode=mode, iter=out_iter)
scalar.add_task("vol_avg(ρ*T*s1)",           name="TE")
scalar.add_task("vol_avg(0.5*ρ*vel_rms**2)", name="KE")
scalar.add_task("vol_avg(Re_rms)", name="Re_rms")
scalar.add_task("vol_avg(Pe_rms)", name="Pe_rms")
scalar.add_task("vol_avg(ur)",  name="ur")
scalar.add_task("vol_avg(uφ)",  name="uφ")
analysis_tasks['scalar'] = scalar

slices = solver.evaluator.add_file_handler(out_dir+'slices', sim_dt=output_dt, max_writes=40, mode=mode, iter=out_iter)
slices.add_task('s1')
slices.add_task('ur')
slices.add_task('uφ')

checkpoint = solver.evaluator.add_file_handler(out_dir+'checkpoint', sim_dt=100*t_buoy, max_writes=1, mode=mode)
checkpoint.add_system(solver.state, layout='c')




### 8. Setup flow tracking for terminal output, including rolling averages
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("Re_rms", name='Re_rms')
flow.add_property("ρ*T*s1", name='TE')
flow.add_property("0.5*ρ*vel_rms**2", name='KE')
flow.add_property("2*pi*right(r*(cond_flux))", name='right_lum')

Hermitian_cadence = 100
# Main loop
try:
    count = Re_avg = 0
    logger.info('Starting loop')
    init_time = last_time = solver.sim_time
    start_iter = solver.iteration
    start_time = time.time()
    while (solver.ok and np.isfinite(Re_avg)):
        dt = CFL.compute_dt()
        solver.step(dt) #, trim=True)

        if solver.iteration % 10 == 0:
            Re_avg = flow.grid_average('Re_rms')
            log_string =  'Iteration: {:5d}, '.format(solver.iteration)
            log_string += 'Time: {:8.3e}, dt: {:8.3e}, '.format(solver.sim_time,  dt)
            log_string += 'Re: {:8.3e}/{:8.3e}, '.format(Re_avg, flow.max('Re_rms'))
            log_string += 'KE/TE: {:8.3e}/{:8.3e}, '.format(flow.grid_average('KE'), flow.grid_average('TE'))
            log_string += 'r_Lum: {:8.3e}, '.format(flow.grid_average('right_lum'))
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
