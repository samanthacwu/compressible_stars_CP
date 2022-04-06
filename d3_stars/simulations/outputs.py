from operator import itemgetter
from collections import OrderedDict
import numpy as np
import dedalus.public as d3
from dedalus.core.future import FutureField
import h5py

from pathlib import Path
from configparser import ConfigParser

from .parser import parse_std_config, parse_ncc_config

output_tasks = {}
output_tasks['u'] = 'u_{0}'
output_tasks['ur'] = 'dot(er_{0}, u_{0})'
output_tasks['u_squared'] = 'dot(u_{0}, u_{0})'
output_tasks['KE'] = '0.5*rho_{0}*' + output_tasks['u_squared']
output_tasks['TE'] = 'rho_{0}*T_{0}*s1_{0}'
output_tasks['Re'] = '('+output_tasks['u_squared']+')**(1/2) / nu_diff_{0}'
output_tasks['p'] = 'p_{0}'
output_tasks['s1'] = 's1_{0}'
output_tasks['grad_s1'] = 'grad_s1_{0}'
output_tasks['L'] = 'cross(r_vec_{0}, rho_{0}*u_{0})' #angular momentum
output_tasks['Lx'] = 'dot(ex_{0},' + output_tasks['L'] + ')'
output_tasks['Ly'] = 'dot(ey_{0},' + output_tasks['L'] + ')'
output_tasks['Lz'] = 'dot(ez_{0},' + output_tasks['L'] + ')'
output_tasks['L_squared'] = 'dot(' + output_tasks['L'] + ',' + output_tasks['L'] + ')'
output_tasks['pomega_hat'] = 'p_{0} - 0.5*dot(u_{0}, u_{0})'
output_tasks['enthalpy'] = output_tasks['pomega_hat'] + ' + T_{0}*s1_{0}'

def initialize_outputs(solver, coords, namespace, bases, timescales, out_dir='./'):
    t_kepler, t_heat, t_rot = timescales
    locals().update(namespace)
    dist = solver.dist
    ## Analysis Setup
    # Cadence
    scalar_dt = 0.25*t_heat
    lum_dt   = 0.5*t_heat
    visual_dt = 0.05*t_heat
    outer_shell_dt = t_kepler
    checkpoint_time = 10*t_heat

    config, raw_config, star_dir, star_file = parse_std_config('controls.cfg')
    with h5py.File(star_file, 'r') as f:
        r_outer = f['r_outer'][()]
    sponge = config['sponge']

    analysis_tasks = OrderedDict()
    even_analysis_tasks = OrderedDict()
    even_analysis_tasks['output_dts'] = []
    config_file = Path('outputs.cfg')
    config = ConfigParser()
    config.read(str(config_file))

    for k in config.keys():
        print(k)
        if 'handler-' in k or k == 'checkpoint':
            h_name = config[k]['name']
            max_writes = int(config[k]['max_writes'])
            time_unit = config[k]['time_unit']
            if time_unit == 'heating':
                t_unit = t_heat
            elif time_unit == 'kepler':
                t_unit = t_kepler
            else:
                print('t unit not found; using t_unit = 1')
                t_unit = 1
            sim_dt = float(config[k]['dt_factor'])*t_unit
            if k == 'checkpoint':
                analysis_tasks[h_name] = solver.evaluator.add_file_handler('{:s}/{:s}'.format(out_dir, h_name), sim_dt=sim_dt, max_writes=max_writes)
                analysis_tasks[h_name].add_tasks(solver.state, layout='g')
            else:
                if config.getboolean(k, 'even_outputs'):
                    even_analysis_tasks['output_dts'].append(sim_dt)
                    even_analysis_tasks[h_name] = solver.evaluator.add_file_handler('{:s}/{:s}'.format(out_dir, h_name), sim_dt=np.inf, max_writes=max_writes)
                else:
                    analysis_tasks[h_name] = solver.evaluator.add_file_handler('{:s}/{:s}'.format(out_dir, h_name), sim_dt=sim_dt, max_writes=max_writes)
    for k in config.keys():
        if 'tasks-' in k:
            if config[k]['handler'] in even_analysis_tasks.keys():
                handler = even_analysis_tasks[config[k]['handler']]
            else:
                handler = analysis_tasks[config[k]['handler']]

            for bn, basis in bases.items():
                solver.problem.namespace['r_vec_{}'.format(bn)] = r_vec = dist.VectorField(coords, name='r_vec_{}'.format(bn), bases=basis)
                r_vec['g'][2] = namespace['r1_{}'.format(bn)]
                if config[k]['type'] == 'equator':
                    items = [item for item in config[k].keys() if 'field' in item ]
                    for item in items:
                        fieldname = config[k][item]
                        fieldstr = output_tasks[fieldname].format(bn)
                        task = eval('({})(theta=np.pi/2)'.format(fieldstr), dict(solver.problem.namespace))
                        handler.add_task(task, name='equator({}_{})'.format(fieldname, bn))
                elif config[k]['type'] == 'meridional':
                    items = [item for item in config[k].keys() if 'field' in item ]
                    interps = [item for item in config[k].keys() if 'interp' in item ]
                    for item in items:
                        fieldname = config[k][item]
                        fieldstr = output_tasks[fieldname].format(bn)
                        for interp in interps:
                            base_interp = config[k][interp]
                            interp = base_interp.replace('pi', 'np.pi')
                            task = eval('({})(phi={})'.format(fieldstr, interp), dict(solver.problem.namespace))
                            handler.add_task(task, name='meridion({}_{},phi={})'.format(fieldname, bn, base_interp))
                elif config[k]['type'] == 'shell':
                    items = [item for item in config[k].keys() if 'field' in item ]
                    interps = [item for item in config[k].keys() if 'interp' in item ]
                    for item in items:
                        fieldname = config[k][item]
                        fieldstr = output_tasks[fieldname].format(bn)
                        for interp in interps:
                            base_interp = config[k][interp]
                            if 'R' in base_interp:
                                if base_interp == 'R':
                                    interp = '1'
                                else:
                                    interp = base_interp.replace('R', '')
                                interp = np.around(float(interp)*r_outer, decimals=2)
                            else:
                                interp = np.around(float(base_interp), decimals=2)
                            if type(basis) == d3.BallBasis and interp > basis.radius:
                                continue
                            elif type(basis) == d3.ShellBasis:
                                if interp <= basis.radii[0] or interp > basis.radii[1] :
                                    continue
                            print('{}(r={})'.format(fieldstr, interp))
                            task = eval('({})(r={})'.format(fieldstr, interp), dict(solver.problem.namespace))
                            handler.add_task(task, name='shell({}_{},r={})'.format(fieldname, bn, base_interp))
                elif config[k]['type'] == 'vol_avg':
                    if type(basis) == d3.BallBasis:
                        volume  = (4/3)*np.pi*basis.radius**3
                    else:
                        Ri, Ro = basis.radii[0], basis.radii[1]
                        volume  = (4/3)*np.pi*(Ro**3 - Ri**3)
                    items = [item for item in config[k].keys() if 'field' in item ]
                    solver.problem.namespace['vol_avg_{}'.format(bn)] = lambda A: d3.Integrate(A/volume, coords)
                    namespace['vol_avg_{}'.format(bn)] = solver.problem.namespace['vol_avg_{}'.format(bn)]

                    for item in items:
                        fieldname = config[k][item]
                        fieldstr = output_tasks[fieldname].format(bn)
                        print('vol_avg_{}({})'.format(bn, fieldstr))
                        task = eval('vol_avg_{}({})'.format(bn, fieldstr), dict(solver.problem.namespace))
                        handler.add_task(task, name='shell({}_{},r={})'.format(fieldname, bn, base_interp))
                elif config[k]['type'] == 's2_avg':
                    #TODO
                    pass
                elif config[k]['type'] == 'luminosity':
                    #TODO
                    pass


                    



    slices = analysis_tasks['slices']
    scalars = analysis_tasks['scalars']
    profiles = analysis_tasks['profiles']
    surface_shell_slices = even_analysis_tasks['shells']
    checkpoint = analysis_tasks['checkpoint']

    az_avg = lambda A: d3.Average(A, coords.coords[0])
    s2_avg = lambda A: d3.Average(A, coords.S2coordsys)

    for bn, basis in bases.items():
        phi, theta, r = itemgetter('phi_'+bn, 'theta_'+bn, 'r_'+bn)(namespace)
        phi1, theta1, r1 = itemgetter('phi1_'+bn, 'theta1_'+bn, 'r1_'+bn)(namespace)
        ex, ey, ez = itemgetter('ex_'+bn, 'ey_'+bn, 'ez_'+bn)(namespace)
        T, rho = itemgetter('T_{}'.format(bn), 'rho_{}'.format(bn))(namespace)
        div_u, E = itemgetter('div_u_RHS_{}'.format(bn), 'E_RHS_{}'.format(bn))(namespace)
        u = namespace['u_{}'.format(bn)]
        p = namespace['p_{}'.format(bn)]
        s1 = namespace['s1_{}'.format(bn)]
        nu_diff = namespace['nu_diff_{}'.format(bn)]
        chi_rad = namespace['chi_rad_{}'.format(bn)]

        namespace['r_vals_{}'.format(bn)] = r_vals = dist.Field(name='r_vals_{}'.format(bn), bases=basis)
        r_vals['g'] = r1
#        r_vec['g'][2] = r1
        r_vals = d3.Grid(r_vals).evaluate()
        er = d3.Grid(namespace['er_{}'.format(bn)]).evaluate()

        u_squared = d3.dot(u, u)
        ur = d3.dot(er, u)
        pomega_hat = p - 0.5*u_squared
        h = pomega_hat + T*s1
        visc_flux = 2*(d3.dot(u, E) - (1/3) * u * div_u)
        visc_flux_r = d3.dot(er, visc_flux)

#        angular_momentum = d3.cross(r_vec, rho*u)
#        am_Lx = d3.dot(ex, angular_momentum)
#        am_Ly = d3.dot(ey, angular_momentum)
#        am_Lz = d3.dot(ez, angular_momentum)

        if type(basis) == d3.BallBasis:
            volume  = (4/3)*np.pi*basis.radius**3
        else:
            index = int(bn.split('S')[-1])-1
            Ri = basis.radii[0]
            Ro = basis.radii[1]
            volume  = (4/3)*np.pi*(Ro**3 - Ri**3)

        lum_prof = namespace['lum_prof_{}'.format(bn)] = lambda A: s2_avg((4*np.pi*r_vals**2) * A)


#        # Add scalars for simple evolution tracking
#        scalars.add_task(vol_avg((u_squared)**(1/2) / nu_diff), name='Re_avg_{}'.format(bn),  layout='g')
#        scalars.add_task(vol_avg(rho*u_squared/2), name='KE_{}'.format(bn),   layout='g')
#        scalars.add_task(vol_avg(rho*T*s1), name='TE_{}'.format(bn),  layout='g')
#        scalars.add_task(vol_avg(am_Lx), name='angular_momentum_x_{}'.format(bn), layout='g')
#        scalars.add_task(vol_avg(am_Ly), name='angular_momentum_y_{}'.format(bn), layout='g')
#        scalars.add_task(vol_avg(am_Lz), name='angular_momentum_z_{}'.format(bn), layout='g')
#        scalars.add_task(vol_avg(d3.dot(angular_momentum, angular_momentum)), name='square_angular_momentum_{}'.format(bn), layout='g')

        # Add profiles to track structure and fluxes
        profiles.add_task(s2_avg(s1), name='s1_profile_{}'.format(bn), layout='g')
        profiles.add_task(lum_prof(rho*ur*pomega_hat),   name='wave_lum_{}'.format(bn), layout='g')
        profiles.add_task(lum_prof(rho*ur*h),            name='enth_lum_{}'.format(bn), layout='g')
        profiles.add_task(lum_prof(-nu_diff*rho*visc_flux_r), name='visc_lum_{}'.format(bn), layout='g')
        profiles.add_task(lum_prof(-rho*T*chi_rad*d3.dot(er, d3.grad(s1))), name='cond_lum_{}'.format(bn), layout='g')
        profiles.add_task(lum_prof(0.5*rho*ur*u_squared), name='KE_lum_{}'.format(bn),   layout='g')

    return analysis_tasks, even_analysis_tasks
