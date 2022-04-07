import functools
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
output_tasks['pomega_hat'] = 'p_{0} - 0.5*dot(u_{0}, u_{0})'
output_tasks['enthalpy'] = output_tasks['pomega_hat'] + ' + T_{0}*s1_{0}'
output_tasks['L'] = 'cross(r_vec_{0}, rho_{0}*u_{0})' #angular momentum
output_tasks['Lx'] = 'dot(ex_{0},' + output_tasks['L'] + ')'
output_tasks['Ly'] = 'dot(ey_{0},' + output_tasks['L'] + ')'
output_tasks['Lz'] = 'dot(ez_{0},' + output_tasks['L'] + ')'
output_tasks['L_squared'] = 'dot(' + output_tasks['L'] + ',' + output_tasks['L'] + ')'

output_tasks['visc_flux'] = '-2*rho_{0}*nu_diff_{0}*(dot(u_{0}, E_RHS_{0}) - (1/3) * u_{0} * div_u_{0})'
output_tasks['wave_flux'] = 'rho_{0}*u_{0}*(' + output_tasks['pomega_hat'] + ')'
output_tasks['enth_flux'] = 'rho_{0}*u_{0}*(' + output_tasks['enthalpy'] + ')'
output_tasks['cond_flux'] = '-rho_{0}*T_{0}*chi_rad_{0}*grad(s1_{0})'
output_tasks['KE_flux']   = '0.5*rho_{0}*u_{0}*' + output_tasks['u_squared']

for flux in ['visc_flux', 'wave_flux', 'enth_flux', 'cond_flux', 'KE_flux']:
    output_tasks['{}_r'.format(flux)] = 'dot(er_{0}, ' + output_tasks[flux] + ')'

def initialize_outputs(solver, coords, namespace, bases, timescales, out_dir='./'):
    t_kepler, t_heat, t_rot = timescales
    locals().update(namespace)
    dist = solver.dist
    ## Analysis Setup
    # Cadence
    az_avg = lambda A: d3.Average(A, coords.coords[0])
    s2_avg = lambda A: d3.Average(A, coords.S2coordsys)

    
    solver.problem.namespace['az_avg'] = az_avg
    solver.problem.namespace['s2_avg'] = s2_avg
    namespace['az_avg'] = solver.problem.namespace['az_avg']
    namespace['s2_avg'] = solver.problem.namespace['s2_avg']

    config, raw_config, star_dir, star_file = parse_std_config('controls.cfg')
    with h5py.File(star_file, 'r') as f:
        r_outer = f['r_outer'][()]

    analysis_tasks = OrderedDict()
    even_analysis_tasks = OrderedDict()
    even_analysis_tasks['output_dts'] = []
    config_file = Path('outputs.cfg')
    config = ConfigParser()
    config.read(str(config_file))

    def vol_avg(A, volume):
        return d3.Integrate(A/volume, coords)

    def luminosity(A, rvals):
        return s2_avg(4*np.pi*r_vals**2*A)

    for bn, basis in bases.items():
        solver.problem.namespace['r_vec_{}'.format(bn)] = r_vec = dist.VectorField(coords, name='r_vec_{}'.format(bn), bases=basis)
        solver.problem.namespace['r_vals_{}'.format(bn)] = r_vals = dist.Field(name='r_vals_{}'.format(bn), bases=basis)
        r_vals['g'] = namespace['r1_{}'.format(bn)]
        r_vec['g'][2] = namespace['r1_{}'.format(bn)]

        if type(basis) == d3.BallBasis:
            vol = namespace['volume_{}'.format(bn)]  = (4/3)*np.pi*basis.radius**3
        else:
            Ri, Ro = basis.radii[0], basis.radii[1]
            vol = namespace['volume_{}'.format(bn)]  = (4/3)*np.pi*(Ro**3 - Ri**3)

        solver.problem.namespace['vol_avg_{}'.format(bn)] = functools.partial(vol_avg, volume=vol)
        solver.problem.namespace['luminosity_{}'.format(bn)] = functools.partial(luminosity, rvals=r_vals)
        namespace['vol_avg_{}'.format(bn)] = solver.problem.namespace['vol_avg_{}'.format(bn)]
        namespace['luminosity_{}'.format(bn)] = solver.problem.namespace['luminosity_{}'.format(bn)]
    

    for k in config.keys():
        if 'handler-' in k or k == 'checkpoint':
            h_name = config[k]['name']
            max_writes = int(config[k]['max_writes'])
            time_unit = config[k]['time_unit']
            if time_unit == 'heating':
                t_unit = t_heat
            elif time_unit == 'kepler':
                t_unit = t_kepler
            else:
                logger.info('t unit not found; using t_unit = 1')
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

                if config[k]['type'] == 'equator':
                    items = [item for item in config[k].keys() if 'field' in item ]
                    for item in items:
                        fieldname = config[k][item]
                        fieldstr = output_tasks[fieldname].format(bn)
                        task = eval('({})(theta=np.pi/2)'.format(fieldstr), dict(solver.problem.namespace))
                        handler.add_task(task, name='equator({}_{})'.format(fieldname, bn))
                elif config[k]['type'] == 'meridian':
                    items = [item for item in config[k].keys() if 'field' in item ]
                    interps = [item for item in config[k].keys() if 'interp' in item ]
                    for item in items:
                        fieldname = config[k][item]
                        fieldstr = output_tasks[fieldname].format(bn)
                        for interp in interps:
                            base_interp = config[k][interp]
                            interp = base_interp.replace('pi', 'np.pi')
                            task = eval('({})(phi={})'.format(fieldstr, interp), dict(solver.problem.namespace))
                            handler.add_task(task, name='meridian({}_{},phi={})'.format(fieldname, bn, base_interp))
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
                            task = eval('({})(r={})'.format(fieldstr, interp), dict(solver.problem.namespace))
                            handler.add_task(task, name='shell({}_{},r={})'.format(fieldname, bn, base_interp))
                elif config[k]['type'] == 'vol_avg':
                    items = [item for item in config[k].keys() if 'field' in item ]

                    for item in items:
                        fieldname = config[k][item]
                        fieldstr = output_tasks[fieldname].format(bn)
                        task = eval('vol_avg_{}({})'.format(bn, fieldstr), dict(solver.problem.namespace))
                        handler.add_task(task, name='vol_avg({}_{})'.format(fieldname, bn))
                elif config[k]['type'] == 's2_avg':
                    items = [item for item in config[k].keys() if 'field' in item ]

                    for item in items:
                        fieldname = config[k][item]
                        fieldstr = output_tasks[fieldname].format(bn)
                        task_str = 's2_avg({})'.format(fieldstr)
                        task = eval(task_str, dict(solver.problem.namespace))
                        handler.add_task(task, name='s2_avg({}_{})'.format(fieldname, bn))

                elif config[k]['type'] == 'luminosity':
                    items = [item for item in config[k].keys() if 'field' in item ]

                    for item in items:
                        fieldname = config[k][item]
                        fieldstr = output_tasks[fieldname].format(bn)
                        task_str = 'luminosity_{}({})'.format(bn, fieldstr)
                        task = eval(task_str, dict(solver.problem.namespace))
                        handler.add_task(task, name='luminosity({}_{})'.format(fieldname, bn))

    return analysis_tasks, even_analysis_tasks
