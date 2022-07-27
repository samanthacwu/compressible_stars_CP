import functools
from operator import itemgetter
from collections import OrderedDict
import numpy as np
import dedalus.public as d3
from dedalus.core.future import FutureField
import h5py

from pathlib import Path
from configparser import ConfigParser

from .parser import name_star
from d3_stars.defaults import config

output_tasks = {}
flux_tags = ['cond', 'KE', 'PE', 'enth', 'visc']
defaults = ['u', 'momentum', 'ur', 'u_squared', 'KE', 'PE', 'IE', 'TotE', 'PE1', 'IE1', 'FlucE', 'Re', 'Ma', 'ln_rho1', \
            'pom1', 'pom_fluc', 'pom_full', 'grad_s1', 'L', 's1']
for k in defaults + ['F_{}'.format(t) for t in flux_tags]:
    output_tasks[k] = '{}'.format(k) + '_{0}'

#angular momentum components
output_tasks['Lx'] = 'dot(ex_{0},L_{0})'
output_tasks['Ly'] = 'dot(ey_{0},L_{0})'
output_tasks['Lz'] = 'dot(ez_{0},L_{0})'
output_tasks['L_squared'] = 'dot(L_{0}, L_{0})'

for t in flux_tags:
    output_tasks['{}_lum'.format(t)] = '(4*np.pi*r_vals_{0}**2) * ( F_' + t + '_{0} )'
    output_tasks['{}_lum_r'.format(t)] = 'dot(er, ' + output_tasks['{}_lum'.format(t)] + ')'

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

    star_dir, out_file = name_star()
    with h5py.File(out_file, 'r') as f:
        r_outer = f['r_outer'][()]

    analysis_tasks = OrderedDict()
    even_analysis_tasks = OrderedDict()
    even_analysis_tasks['output_dts'] = []

    def vol_avg(A, volume):
        return d3.Integrate(A/volume, coords)

    for bn, basis in bases.items():
#        solver.problem.namespace['r_vec_{}'.format(bn)] = r_vec = dist.VectorField(coords, name='r_vec_{}'.format(bn), bases=basis)
#        solver.problem.namespace['r_vals_{}'.format(bn)] = r_vals = dist.Field(name='r_vals_{}'.format(bn), bases=basis)
#        r_vals['g'] = namespace['r1_{}'.format(bn)]
#        r_vec['g'][2] = namespace['r1_{}'.format(bn)]

        if type(basis) == d3.BallBasis:
            vol = namespace['volume_{}'.format(bn)]  = (4/3)*np.pi*basis.radius**3
        else:
            Ri, Ro = basis.radii[0], basis.radii[1]
            vol = namespace['volume_{}'.format(bn)]  = (4/3)*np.pi*(Ro**3 - Ri**3)

        solver.problem.namespace['vol_avg_{}'.format(bn)] = functools.partial(vol_avg, volume=vol)
        namespace['vol_avg_{}'.format(bn)] = solver.problem.namespace['vol_avg_{}'.format(bn)]
    

    for h_name in config.handlers.keys():
        this_dict = config.handlers[h_name]
        max_writes = int(this_dict['max_writes'])
        time_unit = this_dict['time_unit']
        if time_unit == 'heating':
            t_unit = t_heat
        elif time_unit == 'kepler':
            t_unit = t_kepler
        else:
            logger.info('t unit not found; using t_unit = 1')
            t_unit = 1
        sim_dt = float(this_dict['dt_factor'])*t_unit
        if h_name == 'checkpoint':
            analysis_tasks[h_name] = solver.evaluator.add_file_handler('{:s}/{:s}'.format(out_dir, h_name), sim_dt=sim_dt, max_writes=max_writes)
            analysis_tasks[h_name].add_tasks(solver.state, layout='g')
        else:
            if this_dict['even_outputs']:
                even_analysis_tasks['output_dts'].append(sim_dt)
                this_dict['handler'] = even_analysis_tasks[h_name] = solver.evaluator.add_file_handler('{:s}/{:s}'.format(out_dir, h_name), sim_dt=np.inf, max_writes=max_writes)
            else:
                this_dict['handler'] = analysis_tasks[h_name] = solver.evaluator.add_file_handler('{:s}/{:s}'.format(out_dir, h_name), sim_dt=sim_dt, max_writes=max_writes)

        tasks = this_dict['tasks']
        for this_task in tasks:
            handler = this_dict['handler']
            for bn, basis in bases.items():
                if this_task['type'] == 'equator':
                    for fieldname in this_task['fields']:
                        fieldstr = output_tasks[fieldname].format(bn)
                        task = eval('({})(theta=np.pi/2)'.format(fieldstr), dict(solver.problem.namespace))
                        handler.add_task(task, name='equator({}_{})'.format(fieldname, bn))
                elif this_task['type'] == 'meridian':
                    interps = this_task['interps']
                    for fieldname in this_task['fields']:
                        fieldstr = output_tasks[fieldname].format(bn)
                        for base_interp in interps:
                            task = eval('({})(phi={})'.format(fieldstr, base_interp), dict(solver.problem.namespace))
                            handler.add_task(task, name='meridian({}_{},phi={})'.format(fieldname, bn, base_interp))
                elif this_task['type'] == 'shell':
                    interps = this_task['interps']
                    for fieldname in this_task['fields']:
                        fieldstr = output_tasks[fieldname].format(bn)
                        for base_interp in interps:
                            if isinstance(base_interp, str) and 'R' in base_interp:
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
                elif this_task['type'] == 'vol_avg':
                    for fieldname in this_task['fields']:
                        fieldstr = output_tasks[fieldname].format(bn)
                        task = eval('vol_avg_{}({})'.format(bn, fieldstr), dict(solver.problem.namespace))
                        handler.add_task(task, name='vol_avg({}_{})'.format(fieldname, bn))
                elif this_task['type'] == 's2_avg':
                    for fieldname in this_task['fields']:
                        fieldstr = output_tasks[fieldname].format(bn)
                        task_str = 's2_avg({})'.format(fieldstr)
                        task = eval(task_str, dict(solver.problem.namespace))
                        handler.add_task(task, name='s2_avg({}_{})'.format(fieldname, bn))

    return analysis_tasks, even_analysis_tasks
