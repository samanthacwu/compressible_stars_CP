from numpy import pi
from collections import OrderedDict
from copy import deepcopy

handler_defaults = OrderedDict()
handler_defaults['max_writes'] = 40
handler_defaults['time_unit'] = 'heating' #also avail: kepler
handler_defaults['dt_factor'] = 0.1
handler_defaults['even_outputs'] = False
handler_defaults['tasks'] = list()
handler_defaults['parallel'] = None

handlers = OrderedDict()
#for hname in ['output_tasks', 'wave_shells', 'checkpoint']:
#    handlers[hname] = OrderedDict()
#    for k, val in handler_defaults.items():
#        handlers[hname][k] = deepcopy(val)
#
#
#### Checkpoints
#handlers['checkpoint']['max_writes'] = 1
#handlers['checkpoint']['dt_factor'] = 10
#
#### Dynamical Slices
##Equatorial slices
#eq_tasks = OrderedDict()
#eq_tasks['type'] = 'equator'
#eq_tasks['fields'] = ['s1', 'u']
#handlers['output_tasks']['tasks'].append(eq_tasks)
#
##Meridional slices
#mer_tasks = OrderedDict()
#mer_tasks['type'] = 'meridian'
#mer_tasks['fields'] = ['s1', 'u']
#mer_tasks['interps'] = [0, 0.5*pi, pi, 1.5*pi]
#handlers['output_tasks']['tasks'].append(mer_tasks)
#
##Shell slices
#shell_tasks = OrderedDict()
#shell_tasks['type'] = 'shell'
#shell_tasks['fields'] = ['s1', 'u']
#shell_tasks['interps'] = [0.5, 1, '0.75R', '0.95R', 'R']
#handlers['output_tasks']['tasks'].append(shell_tasks)
#
#
### Scalars
#energy_tasks = OrderedDict()
#energy_tasks['type'] = 'full_integ'
#energy_tasks['fields'] = ['KE', 'PE1', 'IE1', 'FlucE', 'L_squared', 'rho_fluc']
#handlers['output_tasks']['tasks'].append(energy_tasks)
#scalar_tasks = OrderedDict()
#scalar_tasks['type'] = 'vol_avg'
#scalar_tasks['fields'] = ['u_squared', 'Re', 'KE']
#handlers['output_tasks']['tasks'].append(scalar_tasks)
#
### Profiles
#prof_tasks = OrderedDict()
#prof_tasks['type'] = 's2_avg'
#prof_tasks['fields'] = ['s1', 'KE_lum_r', 'enth_lum_r', 'visc_lum_r', 'cond_lum_r', 'PE_lum_r', 'N2']
#handlers['output_tasks']['tasks'].append(prof_tasks)
#
#   
#### Hi-cadence shells
#handlers['wave_shells']['max_writes'] = 500
#handlers['wave_shells']['time_unit'] = 'kepler'
#handlers['wave_shells']['dt_factor'] = 1
#handlers['wave_shells']['even_outputs'] = True
#handlers['wave_shells']['parallel'] = 'virtual'
#
#wave_shell_tasks = OrderedDict()
#wave_shell_tasks['type'] = 'shell'
#wave_shell_tasks['fields'] = ['u', 's1', 'enthalpy_fluc']
#wave_shell_tasks['interps'] = [1.1, 1.5, 1.75, 2.0, '0.95R']
#handlers['wave_shells']['tasks'].append(wave_shell_tasks)
#
