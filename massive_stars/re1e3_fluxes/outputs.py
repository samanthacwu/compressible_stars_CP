from numpy import pi
from collections import OrderedDict
from copy import deepcopy

handler_defaults = OrderedDict()
handler_defaults['max_writes'] = 40
handler_defaults['time_unit'] = 'heating' #also avail: kepler
handler_defaults['dt_factor'] = 0.05
handler_defaults['even_outputs'] = False
handler_defaults['tasks'] = list()

handlers = OrderedDict()
for hname in ['slices', 'shells', 'scalars', 'profiles', 'checkpoint']:
    handlers[hname] = OrderedDict()
    for k, val in handler_defaults.items():
        handlers[hname][k] = deepcopy(val)

## Dynamical Slices
#Equatorial slices
eq_tasks = OrderedDict()
eq_tasks['type'] = 'equator'
eq_tasks['fields'] = ['s1', 'u']
handlers['slices']['tasks'].append(eq_tasks)

#Meridional slices
mer_tasks = OrderedDict()
mer_tasks['type'] = 'meridian'
mer_tasks['fields'] = ['s1', 'u']
mer_tasks['interps'] = [0, 0.5*pi, pi, 1.5*pi]
handlers['slices']['tasks'].append(mer_tasks)

#Shell slices
shell_tasks = OrderedDict()
shell_tasks['type'] = 'shell'
shell_tasks['fields'] = ['s1', 'u']
shell_tasks['interps'] = [0.5, 1, '0.75R', '0.95R', 'R']
handlers['slices']['tasks'].append(shell_tasks)

## Scalars
handlers['scalars']['max_writes'] = 400

scalar_tasks = OrderedDict()
scalar_tasks['type'] = 'vol_avg'
scalar_tasks['fields'] = ['u_squared', 'Re', 'KE', 'TE', 'TotE', 'Lx', 'Ly', 'Lz', 'L_squared']
handlers['scalars']['tasks'].append(scalar_tasks)

## Profiles
handlers['profiles']['max_writes'] = 100
handlers['profiles']['dt_factor'] = 0.1

prof_tasks = OrderedDict()
prof_tasks['type'] = 's2_avg'
prof_tasks['fields'] = ['s1', 'KE_lum_r', 'enth_lum_r', 'visc_lum_r', 'cond_lum_r']
handlers['profiles']['tasks'].append(prof_tasks)

### Checkpoints
handlers['checkpoint']['max_writes'] = 1
handlers['checkpoint']['dt_factor'] = 10
   
