"""
Configuration handling. -> inspired by dedalus' config handling
"""
import os
from collections import OrderedDict

from d3_stars.defaults.controls import star, numerics, eigenvalue, dynamics
from d3_stars.defaults.ncc_specs import defaults, nccs
from d3_stars.defaults.outputs import handler_defaults, handlers

if os.path.exists('controls.py'):
    from controls import star, numerics, eigenvalue, dynamics
    from ncc_specs import defaults, nccs
    from outputs import handler_defaults, handlers
