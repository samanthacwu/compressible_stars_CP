"""
Configuration handling. -> inspired by dedalus' config handling
"""
import os

from compstar.defaults.controls import star, numerics, eigenvalue, dynamics
from compstar.defaults.ncc_specs import defaults, nccs
from compstar.defaults.outputs import handler_defaults, handlers

if os.path.exists('controls.py'):
    from controls import star, numerics, eigenvalue, dynamics
    from ncc_specs import defaults, nccs
    from outputs import handler_defaults, handlers
