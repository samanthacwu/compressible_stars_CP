from collections import OrderedDict

defaults = OrderedDict()
#Max coefficient expansion
defaults['nr_max'] = (32, 32, 10)
defaults['vector'] = False
defaults['grid_only'] = False

#special keywords for specific nccs
for k in ['nr_post', 'transition_point', 'width']:
    defaults[k] = None


nccs = OrderedDict()
for field in ['ln_rho', 'grad_ln_rho', 'ln_T', 'grad_ln_T', 'T', 'grad_T',\
              'grad_s0', 'H', 'chi_rad', 'grad_chi_rad', 'nu_diff', 'g_over_cp']:
    nccs[field] = OrderedDict()
    for k, val in defaults.items():
        nccs[field][k] = val

    if 'grad_' in field:
        nccs[field]['vector'] = True

#[grad_s0]
#[grad_chi_rad]

nccs['ln_T']['nr_max'] = (16, 32, 10)
nccs['grad_ln_T']['nr_max'] = (17, 33, 11)

nccs['grad_s0']['nr_max'] = (10, 32, 10)
nccs['grad_s0']['nr_post'] = (60, 32, 10)
nccs['grad_s0']['transition_point'] = 1.05
nccs['grad_s0']['width'] = 0.05

nccs['H']['grid_only'] = True
nccs['H']['nr_max'] = (60, 2, 2)

nccs['chi_rad']['nr_max'] = (1, 31, 10)
nccs['grad_chi_rad']['nr_post'] = (1, 32, 11)

nccs['nu_diff']['nr_max'] = (1, 1, 1)

nccs['g_over_cp']['vector'] = True
