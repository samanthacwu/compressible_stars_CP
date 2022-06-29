from collections import OrderedDict

#[grad_s0]
#[grad_chi_rad]

defaults = OrderedDict()
#Max coefficient expansion
defaults['nr_max'] = (32,32,10)
defaults['vector'] = False
defaults['grid_only'] = False
defaults['get_grad'] = False
defaults['from_grad'] = False
defaults['grad_name'] = None

#special keywords for specific nccs
for k in ['nr_post', 'transition_point', 'width']:
    defaults[k] = None


nccs = OrderedDict()
for field in ['ln_rho', 'grad_s0', 'H', 'chi_rad', 'nu_diff', 'ln_T', 'g', 'pomega_tilde']:
    nccs[field] = OrderedDict()
    for k, val in defaults.items():
        nccs[field][k] = val

    if 'grad_' in field:
        nccs[field]['vector'] = True

nccs['g']['vector'] = True
nccs['g']['nr_max'] = (7,7,7)

nccs['ln_rho']['nr_max'] = (7,7,7)
nccs['ln_rho']['get_grad'] = True
nccs['ln_rho']['grad_name'] = 'grad_ln_rho'

nccs['grad_s0']['nr_max'] = (10,7,7)
nccs['grad_s0']['nr_post'] = (60,7,7)
nccs['grad_s0']['transition_point'] = 1.05
nccs['grad_s0']['width'] = 0.05

nccs['H']['grid_only'] = True
nccs['H']['nr_max'] = (60,1,1)

nccs['chi_rad']['nr_max'] = (1, 15, 15)
nccs['chi_rad']['get_grad'] = True
nccs['chi_rad']['grad_name'] = 'grad_chi_rad'


nccs['ln_T']['nr_max'] = (8,15,5)
nccs['ln_T']['get_grad'] = True
nccs['ln_T']['grad_name'] = 'grad_ln_T'

nccs['nu_diff']['nr_max'] = (1, 1, 1)

new_keys = []
for ncc in nccs.keys():
    if nccs[ncc]['grad_name'] is not None:
        new_keys.append(nccs[ncc]['grad_name'])

for ncc in new_keys:
    nccs[ncc] = OrderedDict()
    for k, val in defaults.items():
        nccs[ncc][k] = val
    nccs[ncc]['vector'] = True
    nccs[ncc]['from_grad'] = True
