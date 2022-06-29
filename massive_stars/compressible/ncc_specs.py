from collections import OrderedDict

defaults = OrderedDict()
#Max coefficient expansion
defaults['nr_max'] = (32,)
defaults['vector'] = False
defaults['grid_only'] = False
defaults['get_grad'] = False
defaults['from_grad'] = False
defaults['grad_name'] = None

#special keywords for specific nccs
for k in ['nr_post', 'transition_point', 'width']:
    defaults[k] = None


nccs = OrderedDict()
for field in ['ln_rho', 'H', 'chi_rad', 'nu_diff', 'T', 'g_phi']:
    nccs[field] = OrderedDict()
    for k, val in defaults.items():
        nccs[field][k] = val

    if 'grad_' in field:
        nccs[field]['vector'] = True

nccs['g_phi']['nr_max'] = (8,)
nccs['g_phi']['get_grad'] = True
nccs['g_phi']['grad_name'] = 'neg_g'

nccs['ln_rho']['nr_max'] = (16,)
nccs['ln_rho']['get_grad'] = True
nccs['ln_rho']['grad_name'] = 'grad_ln_rho'

nccs['H']['grid_only'] = True
nccs['H']['nr_max'] = (60,)

nccs['chi_rad']['nr_max'] = (1,)
nccs['chi_rad']['get_grad'] = True
nccs['chi_rad']['grad_name'] = 'grad_chi_rad'

nccs['nu_diff']['nr_max'] = (1,)

nccs['T']['nr_max'] = (16,)
nccs['T']['get_grad'] = True
nccs['T']['grad_name'] = 'grad_T'

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
