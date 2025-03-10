from collections import OrderedDict

defaults = OrderedDict()
#Max coefficient expansion
defaults['nr_max'] = (60,42,36) #(60,42,36)
defaults['vector'] = False
defaults['grid_only'] = False
defaults['get_grad'] = False
defaults['get_inverse'] = False
defaults['from_grad'] = False
defaults['grad_name'] = None

#special keywords for specific nccs
for k in ['nr_post', 'transition_point', 'width']:
    defaults[k] = None


nccs = OrderedDict()
for field in ['ln_rho0', 'Q', 'chi_rad', 'kappa_rad', 'nu_diff', 'g_phi',  'grad_s0', 'grad_ln_rho0','pom0', 'grad_ln_pom0', 's0']:
    nccs[field] = OrderedDict()
    for k, val in defaults.items():
        nccs[field][k] = val

    if 'grad_' in field:
        nccs[field]['vector'] = True

# nccs['ln_rho0']['get_grad'] = True
# nccs['ln_rho0']['grad_name'] = 'grad_ln_rho0'

nccs['Q']['grid_only'] = True
nccs['Q']['nr_max'] = (60,1,1)

nccs['chi_rad']['nr_max'] = (1,32,20)
nccs['chi_rad']['get_grad'] = True
nccs['chi_rad']['grad_name'] = 'grad_chi_rad'

nccs['kappa_rad']['nr_max'] = (32,32,5)
nccs['kappa_rad']['get_grad'] = True
nccs['kappa_rad']['grad_name'] = 'grad_kappa_rad'


nccs['nu_diff']['nr_max'] = (1,1,1)
nccs['nu_diff']['get_grad'] = True
nccs['nu_diff']['grad_name'] = 'grad_nu_diff'

nccs['g_phi']['get_grad'] = True
nccs['g_phi']['grad_name'] = 'neg_g'

nccs['grad_s0']['vector'] = True
nccs['grad_s0']['nr_max'] = (90,42,36)

nccs['pom0']['get_grad'] = True
nccs['pom0']['grad_name'] = 'grad_pom0'
nccs['pom0']['get_inverse'] = True

nccs['grad_ln_pom0']['vector'] = True

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


new_keys = []
for ncc in nccs.keys():
    if nccs[ncc]['get_inverse'] is not False:
        new_keys.append('inv_{}'.format(ncc))
for ncc in new_keys:
    nccs[ncc] = OrderedDict()
    for k, val in defaults.items():
        nccs[ncc][k] = val


initial_flucts = OrderedDict()
for field in ['ln_rho1', 's1']:
    initial_flucts[field] = OrderedDict()
    for k, val in defaults.items():
        initial_flucts[field][k] = val
    if 'grad_' in field:
        initial_flucts[field]['vector'] = True
#can edit nr_max and stuff if needed
