from collections import OrderedDict

#### Default star info / stratification choices
star = OrderedDict()

#path to MESA profile to load (searchings in $PATH and  d3_stars/stock_models
star['path'] = 'zams_15Msol/LOGS/profile47.data'

#basis boundaries
#Only works with L if you have core convection zone; how to generalize that?
star['r_bounds'] = (0, '1.10L', '0.82R', '0.93R')

#radial resolution(s) -> length = len(r_bounds) - 1
star['nr'] = (96,96,64)

#options for building the star
star['smooth_h'] = True


### Numerical choices
numerics = OrderedDict()

numerics['equations'] = 'FC_HD' #fully compressible hydro

#Need to add a tag saying what the default diffusivity formalism is
#Target reynolds number of simulation; higher needs more resolution
numerics['reynolds_target'] = 1e3

numerics['prandtl'] = 1

# NCC cutoff needs to be small enough to resolve star non-constant coefficients but large enough that machine precision isn't a problem    
numerics['ncc_cutoff'] = 1e-10
numerics['N_dealias'] = 1.5
numerics['L_dealias'] = 1.5

eigenvalue = OrderedDict()
    
#Factor by which to increase the radial resolution for the hi-res EVP solve
eigenvalue['hires_factor'] = 1.5
    
#highest spherical harmonic degree to solve EVP at
eigenvalue['Lmax'] = 1

dynamics = OrderedDict()

dynamics['ntheta'] = 96
dynamics['safety'] = 0.2
dynamics['timestepper'] = 'SBDF2'

#In nondimensional units
dynamics['CFL_max_r'] = 1.05

#Stop conditions
dynamics['wall_hours'] = 47.5
dynamics['buoy_end_time'] = 5e2

#Damping sim if sponge = true; damping term multiplied by tau_factor
dynamics['sponge'] = False
dynamics['tau_factor'] = 1

#Initial noise amplitude.
dynamics['A0'] = 1e-8

