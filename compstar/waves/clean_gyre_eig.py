"""
This file reads in gyre eigenfunctions, calculates the velocity and velocity dual basis, and outputs in a clean format so that it's ready to be fed into the transfer function calculation.
"""
from collections import OrderedDict
import re
import os

import h5py
import numpy as np
import pygyre as pg
import pymsg as pm
import mesa_reader as mr
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.interpolate import interp1d
import scipy.special as ss

import astropy.units as u
from astropy import constants

from compstar.dedalus.star_builder import find_core_cz_radius
from .transfer import calculate_refined_transfer
from .general import  calculate_optical_depths

Rsun_to_cm = 6.957e10
sigma_SB = 5.67037442e-05 #cgs
G = 6.67E-8
solar_z = 0.014


def natural_sort(iterable, reverse=False):
    """
    Sort alphanumeric strings naturally, i.e. with "1" before "10".
    Copied from dedalus; Based on http://stackoverflow.com/a/4836734.
    """
    convert = lambda sub: int(sub) if sub.isdigit() else sub.lower()
    key = lambda item: [convert(sub) for sub in re.split('([0-9]+)', str(item))]

    return sorted(iterable, key=key, reverse=reverse)


class GyreMSGPostProcessor:
    """
    This class reads in a MESA model and a set of GYRE eigenvalues and eigenfunctions, and calculates the 
    photometric magnitude fluctuation eigenfunction at the stellar surface.
    It also has logic for calculating the transfer function which maps the sqrt of the wave luminosity
    to the photometric variability at the surface of a star.
    """

    def __init__(self, ell, pos_summary, pos_details, mesa_pulse_file, mesa_LOG_file, 
                 initial_z=0.006, specgrid=None, filters=['Red',], 
                 MSG_DIR = os.environ['MSG_DIR'], GRID_DIR=None,
                 PASS_DIR=os.path.join('..','gyre_phot','passbands'),
                 output_dir='gyre_output'):
        """
        Initialize the class.

        Parameters
        ----------
        ell : int
            The spherical harmonic degree of the eigenvalues
        pos_summary : str
            The path to the GYRE summary file
        pos_details : list of str
            The paths to the GYRE details files
        mesa_pulse_file : str
            The path to the MESA pulse (.GYRE) file
        mesa_LOG_file : str
            The path to the MESA profile file
        initial_z : float
            The initial metallicity of the star (default: 0.006, which is the LMC value)
        specgrid : str
            The name of the spectral grid to use (options: 'OSTAR2002' or None, which uses the demo grid)
            See http://user.astro.wisc.edu/~townsend/static.php?ref=msg-grids
        filters : list of str
            The names of the filters to use; uses TESS passband by default which only has 'Red' filter.
            See http://user.astro.wisc.edu/~townsend/static.php?ref=msg-passbands
        MSG_DIR : str
            The path to the MSG directory
        GRID_DIR : str
            The path to the directory where the specgrid file is located; if None, uses $MSG_DIR/data/grids
        PASS_DIR : str
            The path to the directory where the passband files are located; Default is ../gyre_phot/passbands
            TODO: fix this so that it's not hard-coded
        output_dir : str
            The path to the directory where the output files will be written
        """
        self.ell = ell
        self.pos_summary = pos_summary
        self.pos_details = pos_details
        self.output_dir=output_dir
        self.mesa_pulse_file = mesa_pulse_file
        self.mesa_LOG_file = mesa_LOG_file
        self.filters = filters
        self.core_cz_radius = find_core_cz_radius(self.mesa_LOG_file)

        # Load the MSG photometric grids (this code taken from the 
        # Python walkthrough in the MSG docs)
        if GRID_DIR is None:
            os.path.join(MSG_DIR, 'data', 'grids'),
        
        if specgrid == 'OSTAR2002':
            specgrid_file_name = os.path.join(GRID_DIR, 'sg-OSTAR2002-high.h5')
        else:
            specgrid_file_name = os.path.join(GRID_DIR, 'sg-demo-ext.h5')

        # Create the photgrid using the TESS passband.
        self.photgrids = {}
        for filter in self.filters:
            passband_file_name = os.path.join(PASS_DIR, f'pb-TESS-TESS.{filter}-Vega.h5')
            print(specgrid_file_name, passband_file_name)
            self.photgrids[filter] = pm.PhotGrid(specgrid_file_name, passband_file_name)

            # Inspect grid parameters
            print('Grid parameters:')
            for label in self.photgrids[filter].axis_labels:
                print(f'  {label} [{self.photgrids[filter].axis_x_min[label]} -> {self.photgrids[filter].axis_x_max[label]}]')

        # Load the stellar model to figure out fundamental stellar properties & Teff and gravity
        self.model = pg.read_model(mesa_pulse_file)
        self.M = self.model.meta['M_star']
        self.R = self.model.meta['R_star']
        self.L = self.model.meta['L_star']
        self.Z = initial_z

        self.ZdZsol = self.Z/solar_z
        self.Teff = (self.L/(4*np.pi*self.R**2*sigma_SB))**0.25
        self.logg = np.log10(G*self.M/self.R**2)

        # Set up the atmosphere parameters dict (to be passed to MSG)
        self.model_x = {'Teff': self.Teff, 'log(g)': self.logg, 'Z/Zo': self.ZdZsol}
        print(f'Teff: {self.Teff}')
        print(f'log(g): {self.logg}')
        print(f'Z/Zo: {self.ZdZsol}')

        # Evaluate the intensity moments (eqn. 15 of Townsend 2003) and their partials
        self.I_0 = {}
        self.I_l = {}
        self.dI_l_dlnTeff = {}
        self.dI_l_dlng = {}
        with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(self.output_dir, self.ell), 'w') as f:
            for filter in self.filters:
                self.I_0[filter] = self.photgrids[filter].D_moment(self.model_x, 0)
                self.I_l[filter] = self.photgrids[filter].D_moment(self.model_x, self.ell)
                print("I_0, I_l: {:.2e}, {:.2e}".format(self.I_0[filter], self.I_l[filter]))

                self.dI_l_dlnTeff[filter] = self.photgrids[filter].D_moment(self.model_x, self.ell, deriv={'Teff': True})*self.Teff
                self.dI_l_dlng[filter] = self.photgrids[filter].D_moment(self.model_x, self.ell, deriv={'log(g)': True})/np.log(10)
                print("dI_l_dlnTeff, dI_l_dlng: {:.2e}, {:.2e}".format(self.dI_l_dlnTeff[filter], self.dI_l_dlng[filter]))

                f['I_0_{}'.format(filter)] = self.I_0[filter]
                f['I_l_{}'.format(filter)] = self.I_l[filter]
                f['dI_l_dlnTeff_{}'.format(filter)] = self.dI_l_dlnTeff[filter]
                f['dI_l_dlng_{}'.format(filter)] = self.dI_l_dlng[filter]


    def sort_eigenfunctions(self):
        """
        Gather and consistently normalize the GYRE eigenfunctions.
        """
        #TODO: move these background info reading lines up to __init__()
        #get info about mesa background
        self.data_dict = OrderedDict()
        p = mr.MesaData(self.mesa_LOG_file)
        r_mesa = p.radius[::-1]*Rsun_to_cm #in cm
        bruntN2_mesa = p.brunt_N2[::-1] #rad^2/s^2
        T       = p.temperature[::-1] #K
        rho     = 10**(p.logRho[::-1]) #g/cm^3
        opacity = p.opacity[::-1] #cm^2 / g
        cp      = p.cp[::-1] #erg / K / g
        chi_rad_mesa = 16 * sigma_SB * T**3 / (3 * rho**2 * cp * opacity)

        #Get pulsation & stratification information    
        pgout = pg.read_output(self.pos_details[0])
        self.rho = pgout['rho']
        self.x = pgout['x']
        self.r = self.x*self.R #cm

        V = pgout['V_2']*self.x**2
        As = pgout['As']
        c_1 = pgout['c_1']
        Gamma_1 = pgout['Gamma_1']

        bruntN2 = As/c_1
        lambS1 = np.sqrt(1*(1+1))*np.sqrt(Gamma_1/(V*c_1))
        chi_rad = 10**(interp1d(r_mesa, np.log10(chi_rad_mesa), bounds_error=False, fill_value='extrapolate')(self.r))

        #re-dimensionalize -- double-check this and compare omega, freq.
        mid_r = self.r[len(self.r)//2]
        self.gyre_tau_nd = 1/np.sqrt(10**(interp1d(r_mesa, np.log10(bruntN2_mesa))(mid_r)) / bruntN2[len(self.r)//2])
        bruntN2 /= self.gyre_tau_nd**2
        lambS1 /= self.gyre_tau_nd

        #Create storage for loading eigenfunctions into
        #data_dicts already has 'freq', 'omega', 'xi_r_ref', 'lag_L_ref', 'l', 'n_pg'
        data = self.data_dict
        for field in ['xi_r_eigfunc', 'xi_h_eigfunc', 'lag_L_eigfunc', 'u_r_eigfunc', 'u_h_eigfunc']:
            data[field] = np.zeros((len(self.pos_details), len(self.x)), dtype=np.complex128) 
        for field in ['freq', 'omega', 'lag_L_ref', 'xi_r_ref']:
            data[field] = np.zeros((len(self.pos_details)), dtype=np.complex128) 
        for field in ['depth', 'n_pg', 'l']:
            data[field] = np.zeros(len(self.pos_details))
        
        #Loop through GYRE detail files, then normalize and load eigenfunctions.
        for i,filename in enumerate(self.pos_details):
            print('reading eigenfunctions from {}'.format(filename))
            pgout = pg.read_output(filename)

            shift = 1
            #scalars
            data['n_pg'][i] = pgout.meta['n_pg']
            data['l'][i]    = pgout.meta['l']
            data['freq'][i] = 1e-6*pgout.meta['freq'] #cgs
            data['omega'][i] = pgout.meta['omega'] #gyre nondimensionalization
            data['lag_L_ref'][i] = shift*np.sqrt(4*np.pi)*pgout.meta['lag_L_ref'] #L/Lstar
            data['xi_r_ref'][i]  = shift*np.sqrt(4*np.pi)*pgout.meta['xi_r_ref']  #r/Rstar
            data['depth'][i] = calculate_optical_depths(np.array([data['freq'][i],]), self.r, bruntN2, lambS1, chi_rad, ell=self.ell)[0]

            #eigenfunctions
            data['xi_r_eigfunc'][i,:]  = shift*np.sqrt(4*np.pi)*self.R*pgout['xi_r'] #arbitrary amplitude; cgs units.
            data['xi_h_eigfunc'][i,:]  = shift*np.sqrt(4*np.pi)*self.R*pgout['xi_h'] #arbitrary amplitude; cgs units.
            data['lag_L_eigfunc'][i,:] = shift*np.sqrt(4*np.pi)*self.L*pgout['lag_L'] #arbitrary amplitude; cgs units
            
        #u = dt(xi) = -i om u by defn.
        #These velocity eigenfunctions are cgs-dimensional but of arbitrary amplitude.
        data['u_r_eigfunc'] = -1j*2*np.pi*data['freq'][:,None]*data['xi_r_eigfunc']
        data['u_h_eigfunc'] = -1j*2*np.pi*data['freq'][:,None]*data['xi_h_eigfunc'] * (np.sqrt(self.ell*(self.ell+1)))
        data['delta_L_dL_top'] = data['lag_L_ref'] #dimensionless (L/Lstar), arbitrary amplitude
      
        #Calculate optical depths over a smooth omega grid.
        smooth_oms = np.logspace(np.log10(np.abs(data['freq'].real).min())-3, np.log10(np.abs(data['freq'].real).max())+1, 100)
        smooth_depths = calculate_optical_depths(smooth_oms/(2*np.pi), self.r, bruntN2, lambS1, chi_rad, ell=self.ell)
        data['smooth_oms'] = smooth_oms
        data['smooth_depths'] = smooth_depths

        #Save arbitrary amplitude, normalized eigenfunctions to output file.
        with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(self.output_dir, self.ell), 'a') as f:
            for k in data.keys():
                f[k] = data[k]
            f['r']   = self.r
            f['x']   = self.x
            f['rho'] = self.rho
        
        return self.data_dict

    def get_Y_l(self):
        """ Retrieves the observer-angle-averaged Y_l per eqn 8 of Townsend 2002."""
        data = self.data_dict
        ms  = np.linspace(-self.ell, self.ell, 2*self.ell + 1)[:,None,None]
        phi = np.linspace(0, np.pi, 100)[None,:,None]
        dphi = np.gradient(phi.ravel())[None,:,None]
        theta = np.linspace(0, np.pi, 100)[None,None,:]
        dtheta = np.gradient(theta.ravel())[None,None,:]
        # Evaluate the average spherical harmonic at many observer locations
        # (note that sph_harm has back-to-front angle labeling!)
        data['Y_l'] = Y_l = (1/(2*self.ell+1))*(1/(4*np.pi))*np.sum(np.sum(np.sum(dtheta*dphi*np.sin(theta)*np.abs(ss.sph_harm(ms, self.ell, phi, theta)),axis=1),axis=1),axis=0)
        print('this ell, Y_l:', self.ell, Y_l)
        with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(self.output_dir, self.ell), 'a') as f:
            f['Y_l'] = Y_l
        return Y_l

    def evaluate_magnitudes(self):
        """ Evaluates the stellar surface photometric magnitude perturbation of each mode. """
        # Read summary file from GYRE
        data = self.data_dict

        # Extract radial displacement and Lagrangian luminosity perturbation
        # amplitudes (note that these are complex quantities)
        data['Delta_R'] = data['xi_r_ref']  # xi_r/R
        data['Delta_L'] = data['lag_L_ref'] # deltaL/L

        # Evaluate the effective temperature perturbation (via
        # Stefan-Boltmann law)
        data['Delta_T'] = 0.25*(data['Delta_L'] - 2*data['Delta_R']) # deltaTeff/Teff

        # Evaluate the effective gravity perturbation (via equation 8
        # of Townsend 2003)
        omega = data['omega'].real
        data['Delta_g'] = -(2 + omega**2)*data['Delta_R']

        # Inspect the Delta's
        print('Delta_R: {}'.format(data['Delta_R']))
        print('Delta_T: {}'.format(data['Delta_T']))
        print('Delta_g: {}'.format(data['Delta_g']))

        Y_l = self.get_Y_l()

        # Evaluate the differential flux functions (eqn. 14 of Townsend 2003)
        dff_R = {}
        dff_T = {}
        dff_G = {}
        dF = {}
        dF_mumag_dict = dict()

        #Calculate magnitude perturbation in mumag for eachmode, for each filter.
        for filter in self.filters:
            dff_R[filter] = (2+self.ell)*(1-self.ell)*self.I_l[filter]/self.I_0[filter]*Y_l
            dff_T[filter] = self.dI_l_dlnTeff[filter]/self.I_0[filter]*Y_l
            dff_G[filter] = self.dI_l_dlng[filter]/self.I_0[filter]*Y_l

            # Evaluate a light curve magnitude in each filter (eqn. 11 of Townsend 2003)
            dF[filter] = ((data['Delta_R']*dff_R[filter] +
                           data['Delta_T']*dff_T[filter] +
                           data['Delta_g']*dff_G[filter]))

            # Convert to micromag using Pogson's law
            dF_mumag = -2.5/np.log(10)*dF[filter]*1E6
            dF_mumag_dict['dF_mumags_{}'.format(filter)] = dF_mumag

        with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(self.output_dir, self.ell), 'a') as f:
            for k, item in dF_mumag_dict.items():
                data[k] = item
                f[k] = item
        return self.data_dict


    def calculate_duals(self, max_cond=1e8, max_eigs=1000):
        """ Calculates the velocity duals of the eigenfunctions. """
        data = self.data_dict
        ur = data['u_r_eigfunc']
        uh = data['u_h_eigfunc']

        self.dr = np.gradient(self.r)
        self.dx = np.gradient(self.x)
        def IP(ur_1,ur_2,uh_1,uh_2):
            """ Calculates the inner product of two eigenfunctions. """
            dr = self.dr
            r = self.r
            return np.sum(dr*4*np.pi*r**2*self.rho*(np.conj(ur_1)*ur_2+np.conj(uh_1)*uh_2),axis=-1)
       
        #Calculate a matrix of the inner product of each mode with each other mode.
        n_modes = ur.shape[0]
        IP_matrix = np.zeros((ur.shape[0], ur.shape[0]),dtype=np.complex128)
        for i in range(ur.shape[0]):
            if i % 10 == 0: print(i)
            for j in range(ur.shape[0]):
                IP_matrix[i,j] = IP(ur[i],ur[j],uh[i],uh[j])
            cond = np.linalg.cond(IP_matrix[:i+1,:i+1])
            if max_cond is not None and i > 0:
                if cond > max_cond:
                    n_modes = i
                    IP_matrix = IP_matrix[:n_modes,:n_modes]
                    break
            if i >= max_eigs:
                n_modes = i
                IP_matrix = IP_matrix[:n_modes,:n_modes]
                break

        print('dual IP matrix cond: {:.3e}; n_modes: {}/{}'.format(cond, n_modes, ur.shape[0]))
        cond = np.linalg.cond(IP_matrix[:i+1,:i+1]) 
        IP_inv = linalg.inv(IP_matrix)
        ur = ur[:n_modes]
        uh = uh[:n_modes]

        # Store truncated eigenvalue/eigenfunction lists which only keep as many modes as we have duals for.
        keys = list(self.data_dict.keys())
        for k in keys:
            if k == 'Y_l':
                continue
            elif 'smooth' not in k:
                self.data_dict['dual_'+k] = self.data_dict[k][:n_modes]

        #Calculate the duals
        data['u_r_dual'] = u_r_dual = np.conj(IP_inv)@ur
        data['u_h_dual'] = u_h_dual = np.conj(IP_inv)@uh

        #Check that velocity duals were evaluated correctly
        IP_check = np.zeros_like(IP_matrix)
        for i in range(ur.shape[0]):
            for j in range(ur.shape[0]):
                IP_check[i,j] = IP(u_r_dual[i], ur[j], u_h_dual[i], uh[j])
        I_matrix = np.eye(IP_matrix.shape[0])

        if np.allclose(I_matrix.real, IP_check.real, rtol=1e-6, atol=1e-6):
            print('duals properly calculated')
        else:
            print('error in dual calculation')
            import sys
            sys.exit()

        with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(self.output_dir, self.ell), 'a') as f:
            for k in data.keys():
                if 'dual' in k:
                    f[k] = data[k]

        return self.data_dict

    def calculate_transfer(self, use_delta_L=False, plot=False, N_om=1000):
        """
        Calculates the transfer function for the GYRE eigenfunctions.

        Parameters
        ----------
        use_delta_L : bool
            If True, calculate the transfer function for delta L / L eigenmode instead of
            for MSG magnitude perturbations.
        plot : bool
            If True, plot the transfer function after calculation.
        N_om : int
            Number of frequencies to calculate the transfer function for.
            Angular frequency span is 0.97*min(omega) to 1.03*max(omega), where omega is the angular eigenfrequency array.
        """
        #Load MESA file info
        p = mr.MesaData(self.mesa_LOG_file)
        mass           = (p.mass[::-1] * u.M_sun).cgs
        r              = (p.radius[::-1] * u.R_sun).cgs
        rho            = 10**p.logRho[::-1] * u.g / u.cm**3
        P              = p.pressure[::-1] * u.g / u.cm / u.s**2
        g               = constants.G.cgs*mass/r**2
        T               = p.temperature[::-1] * u.K
        opacity        = p.opacity[::-1] * (u.cm**2 / u.g)
        cp             = p.cp[::-1]  * u.erg / u.K / u.g
        csound         = p.csound[::-1] * u.cm / u.s
        dlogPdr         = -rho*g/P
        bruntN2         = p.brunt_N2[::-1] / u.s**2
        gamma1          = dlogPdr/(-g/csound**2)
        chi_rad = 16 * constants.sigma_sb.cgs * T**3 / (3 * rho**2 * cp * opacity)
        gamma = gamma1[0]

        #Calculate where the core CZ's boundary is located
        #Force the transfer function from 1.00 to 1.20 times the core CZ radius
        core_cz_radius = find_core_cz_radius(self.mesa_LOG_file)*u.cm
        r0 = 1.00 * core_cz_radius
        r1 = 1.20 * core_cz_radius

        #get info about mesa background for transfer calc
        rho_func = interp1d(r.flatten(), rho.flatten())
        chi_rad_func = interp1d(r.flatten(), chi_rad.flatten())
        N2_func = interp1d(r.flatten(), bruntN2.flatten())
        N2_max = bruntN2.max().value
        N2_force_max = N2_func(r1)
#        N2_adjust = np.sqrt(N2_max/N2_force_max)

        #Get surface eigenfunction value that the transfer function will determine the amplitude of
        if use_delta_L:
            lum_amplitudes = -2.5*1e6*self.data_dict['Y_l']*self.data_dict['dual_delta_L_dL_top']
        else:
            lum_amplitudes = self.data_dict['dual_dF_mumags_Red']

        #Get angular frequencies for transfer calculation
        values = 2*np.pi*self.data_dict['dual_freq']
        om0 = np.min(np.abs(values.real))*0.97
        om1 = np.max(values.real)*1.03
        om = np.logspace(np.log10(om0), np.log10(om1), num=N_om, endpoint=True) 

        #Get forcing radius and dual basis evaluated there.
        r_range = np.linspace(r0.value, r1.value, num=50, endpoint=True)
        uh_dual_interp = interp1d(self.r, self.data_dict['u_h_dual'][:,:], axis=-1)(r_range)

        #Calculate and store transfer function
        good_om, good_T = calculate_refined_transfer(om, values, uh_dual_interp, lum_amplitudes, r_range, self.ell, rho_func, chi_rad_func, N2_func, N2_max, gamma, plot=False)
#        good_T *= np.sqrt(N2_adjust)

        if plot:
            fig = plt.figure()
            plt.loglog(24*60*60*good_om/(2*np.pi),good_T.real, color='black', label='transfer')
            plt.loglog(24*60*60*good_om/(2*np.pi),good_T.imag, color='black', ls='--')
            plt.xlabel('frequency (inv day)')
            plt.ylabel('T')
            fig.savefig('{:s}/transfer_ell{:03d}_eigenvalues.png'.format(self.output_dir, self.ell), dpi=300, bbox_inches='tight')
            plt.close(fig)
    #
        with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(self.output_dir, self.ell), 'a') as f:
            f['transfer_om'] = good_om
            f['transfer_root_lum'] = good_T 


