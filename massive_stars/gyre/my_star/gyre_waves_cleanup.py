"""
This file reads in gyre eigenfunctions, calculates the velocity and velocity dual basis, and outputs in a clean format so that it's ready to be fed into the transfer function calculation.
"""
import re
import os

import h5py
import numpy as np
import pygyre as pg
import pymsg as pm
import tomso as tomso
from tomso import gyre
import mesa_reader as mr
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.interpolate import interp1d
import scipy.special as ss

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

def calculate_optical_depths(eigenfrequencies, r, N2, S1, chi_rad, ell=1):
    #Calculate 'optical depths' of each mode.
    depths = []
#    chi_rad = chi_rad.min()
    for freq in eigenfrequencies.real:
        freq = np.abs(freq)
        om = 2*np.pi*freq
        lamb_freq = np.sqrt(ell*(ell+1) / 2) * S1
        wave_cavity = (2*np.pi*freq < np.sqrt(N2))*(2*np.pi*freq < lamb_freq)
        depth_integrand = np.zeros_like(lamb_freq)

        # from Lecoanet et al 2015 eqn 12. This is the more universal function
        Lambda = np.sqrt(ell*(ell+1))
        k_perp = Lambda/r
        kz = ((-1)**(3/4)/np.sqrt(2))*np.sqrt(-1j*2*k_perp**2 - (om/chi_rad) + np.sqrt(om**3 + 1j*4*k_perp**2*chi_rad*N2)/(chi_rad*np.sqrt(om)) )
#        kz = np.sqrt(-k_perp**2 + 1j*((2*np.pi*freq)/(2*chi_rad))*(1 - np.sqrt(1 + 1j*4*(N2*chi_rad*k_perp**2 / (2*np.pi*freq)**3))))
        depth_integrand[wave_cavity] = kz[wave_cavity].imag
#        plt.semilogy(r, depth_integrand)
#        plt.show()


        #Numpy integrate
        opt_depth = np.trapz(depth_integrand, x=r)
        depths.append(opt_depth)
    return depths



class GyreMSGPostProcessor:

    def __init__(self, ell, pos_summary, pos_details, neg_summary, neg_details, mesa_pulse_file, mesa_LOG_file, 
                 initial_z=0.006, specgrid=None, filters=['Red',], 
                 MSG_DIR = os.environ['MSG_DIR'], 
                 GRID_DIR=None,
                 PASS_DIR=os.path.join('..','gyre_phot','passbands')):
        self.ell = ell
        self.pos_summary = pos_summary
        self.neg_summary = neg_summary
        self.pos_details = pos_details
        self.neg_details = neg_details
        self.mesa_pulse_file = mesa_pulse_file
        self.mesa_LOG_file = mesa_LOG_file
        self.filters = filters

        # Load the MSG photometric grids (this code taken from the 
        # Python walkthrough in the MSG docs)
        if GRID_DIR is None:
            os.path.join(MSG_DIR, 'data', 'grids'),
        
        if specgrid == 'OSTAR2002':
            specgrid_file_name = os.path.join(GRID_DIR, 'sg-OSTAR2002-low.h5')
        else:
            specgrid_file_name = os.path.join(GRID_DIR, 'sg-demo.h5')

        self.photgrids = {}
        for filter in self.filters:
            passband_file_name = os.path.join(PASS_DIR, f'pb-TESS-TESS.{filter}-Vega.h5')
            print(specgrid_file_name, passband_file_name)
            self.photgrids[filter] = pm.PhotGrid(specgrid_file_name, passband_file_name)


            # Inspect grid parameters
            print('Grid parameters:')
            for label in self.photgrids[filter].axis_labels:
                print(f'  {label} [{self.photgrids[filter].axis_x_min[label]} -> {self.photgrids[filter].axis_x_max[label]}]')

        # Load the stellar model to figure out Teff and gravity
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

        # Evaluate the intensity moments (eqn. 15 of Townsend 2003) and their
        # partials
        self.I_0 = {}
        self.I_l = {}
        self.dI_l_dlnTeff = {}
        self.dI_l_dlng = {}

        for filter in self.filters:

            self.I_0[filter] = self.photgrids[filter].D_moment(self.model_x, 0)
            self.I_l[filter] = self.photgrids[filter].D_moment(self.model_x, ell)
            print("I_0, I_l: {:.2e}, {:.2e}".format(self.I_0[filter], self.I_l[filter]))

            self.dI_l_dlnTeff[filter] = self.photgrids[filter].D_moment(self.model_x, 0, deriv={'Teff': True})*self.Teff
            self.dI_l_dlng[filter] = self.photgrids[filter].D_moment(self.model_x, 0, deriv={'log(g)': True})/np.log(10)
            print("dI_l_dlnTeff, dI_l_dlng: {:.2e}, {:.2e}".format(self.dI_l_dlnTeff[filter], self.dI_l_dlng[filter]))

    def evaluate_magnitudes(self, observer=(np.pi/3,np.pi/6),m=0):
        self.data_dicts = []
        for file_list, summary_file in zip((self.pos_details, self.neg_details),(self.pos_summary, self.neg_summary)):
            # Read summary file from GYRE
            summary = pg.read_output(summary_file)
            data = dict()
            for k in ['l', 'n_pg',]:
                data[k] = np.zeros(len(file_list))
            for k in ['freq', 'omega', 'xi_r_ref', 'lag_L_ref']:
                data[k] = np.zeros(len(file_list), dtype=np.complex128)

            for i,filename in enumerate(file_list):
                header = tomso.gyre.load_summary(filename).header
                data['l'][i] = ell
                data['n_pg'][i] = header['n_pg']
                this_row = summary[(summary['l'] == ell)*(summary['n_pg'] == header['n_pg'])]
                for k in ['freq', 'omega', 'xi_r_ref', 'lag_L_ref']:
                    data[k][i] = np.array(this_row[k],dtype=np.complex128)[0]


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

            # Evaluate the spherical harmonic at the observer location
            # (note that sph_harm has back-to-front angle labeling!)

            theta_obs, phi_obs = observer
            Ylm = ss.sph_harm(m, self.ell, phi_obs, theta_obs)

            # Evaluate the differential flux functions (eqn. 14 of Townsend 2003)

            dff_R = {}
            dff_T = {}
            dff_G = {}
            dF = {}
            dF_mumag_dict = dict()

            for filter in self.filters:
                
                dff_R[filter] = (2+ell)*(1-ell)*self.I_l[filter]/self.I_0[filter]*Ylm
                dff_T[filter] = self.dI_l_dlnTeff[filter]/self.I_0[filter]*Ylm
                dff_G[filter] = self.dI_l_dlng[filter]/self.I_0[filter]*Ylm

                # Evaluate a light curve in each filter (eqn. 11 of Townsend 2003)
                dF[filter] = ((data['Delta_R']*dff_R[filter] +
                               data['Delta_T']*dff_T[filter] +
                               data['Delta_g']*dff_G[filter]))

                # Convert to micromag using Pogson's law
                dF_mumag = -2.5/np.log(10)*dF[filter]*1E6
                dF_mumag_dict[filter] = dF_mumag
            data['dF_mumags'] = dF_mumag_dict
            self.data_dicts.append(data)
        return self.data_dicts


    def sort_eigenfunctions(self):
        #TODO: move these background info reading lines up to __init__()
        #get info about mesa background
        p = mr.MesaData(self.mesa_LOG_file)
        r_mesa = p.radius[::-1]*Rsun_to_cm #in cm
        bruntN2_mesa = p.brunt_N2[::-1] #rad^2/s^2
        lambS1_mesa  = p.lamb_S[::-1] #rad/s
        T       = p.temperature[::-1] #K
        rho     = 10**(p.logRho[::-1]) #g/cm^3
        opacity = p.opacity[::-1] #cm^2 / g
        cp      = p.cp[::-1] #erg / K / g
        chi_rad_mesa = 16 * sigma_SB * T**3 / (3 * rho**2 * cp * opacity)

        #Get pulsation & stratification information    
        summary = tomso.gyre.load_summary(self.pos_details[0])
        header = summary.header
        data_mode = summary.data
        self.rho = data_mode['rho'] #g / cm^3
        self.x = data_mode['x']
        self.r = self.x*self.R #cm

        V = data_mode['V_2']*self.x**2
        As = data_mode['As']
        c_1 = data_mode['c_1']
        Gamma_1 = data_mode['Gamma_1']

        bruntN2 = As/c_1
        lambS1 = np.sqrt(1*(1+1))*np.sqrt(Gamma_1/(V*c_1))
        chi_rad = 10**(interp1d(r_mesa, np.log10(chi_rad_mesa), bounds_error=False, fill_value='extrapolate')(self.r))

        #re-dimensionalize
        mid_r = self.r[len(self.r)//2]
        bruntN2 *= 10**(interp1d(r_mesa, np.log10(bruntN2_mesa))(mid_r)) / bruntN2[len(self.r)//2]
        lambS1 *= 10**(interp1d(r_mesa, np.log10(lambS1_mesa))(mid_r)) / lambS1[len(self.r)//2]

        #data_dicts already has 'freq', 'omega', 'xi_r_ref', 'lag_L_ref', 'l', 'n_pg'
        iter = 0
        for file_list, summary_file in zip((self.pos_details, self.neg_details),(self.pos_summary, self.neg_summary)):
            data = self.data_dicts[iter]
            for field in ['xi_r_eigfunc', 'xi_h_eigfunc', 'lag_L_eigfunc', 'u_r_eigfunc', 'u_h_eigfunc']:
                data[field] = np.zeros((len(data['n_pg']), len(self.x)), dtype=np.complex128) 
            data['depth'] = np.zeros(len(data['n_pg']))
            for i,filename in enumerate(file_list):
                print('reading eigenfunctions from {}'.format(filename))
                summary = tomso.gyre.load_summary(filename)
                header = summary.header
                data_mode = summary.data

                data['depth'][i] = calculate_optical_depths(np.array([data['freq'][i]*1e-6,]), self.r, bruntN2, lambS1, chi_rad, ell=self.ell)[0]
                data['xi_r_eigfunc'][i,:] = data_mode['Rexi_r'] + 1j*data_mode['Imxi_r'] #units of r/R
                data['xi_h_eigfunc'][i,:] = data_mode['Rexi_h'] + 1j*data_mode['Imxi_h'] #units of r/R
                data['lag_L_eigfunc'][i,:] = data_mode['Relag_L'] + 1j*data_mode['Imlag_L'] #units of L/L_star
          
            #Is this off by 2\pi? u = dt(xi) by defn.

            data['u_r_eigfunc'] = -1j*data['omega'][:,None]*data['xi_r_eigfunc']
            data['u_h_eigfunc'] = -1j*data['omega'][:,None]*data['xi_h_eigfunc']
            data['L_top'] = data['lag_L_eigfunc'][:,-1]
          
            smooth_oms = np.logspace(np.log10(np.abs(data['freq'].real*1e-6).min())-3, np.log10(np.abs(data['freq'].real*1e-6).max())+1, 100)
            smooth_depths = calculate_optical_depths(smooth_oms/(2*np.pi), self.r, bruntN2, lambS1, chi_rad, ell=self.ell)
            data['smooth_oms'] = smooth_oms
            data['smooth_depths'] = smooth_depths
            iter += 1
        return data_dicts

    def calculate_duals(self):
        data = self.data_dicts[0]
        ur = data['u_r_eigfunc']
        uh = data['u_h_eigfunc']
      
        def IP(ur_1,ur_2,uh_1,uh_2):
          """
          Per daniel:
          for the inner product, you need the ell(ell+1) because what gyre calls uh is actually uh/sqrt(ell(ell+1)).
          (because the actual angular velocity has two components and is uh = xi_h * f * grad(Y_ell,m)) [so grad_h is the angular part of the gradient without any 1/r factor]
          but when you take <uh, uh> you can integrate-by-parts on one of the grad's to turn it into laplacian(Y_ell,m)=-(ell(ell+1)) Y_ell,m
          """
          dx = np.gradient(self.x)
          return np.sum(dx*4*np.pi*self.x**2*self.rho*(np.conj(ur_1)*ur_2+self.ell*(self.ell+1)*np.conj(uh_1)*uh_2),axis=-1)
        
        IP_matrix = np.zeros((len(data['n_pg']),len(data['n_pg'])),dtype=np.complex128)
        for i in range(len(ur)):
          if i % 10 == 0: print(i)
          for j in range(len(ur)):
            IP_matrix[i,j] = IP(ur[i],ur[j],uh[i],uh[j])

        print('dual IP matrix cond: {:.3e}'.format(np.linalg.cond(IP_matrix)))
        
        IP_inv = linalg.inv(IP_matrix)
        
        data['u_r_dual'] = np.conj(IP_inv)@ur
        data['u_h_dual'] = np.conj(IP_inv)@uh
        self.data_dicts[1]['u_r_dual'] = np.conj(data['u_r_dual'])
        self.data_dicts[1]['u_h_dual'] = np.conj(data['u_h_dual'])
        return self.data_dicts

Lmax = 1
ell_list = np.arange(1, Lmax+1)
for ell in ell_list:
    om_list = np.logspace(-8, -2, 1000) #Hz * 2pi

    pulse_file = 'LOGS/profile47.data.GYRE'
    mesa_LOG = 'LOGS/profile47.data'
    pos_mode_base = './gyre_output/pos_mode_ell{:03d}_m+00_n{:06d}.txt'
    neg_mode_base = pos_mode_base.replace('pos', 'neg')
    pos_files = []
    neg_files = []

    neg_summary_file='gyre_output/neg_ell{:02d}_summary.txt'.format(ell)
    pos_summary_file='gyre_output/pos_ell{:02d}_summary.txt'.format(ell)
    neg_summary = pg.read_output(neg_summary_file)
    pos_summary = pg.read_output(pos_summary_file)

    neg_ell = neg_summary['l']
    neg_n_pg = neg_summary['n_pg']
    for row in pos_summary:
        ell = row['l']
        n_pg = row['n_pg']
        freq = row['freq']
        found_negative = (ell in neg_ell)*(n_pg in neg_n_pg)
        if found_negative:
            neg_freq = neg_summary[(neg_ell == ell)*(n_pg == neg_n_pg)]['freq']
            if len(neg_freq) > 1:
                print("skipping {}; too many negative frequencies".format(freq))
                continue
            good = (1 - neg_freq.imag/freq.imag < 1e-10)*(1 + neg_freq.real/freq.real < 1e-10)
            if good:
                pos_files.append(pos_mode_base.format(ell, n_pg))
                neg_files.append(neg_mode_base.format(ell, n_pg))

    pos_files = pos_files[::-1]
    neg_files = neg_files[::-1]

    post = GyreMSGPostProcessor(ell, pos_summary_file, pos_files, neg_summary_file, neg_files, pulse_file, mesa_LOG,
                  specgrid='OSTAR2002', filters=['Red',],
                  MSG_DIR = os.environ['MSG_DIR'],
                  GRID_DIR=os.path.join('..','gyre-phot','specgrid'),
                  PASS_DIR=os.path.join('..','gyre-phot','passbands'))
    data_dicts = post.evaluate_magnitudes(observer=(np.pi/3,np.pi/6),m=0)
    post.sort_eigenfunctions()
    data_dicts = post.calculate_duals()

    print(data_dicts[0])

    with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format('gyre_output', ell), 'w') as f:
        for k in data_dicts[0].keys():
            if k == 'dF_mumags':
                f[k+'_Red'] = np.concatenate((data_dicts[0][k]['Red'], data_dicts[1][k]['Red']), axis=0)
            else:
                f[k] = np.concatenate((data_dicts[0][k], data_dicts[1][k]), axis=0)
        f['r'] = post.r
        f['x'] = post.x
        f['rho'] = post.rho

