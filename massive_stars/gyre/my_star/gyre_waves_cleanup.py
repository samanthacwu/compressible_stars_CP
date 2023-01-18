"""
This file reads in gyre eigenfunctions, calculates the velocity and velocity dual basis, and outputs in a clean format so that it's ready to be fed into the transfer function calculation.
"""
import numpy as np
import tomso as tomso
from tomso import gyre
import mesa_reader as mr
import glob
import time
import matplotlib.pyplot as plt
from scipy import linalg
import h5py
import re

mesa_LOG = 'LOGS/profile47.data'
Rsun_to_cm = 6.957e10
sigma_SB = 5.67037442e-05 #cgs


def natural_sort(iterable, reverse=False):
    """
    Sort alphanumeric strings naturally, i.e. with "1" before "10".
    Based on http://stackoverflow.com/a/4836734.
    """

    convert = lambda sub: int(sub) if sub.isdigit() else sub.lower()
    key = lambda item: [convert(sub) for sub in re.split('([0-9]+)', str(item))]

    return sorted(iterable, key=key, reverse=reverse)

# load modes

def read_modes(file_bases, ell):

    #get info about mesa background
    p = mr.MesaData(mesa_LOG)
    r = p.radius[::-1]*Rsun_to_cm #in cm
    bruntN2 = p.brunt_N2[::-1] #rad^2/s^2
    lambS1  = p.lamb_S[::-1] #rad/s
    T       = p.temperature[::-1] #K
    rho     = 10**(p.logRho[::-1]) #g/cm^3
    opacity = p.opacity[::-1] #cm^2 / g
    cp      = p.cp[::-1] #erg / K / g
    chi_rad = 16 * sigma_SB * T**3 / (3 * rho**2 * cp * opacity)

    freq_list = []
    xir_list = []
    xih_list = []
    L_list = []
    omega_list = []
    file_list = natural_sort([file for base in file_bases for file in glob.glob('%s*n-*.txt' %base)])
  
    summary = tomso.gyre.load_summary(glob.glob('%s*.txt' %file_bases[0])[0])
    header = summary.header
    data_mode = summary.data
    rho = data_mode['rho'] #g / cm^3
    Rstar = header['R_star'] #cm
    Mstar = header['M_star'] #g
    Lstar = header['L_star'] #erg/s
    r = data_mode['x']*Rstar #cm
 
    for i,filename in enumerate(file_list):
        print(filename)
        summary = tomso.gyre.load_summary(filename)
        header = summary.header
        data_mode = summary.data
        freq_list.append(header['Refreq'] + 1j*header['Imfreq'])
        omega_list.append(header['Reomega'] + 1j*header['Imomega'])
        xir_list.append(data_mode['Rexi_r'] + 1j*data_mode['Imxi_r']) #units of r/R
        xih_list.append(data_mode['Rexi_h'] + 1j*data_mode['Imxi_h']) #units of r/R
        L_list.append(Lstar*(data_mode['Relag_L'] + 1j*data_mode['Imlag_L']))
#        plt.semilogy(r, bruntN2)
#        plt.semilogy(r, lambS1**2)
#        plt.axhline((2*np.pi*1e-6*freq_list[-1].real)**2)
#        plt.show()
  
    summary = tomso.gyre.load_summary(glob.glob('%s*.txt' %file_bases[0])[0])
    header = summary.header
    data_mode = summary.data
  
    freq = np.array(freq_list)*1e-6 #in Hz
    omega = np.array(omega_list) #dimensionless eigenfrequency
    #technically probs off by 2pi but doesn't matter bc eigenfunctions have arbitrary normalization.
    ur = Rstar*np.array(xir_list)*1j*freq[:,None] #cm/s
    uh = Rstar*np.array(xih_list)*1j*freq[:,None] #cm/s
    L = np.array(L_list)#erg/s
    L_top = L[:,-1]
  
  
    depths = calculate_optical_depths(freq, r, bruntN2, lambS1, chi_rad, ell=ell)
    smooth_oms = np.logspace(np.log10(np.abs(freq.real).min())-1, np.log10(np.abs(freq.real).max())+1, 100)
    smooth_depths = calculate_optical_depths(smooth_oms/(2*np.pi), r, bruntN2, lambS1, chi_rad, ell=ell)
    return freq,omega,r,ur,uh,L,L_top,rho,depths,smooth_oms, smooth_depths
  

def calculate_optical_depths(eigenfrequencies, r, N2, S1, chi_rad, ell=1):
    #Calculate 'optical depths' of each mode.
    depths = []
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


        #Numpy integrate
        opt_depth = np.trapz(depth_integrand, x=r)
        depths.append(opt_depth)
    return depths


def calculate_duals(bases,ell,om_list):
    freq, omega, r, ur, uh, L, L_top, rho, depths, smooth_oms, smooth_depths = read_modes(bases, ell)
  
    def IP(ur_1,ur_2,uh_1,uh_2):
      """
      Per daniel:
      for the inner product, you need the ell(ell+1) because what gyre calls uh is actually uh/sqrt(ell(ell+1)).
      (because the actual angular velocity has two components and is uh = xi_h * f * grad(Y_ell,m)) [so grad_h is the angular part of the gradient without any 1/r factor]
      but when you take <uh, uh> you can integrate-by-parts on one of the grad's to turn it into laplacian(Y_ell,m)=-(ell(ell+1)) Y_ell,m
      """
      dr = np.gradient(r)
      return np.sum(dr*4*np.pi*r**2*rho*(np.conj(ur_1)*ur_2+ell*(ell+1)*np.conj(uh_1)*uh_2),axis=-1)
    
    IP_matrix = np.zeros((len(ur),len(ur)),dtype=np.complex128)
    for i in range(len(ur)):
      if i % 10 == 0: print(i)
      for j in range(len(ur)):
        IP_matrix[i,j] = IP(ur[i],ur[j],uh[i],uh[j])
    
    IP_inv = linalg.inv(IP_matrix)
    
    ur_dual = np.conj(IP_inv)@ur
    uh_dual = np.conj(IP_inv)@uh
    return freq, omega, r, ur, uh, L, L_top, rho, depths, smooth_oms, smooth_depths, ur_dual, uh_dual

Lmax = 4
ell_list = np.arange(1, Lmax+1)
for ell in ell_list:
    om_list = np.logspace(-8, -2, 1000) #Hz * 2pi

    base1 = './gyre_output/mode_ell{:03d}'.format(ell)
    freq, omega, r, ur, uh, L, L_top, rho, depths, smooth_oms, smooth_depths, ur_dual, uh_dual = calculate_duals([base1],ell,om_list)
    print(freq)
    with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format('gyre_output', ell), 'w') as f:
        f['dimensional_freqs'] = freq
        f['nondim_om'] = omega
        f['depths'] = depths
        f['r'] = r
        f['ur'] = ur
        f['uh'] = uh
        f['L'] = L
        f['L_top'] = L_top
        f['rho'] = rho
        f['ur_dual'] = ur_dual
        f['uh_dual'] = uh_dual
        f['smooth_oms'] = smooth_oms
        f['smooth_depths'] = smooth_depths
