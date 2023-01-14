"""
This file reads in gyre eigenfunctions, calculates the velocity and velocity dual basis, and outputs in a clean format so that it's ready to be fed into the transfer function calculation.
"""
import numpy as np
import tomso as tomso
from tomso import gyre
import glob
import time
import matplotlib.pyplot as plt
from scipy import linalg
import h5py
import re


def natural_sort(iterable, reverse=False):
    """
    Sort alphanumeric strings naturally, i.e. with "1" before "10".
    Based on http://stackoverflow.com/a/4836734.
    """

    convert = lambda sub: int(sub) if sub.isdigit() else sub.lower()
    key = lambda item: [convert(sub) for sub in re.split('([0-9]+)', str(item))]

    return sorted(iterable, key=key, reverse=reverse)

# load modes

def read_modes(file_bases):
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
      freq_list.append(-1j*(header['Refreq'] + 1j*header['Imfreq']))
      omega_list.append(header['Reomega'] + 1j*header['Imomega'])
      xir_list.append(data_mode['Rexi_r'] + 1j*data_mode['Imxi_r']) #units of r/R
      xih_list.append(data_mode['Rexi_h'] + 1j*data_mode['Imxi_h']) #units of r/R
      L_list.append(Lstar*(data_mode['Relag_L'] + 1j*data_mode['Imlag_L']))

  summary = tomso.gyre.load_summary(glob.glob('%s*.txt' %file_bases[0])[0])
  header = summary.header
  data_mode = summary.data

  freq = np.array(freq_list)*1e-6 #in Hz
  omega = np.array(omega_list) #dimensionless eigenfrequency
  ur = Rstar*np.array(xir_list)*1j*freq[:,None] #cm/s
  uh = Rstar*np.array(xih_list)*1j*freq[:,None] #cm/s
  L = np.array(L_list)#erg/s
  L_top = L[:,-1]
  return freq,omega,r,ur,uh,L,L_top,rho 

def calculate_duals(bases,ell,om_list):
  freq, omega, r, ur, uh, L, L_top, rho = read_modes(bases)

  def IP(ur_1,ur_2,uh_1,uh_2):
    dr = np.gradient(r)
    return np.sum(dr*4*np.pi*r**2*rho*(np.conj(ur_1)*ur_2+ell*(ell+1)*np.conj(uh_1)*uh_2),axis=-1) #TODO: check consistency
  
  IP_matrix = np.zeros((len(ur),len(ur)),dtype=np.complex128)
  for i in range(len(ur)):
    if i % 10 == 0: print(i)
    for j in range(len(ur)):
      IP_matrix[i,j] = IP(ur[i],ur[j],uh[i],uh[j])
  
  IP_inv = linalg.inv(IP_matrix)
  
  ur_dual = np.conj(IP_inv)@ur
  uh_dual = np.conj(IP_inv)@uh
  return freq, omega, r, ur, uh, L, L_top, rho, ur_dual, uh_dual

#  dx = np.gradient(x)
#  L_top_list = []
#  for i in range(50):
#    bottom_point = 0.25+1e-4*i*4
#    print(bottom_point)
#    delta = 0*x
#    RCB_index = np.argmin(np.abs(x - bottom_point))
#    delta[RCB_index] = 1./dx[RCB_index]
#    kperp = np.sqrt(ell*(ell+1))/x[RCB_index]
#    L_top_om = np.abs(np.sum( IP(ur_dual,0*x,uh_dual,delta)[:,None]/(ell*(ell+1))*L_top[:,None]*om_list[None,:]/kperp/( om_list[None,:] - 1j*freq[:,None] ), axis=0 ))
#    L_top_list.append(L_top_om)
#  L_top_list = np.mean(np.array(L_top_list),axis=0)
#
#  return L_top_list

ell=1
om_list = np.logspace(-8, -2, 1000) #Hz * 2pi

base1 = './gyre_output/mode_ell{:03d}'.format(ell)
freq, omega, r, ur, uh, L, L_top, rho, ur_dual, uh_dual = calculate_duals([base1],ell,om_list)
with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format('gyre_output', ell), 'w') as f:
    f['real_freqs'] = freq
    f['nondim_om'] = omega
    f['r'] = r
    f['ur'] = ur
    f['uh'] = uh
    f['L'] = L
    f['L_top'] = L_top
    f['rho'] = rho
    f['ur_dual'] = ur_dual
    f['uh_dual'] = uh_dual
