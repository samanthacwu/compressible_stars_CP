
import numpy as np
import tomso as tomso
from tomso import gyre
import glob
import time
import matplotlib.pyplot as plt
from scipy import linalg

# load modes

def read_modes(file_bases):
  freq_list = []
  xir_list = []
  xih_list = []
  L_list = []
  omega_list = []
  print(file_bases)
  file_list = [file for base in file_bases for file in glob.glob('%s*n-*.txt' %base)]

  for i,filename in enumerate(file_list):
    print(filename)
    summary = tomso.gyre.load_summary(filename)
    header = summary.header
    data_mode = summary.data
    freq_list.append(-1j*(header['Refreq'] + 1j*header['Imfreq']))
    omega_list.append(header['Reomega'] + 1j*header['Imomega'])
    xir_list.append(data_mode['Rexi_r'] + 1j*data_mode['Imxi_r'])
    xih_list.append(data_mode['Rexi_h'] + 1j*data_mode['Imxi_h'])
    L_list.append(data_mode['Relag_L'] + 1j*data_mode['Imlag_L'])
  
  summary = tomso.gyre.load_summary(glob.glob('%s*.txt' %file_bases[0])[0])
  header = summary.header
  data_mode = summary.data
  rho = data_mode['rho']
  x = data_mode['x']

  freq = np.array(freq_list)*1e-6*24*60*60
  omega = np.array(omega_list)
  ur = np.array(xir_list)*1j*freq[:,None]
  uh = np.array(xih_list)*1j*freq[:,None]
  L = np.array(L_list)
  L_top = L[:,-1]

  return freq,omega,x,ur,uh,L,L_top,rho 

def calculate_L(bases,ell,om_list):
  freq, omega, x, ur, uh, L, L_top, rho = read_modes(bases)

  def IP(ur_1,ur_2,uh_1,uh_2):
    dx = np.gradient(x)
    return np.sum(dx*4*np.pi*x**2*rho*(np.conj(ur_1)*ur_2+ell*(ell+1)*np.conj(uh_1)*uh_2),axis=-1)
  
  IP_matrix = np.zeros((len(ur),len(ur)),dtype=np.complex128)
  for i in range(len(ur)):
    if i % 10 == 0: print(i)
    for j in range(len(ur)):
      IP_matrix[i,j] = IP(ur[i],ur[j],uh[i],uh[j])
  
  IP_inv = linalg.inv(IP_matrix)
  
  ur_dual = np.conj(IP_inv)@ur
  uh_dual = np.conj(IP_inv)@uh

  dx = np.gradient(x)
  L_top_list = []
  for i in range(50):
    bottom_point = 0.25+1e-4*i*4
    print(bottom_point)
    delta = 0*x
    RCB_index = np.argmin(np.abs(x - bottom_point))
    delta[RCB_index] = 1./dx[RCB_index]
    kperp = np.sqrt(ell*(ell+1))/x[RCB_index]
    L_top_om = np.abs(np.sum( IP(ur_dual,0*x,uh_dual,delta)[:,None]/(ell*(ell+1))*L_top[:,None]*om_list[None,:]/kperp/( om_list[None,:] - 1j*freq[:,None] ), axis=0 ))
    L_top_list.append(L_top_om)
  L_top_list = np.mean(np.array(L_top_list),axis=0)

  return L_top_list

om_list = np.logspace(-2, 2, 1000)

base1 = './gyre_output/mode_ell001'
L_top1 = calculate_L([base1],1,om_list)

plt.loglog(om_list, L_top1)
plt.show()

print(om_list, L_top1)

data = np.vstack([om_list,L_top1])
np.savetxt('L_GYRE.dat',data)

