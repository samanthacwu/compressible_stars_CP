import numpy as np

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
        depth_integrand[wave_cavity] = kz[wave_cavity].imag


        #Numpy integrate
        opt_depth = np.trapz(depth_integrand, x=r)
        depths.append(opt_depth)
    return depths



