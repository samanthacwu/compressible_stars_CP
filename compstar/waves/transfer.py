import numpy as np

import logging
logger = logging.getLogger(__name__)

def transfer_function(om, values, u_dual, field_outer, r_range, ell, rho_func, chi_rad_func, N2_max, gamma, discard_num=0, plot=False):
    """
    Calculates the transfer function of linear, damped waves of a field at a star/simulation's
    surface (specfied by field_outer) when driven near a radiative-convective boundary (r_range).

    The output transfer function must be multiplied by the radial velocity field (u_r) corresponding to
    the convective driving in the vicinity of the forcing radius.

    Inputs:
    ------
     om : NumPy array (float64)
        The (real) angular frequencies at which to  calculate the transfer function
     values : NumPy array (complex128)
        The (complex) eigenvalues corresponding to the gravity waves.
     u_dual : NumPy array (complex128)
        The horizontal component of the dual velocity basis; evaluated for each eigenfunction and at each forcing radius.
     field_outer : NumPy array (complex128)
        The surface value of each eigenfunction for the field that the transfer function is being calculated for (entropy, luminosity, etc.)
     r_range : NumPy array (float64)
        The radial coordinates of delta-function forcings that the transfer function is evaluated for.
     ell : float
        The spherical harmonic degree
     rho_func : function
        A function of the mass density to be evaluated at each radial coordinate in r_range.
     chi_rad_func : function
        A function of chi_rad to be evaluated at each radial coordinate in r_range.
     N2_max : float64
        Maximal value of N^2 achieved in the domain
     gamma : float
        Adiabatic index

    Outputs:
    --------
     T : NumPy array (complex128)
        The transfer function, which must be multiplied by sqrt(wave luminosity).
    """
    if discard_num > 0:
        u_dual = u_dual[:-discard_num]
        values = values[:-discard_num]
        field_outer = field_outer[:-discard_num]
    #The none's expand dims
    #dimensionality is [omega', rf, omega]
    r_range         = r_range[None, :, None]
    dr = np.ones_like(r_range)
    big_om                   = om[None, None, :] 
    u_dual           = u_dual[:, :, None]
    values           = values[:, None, None] #no rf
    field_outer = field_outer[:, None, None] #no rf
    om               = om[None, :]

    #Get structure variables
    chi_rad = chi_rad_func(r_range)
    rho = rho_func(r_range)
    cpmu_div_R = gamma/(gamma-1)

    #Get wavenumbers
    k_h = np.sqrt(ell * (ell + 1)) / r_range
    k_r = (((-1)**(3/4) / np.sqrt(2))\
                       *np.sqrt(-2*1j*k_h**2 - (big_om/chi_rad) + np.sqrt((big_om)**3 + 4*1j*k_h**2*chi_rad*N2_max)/(chi_rad*np.sqrt(big_om)) )).real
    k2 = k_r**2 + k_h**2

    #Calculate transfer
    bulk_to_bound_force = np.sqrt(2) * big_om / k_h #times ur -> comes later.
    P_ur_to_h = rho * (cpmu_div_R) * ((big_om) / k_h**2) * (-k_r) #assuming H_p -> infinity.
    root_lum_to_ur = np.sqrt(1/(4*np.pi*r_range**2*P_ur_to_h)).real

    inner_prod = 4*np.pi*r_range**2*rho*bulk_to_bound_force*root_lum_to_ur*np.conj(u_dual) * dr
    Eig = inner_prod * field_outer / ((values - big_om)*(values + big_om))
    Eig_cos = (Eig*big_om).real
    Eig_sin = (Eig*(-1j)*values).real

    T_pieces = np.abs(np.sum(Eig_cos + 1j*Eig_sin,axis=0)) # sum over eigenfunctions, then take abs()
    T = np.sqrt(np.mean(T_pieces**2, axis=0))# #get median as function of radius
    if plot:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        cmap = mpl.cm.viridis
        norm = mpl.colors.Normalize(vmin=r_range.min(), vmax=r_range.max())
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        for i in range(T_pieces.shape[0]):
            ax1.loglog(om.ravel()/(2*np.pi), T_pieces[i,:], c=sm.to_rgba(r_range.ravel()[i]))
        cmap = mpl.cm.plasma
        norm = mpl.colors.Normalize(vmin=0, vmax=Eig_cos.shape[0])
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        for i in range(Eig_cos.shape[0]):
            ax1.loglog(om.ravel()/(2*np.pi),  Eig_cos[i,0,:], c=sm.to_rgba(i))
            ax1.loglog(om.ravel()/(2*np.pi), -Eig_cos[i,0,:], ls='--', c=sm.to_rgba(i))
            ax1.set_ylabel('cos')
            ax2.loglog(om.ravel()/(2*np.pi),  Eig_sin[i,0,:], c=sm.to_rgba(i))
            ax2.loglog(om.ravel()/(2*np.pi), -Eig_sin[i,0,:], ls='--', c=sm.to_rgba(i))
            ax2.set_ylabel('sin')
        
        plt.show()

    return T

def calculate_refined_transfer(om, *args, max_iters=50, plot=False, **kwargs):
    """
    Iteratively calculates the transfer function by calling transfer_function()

    This routine locates peaks in the transfer function and inserts a denser frequency mesh
    into the frequency array around each peak to ensure proper peak resolution.

    Inputs: See transfer_function()

    Outputs:
    --------
     om: NumPy array (float64)
        The real angular frequencies at which the transfer function is calculated.
     T : NumPy array (complex128)
        Same as for transfer_function()
    """

    T = transfer_function(om, *args, plot=plot, **kwargs)

    peaks = 1
    iters = 0
    while peaks > 0 and iters < max_iters:
        i_peaks = []
        for i in range(1,len(om)-1):
            if (T[i]>T[i-1]) and (T[i]>T[i+1]):
                delta_m = np.abs(T[i]-T[i-1])/T[i]
                delta_p = np.abs(T[i]-T[i+1])/T[i]
                if delta_m > 0.01 or delta_p > 0.01:
                    i_peaks.append(i)

        peaks = len(i_peaks)
        print("number of peaks: %i" % (peaks))

        om_new = np.array([])
        for i in i_peaks:
            om_low = om[i-1]
            om_high = om[i+1]
            om_new = np.concatenate([om_new,np.linspace(om_low,om_high,10)])

        T_new = transfer_function(om_new, *args, **kwargs)

        om = np.concatenate([om,om_new])
        T = np.concatenate([T,T_new])

        om, sort = np.unique(om, return_index=True)
        T = T[sort]
        iters += 1

    return om, T


