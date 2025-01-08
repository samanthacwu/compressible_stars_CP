from collections import OrderedDict
import numpy as np

import mesa_reader as mr
from astropy import units as u
from astropy import constants

import logging
logger = logging.getLogger(__name__)


class DimensionalMesaReader:
    """ Class to read in MESA profile and store it in a dictionary with astropy units """

    def __init__(self, filename):
        #TODO: figure out how to make MESA the file path w.r.t. stock model path w/o supplying full path here
        logger.info("Reading MESA file {}".format(filename))
        self.p = p = mr.MesaData(filename)
        self.structure = OrderedDict()
        self.structure['mass']           = mass           = (p.mass[::-1] * u.M_sun).cgs
        self.structure['r']              = r              = (p.radius[::-1] * u.R_sun).cgs
        self.structure['rho']            = rho            = 10**p.logRho[::-1] * u.g / u.cm**3
        self.structure['P']              = P              = p.pressure[::-1] * u.g / u.cm / u.s**2
        self.structure['T']              = T              = p.temperature[::-1] * u.K
        self.structure['nablaT']         = nablaT         = p.gradT[::-1] #dlnT/dlnP
        self.structure['nablaT_ad']      = nablaT_ad      = p.grada[::-1]
        self.structure['chiRho']         = chiRho         = p.chiRho[::-1]
        self.structure['chiT']           = chiT           = p.chiT[::-1]
        self.structure['cp']             = cp             = p.cp[::-1]  * u.erg / u.K / u.g
        self.structure['opacity']        = opacity        = p.opacity[::-1] * (u.cm**2 / u.g)
        self.structure['Luminosity']     = Luminosity     = (p.luminosity[::-1] * u.L_sun).cgs
        self.structure['conv_L_div_L']   = conv_L_div_L   = p.lum_conv_div_L[::-1]
        self.structure['csound']         = csound         = p.csound[::-1] * u.cm / u.s
        self.structure['N2']             = N2 = N2_mesa   = p.brunt_N2[::-1] / u.s**2
        self.structure['N2_structure']   = N2_structure   = p.brunt_N2_structure_term[::-1] / u.s**2
        self.structure['N2_composition'] = N2_composition = p.brunt_N2_composition_term[::-1] / u.s**2
        self.structure['eps_nuc']        = eps_nuc        = p.eps_nuc[::-1] * u.erg / u.g / u.s
        self.structure['mu']             = mu             = p.mu[::-1] * u.g / u.mol
        self.structure['lamb_freq']      = lamb_freq = lambda ell : np.sqrt(ell*(ell + 1)) * csound/r

        self.structure['R_star'] = R_star = (p.photosphere_r * u.R_sun).cgs

        self.structure['R_gas'] = R_gas                     = constants.R.cgs / mu[0]
        self.structure['g'] = g                             = constants.G.cgs*mass/r**2
        self.structure['dlogPdr'] = dlogPdr                 = -rho*g/P
        self.structure['gamma1'] = gamma1                   = dlogPdr/(-g/csound**2)
        self.structure['dlogrhodr'] = dlogrhodr             = dlogPdr*(chiT/chiRho)*(nablaT_ad - nablaT) - g/csound**2
        self.structure['dlogTdr'] = dlogTdr                 = dlogPdr*(nablaT)
        self.structure['grad_s_over_cp'] = grad_s_over_cp   = N2/g #entropy gradient, for NCC, includes composition terms
        self.structure['grad_s'] = grad_s                   = cp * grad_s_over_cp
        self.structure['L_conv'] = L_conv                   = conv_L_div_L*Luminosity
        self.structure['dTdr'] = dTdr                       = (T)*dlogTdr

        # Calculate k_rad and radiative diffusivity using luminosities and smooth things.
        self.structure['k_rad']    = k_rad    = rad_cond = -(Luminosity - L_conv)/(4*np.pi*r**2*dTdr)
        self.structure['rad_diff'] = rad_diff = k_rad / (rho * cp)
        #rad_diff        = (16 * constants.sigma_sb.cgs * T**3 / (3 * rho**2 * cp * opacity)).cgs # this is less smooth


def find_core_cz_radius(mesa_file, dimensionless=True, L_conv_threshold=1):
    """ 
    Find the radius of the core convection zone in a MESA profile.
    The convective core is defined as the region where the convective luminosity is greater than a threshold
    and the mass coordinate is less than 90% of the total stellar mass.

    Arguments:
    ----------
    mesa_file : str
        Path to MESA profile file
    dimensionless : bool
        If True, return the radius as a float without astropy units attached.
    L_conv_threshold : float
        Threshold for determining if a region is convective.
    """
    p = mr.MesaData(mesa_file)
    r              = (p.radius[::-1] * u.R_sun).cgs
    mass           = (p.mass[::-1] * u.M_sun).cgs
    Luminosity     = (p.luminosity[::-1] * u.L_sun).cgs
    conv_L_div_L   = p.lum_conv_div_L[::-1]
    L_conv         = conv_L_div_L*Luminosity

    cz_bool = (L_conv.value > L_conv_threshold)*(mass < 0.9*mass[-1])
    core_index  = np.argmin(np.abs(mass - mass[cz_bool][-1]))
    core_cz_radius = r[core_index]
    if dimensionless: #no astropy units.
        return core_cz_radius.value
    else:
        return core_cz_radius
    
def adjust_opacity(mesa_file, dimensionless=False):
        """
        Change the opacity to one that follows a Kramer's opacity + constant (electron scattering)
        """
        p = mr.MesaData(mesa_file)
        opacity = p.opacity[::-1] #* (u.cm**2 / u.g)
        h1 = p.h1[::-1]
        he3 = p.he3[::-1]
        he4 = p.he4[::-1]
        ye = p.ye[::-1]
        x_frac = h1
        z_frac = 1-(h1 + he3 + he4)
        rho = 10**p.logRho[::-1] #* u.g / u.cm**3
        T = p.temperature[::-1] #* u.K

        kappa_ff = lambda rho,T,Z,X: 3.68e22*(1-Z)*(1+X)*rho*T**(-7/2)
        # kappa_bf = lambda rho,T,Z,X: 4.34e25*Z*(1+X)*rho*T**(-7/2)

        z_frac = (1-h1-he4-he3)[0]
        x_frac = h1[0]
        new_opacity = ye[0]*(0.2*(1+x_frac)+kappa_ff(rho,T,z_frac,x_frac))
        gff = (opacity[0] - new_opacity[0])/ye[0]/kappa_ff(rho,T,z_frac,x_frac)[0] + 1
        # print(gff)
        opacity_adj = ye[0]*(0.2*(1+x_frac)+gff*kappa_ff(rho,T,z_frac,x_frac)) 
        if dimensionless:
            return opacity_adj.value, gff, z_frac, x_frac, ye[0]
        else:
            return opacity_adj * (u.cm**2 / u.g), gff, z_frac, x_frac, ye[0]
def opacity_func(rho,T,gff,z_frac,x_frac,ye,dimensionless=False): #this is dimensional
    kappa_ff = lambda rho,T,Z,X: 3.68e22*(1-Z)*(1+X)*rho*T**(-7/2)
    if dimensionless:
        return ye*(0.2*(1+x_frac)+gff*kappa_ff(rho,T,z_frac,x_frac))
    else:
        return ye*(0.2*(1+x_frac)+gff*kappa_ff(rho,T,z_frac,x_frac)) * (u.cm**2 / u.g)