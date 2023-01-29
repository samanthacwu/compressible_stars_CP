"""
d3 script for eigenvalue problem of anelastic convection / waves in a massive star.

"""
import gc
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from docopt import docopt
from configparser import ConfigParser
import dedalus.public as d3
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import logging
logger = logging.getLogger(__name__)

from d3_stars.simulations.evp_functions import StellarEVP

if __name__ == '__main__':
    eigenvalues = StellarEVP()
    for ell in eigenvalues.ells:
        if ell == 1:
            max_modes = 45
        else:
            max_modes = 50
#        elif ell == 2:
#            max_modes = 55
#        elif ell == 3:
#            max_modes = 60
#        elif ell == 4:
#            max_modes = 65
#        elif ell == 5:
#            max_modes = 70
#        elif ell == 6:
#            max_modes = 75
#        else:
#            max_modes = 80
        logger.info('solving EVP for ell = {}'.format(ell))
        eigenvalues.solve(ell)
        eigenvalues.check_eigen(max_modes=max_modes)
        eigenvalues.output()
        eigenvalues.get_duals()
        logger.info('solved EVP for ell = {}'.format(ell))
