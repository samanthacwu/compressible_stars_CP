"""
This script plots snapshots of the evolution of 2D slices from a 2D simulation in polar geometry.

The fields specified in 'fig_type' are plotted (temperature and enstrophy by default).
To plot a different set of fields, add a new fig type number, and expand the fig_type if-statement.
"""
import re
import gc
import os
import time
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
from configparser import ConfigParser
import dedalus.public as d3
from dedalus.tools import logging
from scipy import sparse
from mpi4py import MPI

from plotpal.file_reader import SingleTypeReader as SR
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.config import config


# Parameters
dtype = np.float64
dealias = 1
radius = 2.3
c = d3.SphericalCoordinates('φ', 'θ', 'r')
dealias_tuple = (dealias, dealias, dealias)
resolution = (128, 64, 1)
Lmax = resolution[1]-1

dr = 1e-4
shell_vol = (4/3)*np.pi*(radius**3 - (radius-dr)**3)
dist = d3.Distributor((c,), mesh=None, comm=MPI.COMM_SELF, dtype=dtype)
basis = d3.ShellBasis(c, resolution, radii=(radius-dr, radius), dtype=dtype, dealias=dealias_tuple)
φg, θg, rg = basis.global_grids(basis.dealias)

ell_maps = basis.ell_maps
m_maps   = basis.sphere_basis.m_maps

max_ell = 0
max_m = 0
for m, mg_slice, mc_slice, n_slice in m_maps:
    if m > max_m:
        max_m = m
for ell, m_ind, ell_ind in ell_maps:
    if ell > max_ell:
        max_ell = ell

ell_values = np.arange(max_ell+1).reshape((max_ell+1,1))
m_values = np.arange(max_m+1).reshape((1,max_m+1))

scalar_field = dist.Field(bases=basis)
vector_field = dist.VectorField(bases=basis, coordsys=c)
power_scalar_op = d3.integ(scalar_field**2) / shell_vol
power_vector_op = d3.integ(vector_field@vector_field) / shell_vol

slices = dict()
for i in range(scalar_field['c'].shape[0]):
    for j in range(scalar_field['c'].shape[1]):
        groups = basis.elements_to_groups((False, False, False), (np.array((i,)),np.array((j,)), np.array((0,))))
        m = groups[0][0]
        ell = groups[1][0]
        key = '{},{}'.format(ell, m)
        this_slice = (slice(i, i+1, 1), slice(j, j+1, 1), slice(None))
        if key not in slices.keys():
            slices[key] = [this_slice]
        else:
            slices[key].append(this_slice)

#Test scalar
scalar_field['g'] = np.sin(θg)**2 * np.cos(5*np.pi*φg)
shape = [0,0]
shape[0] = ell_values.shape[0]
shape[1] = m_values.shape[1]
scalar_out_field = np.zeros(shape, dtype=np.complex128)

#Test vector
vector_field['g'][0] = np.sin(θg)**2 * np.cos(5*np.pi*φg)
vector_field['g'][1] = np.sin(3*θg)**2 * np.cos(2*np.pi*φg)
vector_field['g'][2] = np.sin(5*θg)**2 * np.cos(10*np.pi*φg)
shape = [3,0,0]
shape[1] = ell_values.shape[0]
shape[2] = m_values.shape[1]
vector_out_field = np.zeros(shape, dtype=np.complex128)

for j, ell in enumerate(ell_values.squeeze()):
    for k, m in enumerate(m_values.squeeze()):
        if '{},{}'.format(ell,m) in slices.keys():
            sl = slices['{},{}'.format(ell,m)]
            value1 = scalar_field['c'][sl[0]].squeeze()
            value2 = scalar_field['c'][sl[1]].squeeze()
            scalar_out_field[j,k] = value1 + 1j*value2
            for v in range(3):
                v_sl1 = tuple([v] + list(sl[0]))
                v_sl2 = tuple([v] + list(sl[1]))
                value1 = vector_field['c'][v_sl1].squeeze()
                value2 = vector_field['c'][v_sl2].squeeze()
                vector_out_field[v,j,k] = value1 + 1j*value2

scalar_out_field[0,:]  /= np.sqrt(2) #m == 0 normalization
scalar_out_field[1:,:] /= 2*np.sqrt(np.pi)          #m != 0 normalization
vector_out_field[:,0,:]  /= np.sqrt(2) #m == 0 normalization
vector_out_field[:,1:,:] /= 2*np.sqrt(np.pi)          #m != 0 normalization

#Check power conservation
power_scalar_transform = np.sum(scalar_out_field * np.conj(scalar_out_field)).real
power_vector_transform = np.sum(vector_out_field * np.conj(vector_out_field)).real
power_scalar = ((((radius)**3 - (radius-dr)**3)/3)*np.trapz(np.trapz(np.sin(θg)*scalar_field['g']**2, axis=1, x=θg), axis=0, x=φg.ravel()))[0] / shell_vol
print(power_scalar/power_scalar_transform)
power_scalar = power_scalar_op.evaluate()['g'].ravel()[0]
print(power_scalar/power_scalar_transform)
power_vector = power_vector_op.evaluate()['g'].ravel()[0]
logger.info('power factors: scalar {:.3e}, vector {:.3e}'.format(power_scalar/power_scalar_transform, power_vector/power_vector_transform))
logger.info('inverse power factors: scalar {:.3e}, vector {:.3e}'.format(1/(power_scalar/power_scalar_transform), 1/(power_vector/power_vector_transform)))
