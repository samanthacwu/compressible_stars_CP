""" Tests of spherical harmonic transform operations. """

import pytest
import numpy as np
from scipy.special import sph_harm

import dedalus.public as d3
from d3_stars.post.transforms import SHTransformer


ntheta = 8
nphi = 2*ntheta

def analytic_sph_harm(ell, m, dtype):
    """ 
    analytical spherical harmonics taken from wikipedia:
        https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    
    Returns a function (arguments: phi, theta) and the integrated power (defined as avg(func**2))
    """
    if ell == 0:
        func = lambda phi, theta: np.sqrt(1/(4*np.pi))
        power = 1/(4*np.pi)
    elif ell == 1:
        if np.abs(m) == 1:
            func = lambda phi, theta: -m*0.5*np.sqrt(3/(2*np.pi))*np.exp(1j*phi)*np.sin(theta)
            if dtype is np.float64:
                power = 0.5/(4*np.pi)
            elif dtype is np.complex128:
                power = 1/(4*np.pi)
            else:
                raise NotImplementedError("Invalid dtype")
        elif m == 0:
            func = lambda phi, theta: 0.5 * np.sqrt(3/np.pi)*np.cos(theta)
            power = 1/(4*np.pi)
    else:
        raise NotImplementedError("ell is too large")
    return func, power

@pytest.mark.parametrize('ell', [0,1,2])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('vector', [True, False])
def test_ell_m_indexing(ell, dtype, vector):
    m_range = np.arange(ell+1, dtype=np.int32)
    transformer = SHTransformer(nphi, ntheta, dtype=dtype)

    for m in m_range:
        if vector:
            field = np.zeros_like(transformer.vector_field['g'])
            field[2,:] = np.array(sph_harm(m, ell, transformer.phi, transformer.theta), dtype=dtype)
            transform = transformer.transform_vector_field(field)
        else:
            field = np.zeros_like(transformer.vector_field['g'])
            field = np.array(sph_harm(m, ell, transformer.phi, transformer.theta), dtype=dtype)
            transformer.transform_scalar_field(field)
       
        #make sure there isn't power at ell' != ell and m' != m
        other_pow = 0
        for ell_prime in range(ntheta-1):
            for m_prime in range(ell_prime):
                if ell_prime == ell and m_prime == m: continue
                val = transformer.get_ell_m_value(ell_prime, m_prime)
                other_pow += np.conj(val)*val
        assert np.allclose(other_pow, 0)

        #Make sure there is power at ell, m.
        true_val = transformer.get_ell_m_value(ell, m)
        power = true_val*np.conj(true_val)
        assert np.sum(power) > 0


@pytest.mark.parametrize('ell', [0,1,2])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('vector', [True, False])
def test_power_conservation(ell, dtype, vector):
    m_range = np.arange(ell+1, dtype=np.int32)
    transformer = SHTransformer(nphi, ntheta, dtype=dtype)
    if vector:
        grid_power_op = d3.Average(transformer.conj_vector_field@transformer.vector_field)
    else:
        grid_power_op = d3.Average(transformer.scalar_field*np.conj(transformer.scalar_field))

    for m in m_range:
        if vector:
            field = np.zeros_like(transformer.vector_field['g'])
            field[2,:] = np.array(sph_harm(m, ell, transformer.phi, transformer.theta), dtype=dtype)
            transform = transformer.transform_vector_field(field)
        else:
            field = np.zeros_like(transformer.vector_field['g'])
            field = np.array(sph_harm(m, ell, transformer.phi, transformer.theta), dtype=dtype)
            transformer.transform_scalar_field(field)

        transformed_power = np.sum(np.conj(transformer.out_field)*transformer.out_field)
        grid_power = grid_power_op.evaluate()['g']

        assert np.allclose(transformed_power, grid_power)

@pytest.mark.parametrize('ell', [0,1])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('vector', [True, False])
def test_power_value(ell, dtype, vector):
    m_range = np.arange(2, dtype=np.int32)
    transformer = SHTransformer(nphi, ntheta, dtype=dtype)

    goodval = True
    for m in m_range:
        func, power = analytic_sph_harm(ell, m, dtype)
        if vector:
            field = np.zeros_like(transformer.vector_field['g'])
            field[2,:] = np.array(func(transformer.phi, transformer.theta), dtype=dtype)
            transform = transformer.transform_vector_field(field)
        else:
            field = np.zeros_like(transformer.vector_field['g'])
            field = np.array(func(transformer.phi, transformer.theta), dtype=dtype)
            transformer.transform_scalar_field(field)

        transformed_power = np.sum(np.conj(transformer.out_field)*transformer.out_field)
        if not np.allclose(transformed_power, power):
            goodval = False
            print(ell, m, transformed_power, power)
    assert goodval
