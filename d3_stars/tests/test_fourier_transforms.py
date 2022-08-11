""" Tests of spherical harmonic transform operations. """

import pytest
import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt

import dedalus.public as d3
from d3_stars.post.power_spectrum_functions import FourierTransformer, ShortTimeFourierTransformer


@pytest.mark.parametrize('N', [1000, 10000])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('modes', [1, 2, 3])
@pytest.mark.parametrize('window', [None, np.hanning])
def test_power(N, dtype, modes, window):
    t_final = 200
    time = np.linspace(0, t_final, N, endpoint=False)
    dt = (time.max()-time.min())/(N-1)
    f_nyq = 1/(2*dt)

    data = np.zeros_like(time, dtype=dtype)
    for f in np.arange(modes):
        freq = f_nyq/(2*(f+1))
        omega = 2*np.pi*freq
        data += np.array(5*np.exp(1j*omega*time), dtype=dtype)
    time_series_power = np.sum(np.conj(data)*data*dt)/(N*dt)

    transformer = FourierTransformer(time, data, window=window)
    transformer.take_transform()
    power = transformer.get_power()
    transformed_power = np.sum(power)

    assert np.allclose(transformed_power, time_series_power, rtol=2/N)

@pytest.mark.parametrize('N', [1000, 10000])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('modes', [1, 2, 5])
@pytest.mark.parametrize('window', [None, np.hanning])
def test_peak_power(N, dtype, modes, window):
    t_final = 200
    time = np.linspace(0, t_final, N, endpoint=False)
    dt = (time.max()-time.min())/(N-1)
    f_nyq = 1/(2*dt)
    f_min = 1/t_final


    data = np.zeros_like(time, dtype=dtype)
    freqs = []
    for i in np.arange(modes):
        freq = f_nyq/2 - f_min*(i+1)*2
        omega = 2*np.pi*freq
        freqs.append(freq)
        data += np.array(5*np.exp(1j*omega*time), dtype=dtype)

    power = 5**2

    transformer = FourierTransformer(time, data, window=window)
    transformer.take_transform()
    for f in freqs:
        print(f, transformer.get_peak_power(f))
        assert np.allclose(power, transformer.get_peak_power(f), rtol=2/N)

@pytest.mark.parametrize('N', [1000, 10000])
@pytest.mark.parametrize('min_factor', [1/10, 1/20, 1/50])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('modes', [1,])
@pytest.mark.parametrize('window', [np.hanning])
def test_stft_setup(N, dtype, modes, window, min_factor):
    t_final = 200
    time = np.linspace(0, t_final, N, endpoint=False)
    dt = (time.max()-time.min())/(N-1)
    f_nyq = 1/(2*dt)
    f_min = 1/t_final
    min_freq = min_factor*f_nyq


    data = np.zeros_like(time, dtype=dtype)
    freqs = []
    for i in np.arange(modes):
        freq = f_nyq/2 - min_freq*(i+1)*2
        omega = 2*np.pi*freq
        freqs.append(freq)
        data += np.array(time*np.exp(1j*omega*time), dtype=dtype)

    transformer = ShortTimeFourierTransformer(time, data, min_freq, window=window)
    assert np.allclose(transformer.dt, dt) #ensure proper dt calculation

    transformer.take_transforms()
    for fc in transformer.freq_chunks:
        #ensure that FTs have proper min frequency
        assert np.allclose(min_freq, fc[fc > 0].min())

@pytest.mark.parametrize('N', [1000, 10000])
@pytest.mark.parametrize('min_factor', [1/20, 1/50])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('modes', [1,2])
@pytest.mark.parametrize('window', [np.hanning])
def test_stft_peak_power_evolution(N, dtype, modes, window, min_factor):
    t_final = 200
    time = np.linspace(0, t_final, N, endpoint=False)
    dt = (time.max()-time.min())/(N-1)
    f_nyq = 1/(2*dt)
    f_min = 1/t_final
    min_freq = min_factor*f_nyq


    data = np.zeros_like(time, dtype=dtype)
    freqs = []
    for i in np.arange(modes):
        freq = f_nyq/2 - min_freq*(i+1)*4
        omega = 2*np.pi*freq
        freqs.append(freq)
        data += np.array(((freq/f_nyq)*(time+dt)/t_final)*np.exp(1j*omega*time), dtype=dtype)

    transformer = ShortTimeFourierTransformer(time, data, min_freq, window=window)
    transformer.take_transforms()
    times, pows = transformer.get_peak_evolution(freqs)

    for f in freqs:
        analytic_power = ((f/f_nyq)*(times+dt)/t_final)**2
        p = pows[f]
        print(p/analytic_power)
        assert np.allclose(p, analytic_power, rtol=2/transformer.stft_N)


