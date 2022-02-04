import numpy as np
import matplotlib.pyplot as plt

hann_power_normalizer = 8/3


#def hann_windwindow_normalize_fraction(times, window, data):
#    """
#    Calculates the fraction by which a power spectrum must be boosted to "undo" an applied window function.
#    For the Hann window, in the analytic limit, this function would return 8/3.
#
#    Inputs:
#    -------
#        times : numpy array (float64)
#            A 1-D array of time values
#        window : numpy array (float64)
#            An N-D array of the window function (where N is the dimensionality of the data, the first dimension has the same shape as time, and the other dimensions have shape 1)
#        data : numpy array (float64 or complex128)
#            An N-D array of data values, where the first dimension is the time dimension.
#
#    Returns:
#    -------
#        the power in the time series divided by the power in the windowed time series.
#    """
#    dt = np.gradient(times.flatten())
#    for i in range(len(data.shape) - 1):
#        dt = np.expand_dims(dt, axis=-1)
#    if data.dtype == np.float64:
#        true_power = np.sum((data)**2*dt, axis=0)/np.sum(dt, axis=0)
#        windowed_power = np.sum((window*data)**2*dt, axis=0)/np.sum(dt, axis=0)
#    elif data.dtype == np.complex128:
#        true_power = np.sum((data*np.conj(data)).real*dt, axis=0)/np.sum(dt, axis=0)
#        windowed_power = np.sum(window**2*(data*np.conj(data)).real*dt, axis=0)/np.sum(dt, axis=0)
#    fraction = true_power/windowed_power
#    if len(data.shape) > 1:
#        fraction = np.expand_dims(fraction, axis=0)
#    return fraction

def clean_rfft(times, data):
    """
    Takes the Fourier transform of a time series of real data.
    Utilizes a Hann window & normalizes appropriately so that the integrated power spectrum has the same power as the time series.

    Inputs
    ------
        times : numpy array (float64)
            A 1-D array of time values
        data : numpy array (float64)
            An N-D array of data values, where the first dimension is the time dimension.

    Returns
    -------
        freqs : numpy array (float64)
            A 1-D array of (positive) frequency values, in inverse units of the input times array.
        ft : numpy array (complex128)
            An N-D array of the fourier transformed data, where the first dimension is the frequency dimension.
    """
    N = times.shape[0]
    window = np.hanning(N)
    while len(window.shape) < len(data.shape):
        window = np.expand_dims(window, axis=-1)
    freqs = np.fft.rfftfreq(times.shape[0], d=np.median(np.gradient(times.flatten()))) 
    ft = np.fft.rfft(window*data, axis=0)/N
    ft[freqs != 0] *= np.sqrt(2)
#    window_fraction = window_normalize_fraction(times, window, data)
    window_fraction = hann_power_normalizer
    ft *= np.sqrt(window_fraction) #Hann window correction factor
    return freqs, ft

def clean_cfft(times, data):
    """
    Takes the Fourier transform of a time series of complex data.
    Utilizes a Hann window & normalizes appropriately so that the integrated power spectrum has the same power as the time series.

    Inputs
    ------
        times : numpy array (float64)
            A 1-D array of time values
        data : numpy array (complex128)
            An N-D array of data values, where the first dimension is the time dimension.

    Returns
    -------
        freqs : numpy array (float64)
            A 1-D array of (negative and positive) frequency values, in inverse units of the input times array.
        ft : numpy array (complex128)
            An N-D array of the fourier transformed data, where the first dimension is the frequency dimension.
    """

    N = times.shape[0]
    window = np.hanning(N)
    while len(window.shape) < len(data.shape):
        window = np.expand_dims(window, axis=-1)
    freqs = np.fft.fftfreq(times.shape[0], d=np.median(np.gradient(times.flatten()))) 
    ft = np.fft.fft(window*data, axis=0)/N
#    window_fraction = window_normalize_fraction(times, window, data)
    window_fraction = hann_power_normalizer
    ft *= np.sqrt(window_fraction)
    return freqs, ft

def normalize_cfft_power(freqs, ft):
    """
    Calculates the power spectrum of a complex fourier transform by collapsing negative and positive frequencies.

    Inputs
    ------
        freqs : numpy array (float64)
            A 1-D array of frequency values
        ft : numpy array (complex128)
            An N-D array of the fourier-transformed data values, where the first dimension is the frequency dimension.

    Returns
    -------
        p_freqs : numpy array (float64)
            The positive frequency values.
        p_power : numpy array (float64)
            An N-D array of the power spectrum, where the first dimension is the frequency dimension.

    """
    power = (ft*np.conj(ft)).real
    for f in freqs:
        if f < 0:
            power[freqs == -f] += power[freqs == f]
    if len(power.shape) == 1:
        p_power = power[freqs >= 0]
    else:
        p_power = power[freqs >= 0,:]
    p_freqs = freqs[freqs >= 0]
    return p_freqs, p_power

if __name__ == "__main__":
    N = 10000
    t_start = 0
    t_final = 200
    t_tot   = t_final - t_start
    time = np.linspace(t_start, t_final, N)
    dt = np.gradient(time)

    #Properly normalized real FT
    data = np.ones_like(time)
    for f in np.arange(10):
        data += 5*np.cos(2*np.pi*0.05*(f+1)*time)
    time_series_power = np.sum((data)**2*dt)/np.sum(dt)

    freqs, ft = clean_rfft(time, data)
    power_spectrum = (ft*np.conj(ft)).real
    fft_power = np.sum(power_spectrum)

    print('real', time_series_power, fft_power, time_series_power - fft_power, (time_series_power - fft_power)/time_series_power)


    #Properly normalized complex FT
    data = np.ones_like(time, dtype=np.complex128)
    for f in np.arange(10):
        data += 5*np.cos(2*np.pi*0.05*(f+1)*time) * ((f+1) % 2 +  1j*(f % 2))
    time_series_power = np.sum(data*np.conj(data)*dt)/np.sum(dt)

    freqs, ft = clean_cfft(time, data)
    freqs, power = normalize_cfft_power(freqs, ft)
    fft_power = np.sum(power)

    print('complex', time_series_power, fft_power, (time_series_power - fft_power).real/time_series_power.real)


    ##Now we test multiple dimensions.
    Nx = 100
    x = np.linspace(0, 10, Nx).reshape(1, Nx)
    dx = np.gradient(x.flatten()).reshape(1, Nx)
    time = time.reshape(N, 1)
    dt = dt.reshape(N, 1)

    #Properly normalized real FT (multi-D)
    data = np.ones((N, Nx))
    for f in np.arange(10):
        for kx in np.arange(5):
            data += 5*np.cos(2*np.pi*0.05*(f+1)*time)*np.cos(kx*x)
    time_series_power = np.sum((data)**2*dt*dx)/np.sum(dt)

    freqs, ft = clean_rfft(time, data)
    power_spectrum = (ft*np.conj(ft)).real
    fft_power = np.sum(dx*power_spectrum)

    print('real', time_series_power, fft_power, time_series_power - fft_power, (time_series_power - fft_power)/time_series_power)


    #Properly normalized complex FT
    data = np.ones((N, Nx), dtype=np.complex128)
    for f in np.arange(10):
        for kx in np.arange(5):
            data += 5*np.cos(2*np.pi*0.05*(f+1)*time)*np.cos(kx*x) * ((f+1) % 2 +  1j*(f % 2))
    time_series_power = np.sum(data*np.conj(data)*dt*dx)/np.sum(dt)

    freqs, ft = clean_cfft(time, data)
    freqs, power = normalize_cfft_power(freqs, ft)
    fft_power = np.sum(power*dx)

    print('complex', time_series_power, fft_power, (time_series_power - fft_power).real/time_series_power.real)
