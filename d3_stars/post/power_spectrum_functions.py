import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

hann_power_normalizer = 8/3
hann_amp_normalizer = 2


class FourierTransformer:

    def __init__(self, times, signal, window=np.hanning):
        self.times = times
        self.signal = np.array(signal)
        self.N = self.times.shape[0]
        if window is None:
            self.window = np.ones(self.N)
        else:
            self.window = window(self.N)
        while len(self.window.shape) < len(self.signal.shape):
            self.window = np.expand_dims(self.window, axis=-1)
        
        self.power_norm = 1
        self.amp_norm = 1
        if np.hanning == window:
            print('using hann window')
            self.power_norm = hann_power_normalizer
            self.amp_norm = hann_amp_normalizer

       
        if self.signal.dtype == np.float64:
            self.complex = False
        else:
            self.complex = True

        self.freqs = None
        self.ft = None
        self.power = None
        self.power_freqs = None


    def take_transform(self):
        """
        Takes the Fourier transform of a time series of real data.
        Utilizes a Hann window & normalizes appropriately so that the integrated power spectrum has the same power as the time series.
        """
        if self.complex:
            self.clean_cfft()
        else:
            self.clean_rfft()
        self.power_interp = interp1d(self.power_freqs, self.power)
        return self.freqs, self.ft

    def clean_rfft(self):
        self.freqs = np.fft.rfftfreq(self.times.shape[0], d=np.median(np.gradient(self.times.flatten()))) 
        self.ft = np.fft.rfft(self.window*self.signal, axis=0, norm="forward")
        self.power = self.normalize_rfft_power()
        self.power_freqs = self.freqs
        return self.freqs, self.ft

    def clean_cfft(self):
        self.freqs = np.fft.fftfreq(self.times.shape[0], d=np.median(np.gradient(self.times.flatten()))) 
        self.ft = np.fft.fft(self.window*self.signal, axis=0, norm="forward")
        self.power = self.normalize_cfft_power()
        return self.freqs, self.ft

    def get_power(self):
        """ returns power spectrum, accounting for window normalization so that parseval's theorem is satisfied"""
        return self.power * self.power_norm

    def get_power_freqs(self):
        """ returns power spectrum, accounting for window normalization so that parseval's theorem is satisfied"""
        return self.power_freqs

    def get_peak_power(self, freq):
        """ returns the power at a given frequency, normalized so that the window does not mess with its value if it's a peak"""
        return self.power_interp(freq) * self.amp_norm

    def normalize_cfft_power(self):
        """
        Calculates the power spectrum of a complex fourier transform by collapsing negative and positive frequencies.
        """
        power = (self.ft*np.conj(self.ft)).real
        self.power_freqs = np.unique(np.abs(self.freqs))
        self.power = np.zeros((self.power_freqs.size,*tuple(power.shape[1:])))
        for i, f in enumerate(self.power_freqs):
            good = np.logical_or(self.freqs == f, self.freqs == -f)
            self.power[i] = np.sum(power[good], axis=0)
        return self.power

    def normalize_rfft_power(self):
        """
        Calculates the power spectrum of a real fourier transform accounting for its hermitian-ness
        """
        power = (self.ft*np.conj(self.ft)).real
        self.power_freqs = np.unique(np.abs(self.freqs))
        self.power = np.zeros((self.power_freqs.size,*tuple(power.shape[1:])))
        for i, f in enumerate(self.power_freqs):
            if f != 0:
                self.power[i] = 2*power[i] #account for negative frequencies which are conj(positive freqs)
            else:
                self.power[i] = power[i]
        return self.power



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
