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
        """ 
        returns the power of a peak at the given frequency as if that peak corresponded to a sine wave.
        So if your signal has a sine wave, A * sin(omega * t), we expect a peak at f = +/- omega/(2pi).
        Each peak will have amplitude (A/2), so the total power will be 2(A^2/4) = A^2/2.

        In the complex case, you can have a real and imaginary wave component with total amplitude A^2.
        So the complex case has a cos^2 + sin^2 ~ 1 thing going for it.
        The real case only has cos^2 ~ 1/2 and needs the extra factor of 2.
        """
        if self.complex:
            return self.power_interp(freq) * self.amp_norm**2 * (self.N/(self.N-2))
        else:
            return 2*self.power_interp(freq) * self.amp_norm**2 * (self.N/(self.N-2))

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

