import numpy as np
import pysynphot
import pysynphot.binning
import pysynphot.spectrum
import pysynphot.reddening
from scipy.integrate import simps

from pfs.ga.pfsspec.stellar.grid.kurucz import KuruczSpectrum

class BoszSpectrum(KuruczSpectrum):
    def __init__(self, orig=None):
        super(BoszSpectrum, self).__init__(orig=orig)

        if not isinstance(orig, BoszSpectrum):
            self.M_H = np.nan
            self.M_H_err = np.nan
            self.a_M = np.nan
            self.a_M_err = np.nan
            self.C_M = np.nan
            self.C_M_err = np.nan
        else:
            self.M_H = orig.M_H
            self.M_H_err = orig.M_H_err
            self.a_M = orig.a_M
            self.a_M_err = orig.a_M_err
            self.C_M = orig.C_M
            self.C_M_err = orig.C_M_err

    def get_param_names(self):
        params = super().get_param_names()
        params = params + [
            'M_H', 'M_H_err',
            'a_M', 'a_M_err',
            'C_M', 'C_M_err'
        ]
        return params

    def synthmag_carrie(self, filter, log_L):
        '''
        pass in a filter (filter), which is just a dataframe/structured array with columns of wavelength and throughput. Here, 
        the format is filter.wave for wavelength dimension, filter.thru for throughput dimension. You also pass in a spectrum 
        with columns of wavelength and flux (here self.wave, self.flux). Ensure your flux units are flam, and your wavelength
        units are angstroms. You also pass in a luminosity (given as log L/Lsun) and temperature (log Teff) that you get from 
        either divine knowledge or an isochrone. These are used to compute the radius of the star, which scales the flux 
        (Radius of the star / 10 pc)^2 scales the flux such that we get out absolute magnitudes.
        '''
        
        #normalising spectra
        #getting bounds of integral
        lam = self.wave[(self.wave <= filter.wave.max()) & (self.wave >= filter.wave.min())]
        T = np.interp(lam, filter.wave, filter.thru)
        T = np.where(T < .001, 0, T)

        R = self.get_radius(log_L, np.log10(self.T_eff))
        #1/(3.08567758128*10**(19))**2 is just 1/10pc^2 in cm! (1/(3.086e19)**2)
        
        s = self.flux[(self.wave <= filter.wave.max()) & (self.wave >= filter.wave.min())]
        s = s * np.pi * (R / 3.086e19) ** 2         #multiply by pi!!
        
        # Doing classic integral to get flux in bandpass
        stzp = 3.631e-9
        i1 = simps(s * T * lam, lam)
        i2 = simps(T * lam, lam)
        i3 = simps(T / lam, lam)
        a = -2.5 * np.log10(i1 / (stzp * i2))
        b = -2.5 * np.log10(i2 / i3)
        
        return a + b + 18.6921

    def print_info(self):
        super(BoszSpectrum, self).print_info()

        print('[M/H]=', self.M_H)
        print('[a/M]=', self.a_M)