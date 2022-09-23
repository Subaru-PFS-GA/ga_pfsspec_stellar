import numpy as np

from pfs.ga.pfsspec.core import Spectrum
from pfs.ga.pfsspec.core import Physics

class StellarSpectrum(Spectrum):
    # TODO: make it a mixin instead of an inherited class
    def __init__(self, orig=None):
        super(StellarSpectrum, self).__init__(orig=orig)
        
        if isinstance(orig, StellarSpectrum):
            self.T_eff = orig.T_eff
            self.T_eff_err = orig.T_eff_err
            self.log_g = orig.log_g
            self.log_g_err = orig.log_g_err
        else:
            self.T_eff = np.nan
            self.T_eff_err = np.nan
            self.log_g = np.nan
            self.log_g_err = np.nan

    def get_param_names(self):
        params = super(StellarSpectrum, self).get_param_names()
        params = params + ['T_eff', 'T_eff_err',
                           'log_g', 'log_g_err']
        return params

    def set_rv(self, rv):
        z = Physics.vel_to_z(rv)
        self.set_redshift(z)

    def normalize_by_T_eff(self, T_eff=None):
        T_eff = T_eff or self.T_eff
        self.logger.debug('Normalizing spectrum with black-body of T_eff={}'.format(T_eff))
        n = 1e-7 * Physics.planck(self.wave*1e-10, T_eff)
        self.multiply(1 / n)

    def denormalize_by_T_eff(self, T_eff=None):
        T_eff = T_eff or self.T_eff
        self.logger.debug('Denormalizing spectrum with black-body of T_eff={}'.format(T_eff))
        n = 1e-7 * Physics.planck(self.wave*1e-10, T_eff)
        self.multiply(n)

    def get_radius(self, log_L, log_T_eff):
        sb = 5.67e-5                            # grams s^-3 kelvin^-4
        lsun = 3.8e33                           # erg/s 
        l = lsun * (10 ** log_L)                # luminosity from isochrone is in log(L/lsun)
        t = np.round(10 ** log_T_eff)           # T_eff from isochrone is in log(teff)
        radius = np.sqrt(l / (4 * np.pi * sb * t**4))
        return radius

    def print_info(self):
        super(StellarSpectrum, self).print_info()

        print('T_eff=', self.T_eff)
        print('log g=', self.log_g)