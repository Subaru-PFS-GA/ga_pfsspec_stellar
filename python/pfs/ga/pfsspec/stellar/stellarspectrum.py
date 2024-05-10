import numpy as np

from pfs.ga.pfsspec.core import Spectrum
from pfs.ga.pfsspec.core import Physics, Astro

class StellarSpectrum(Spectrum):
    # TODO: make it a mixin instead of an inherited class
    def __init__(self, orig=None):
        super().__init__(orig=orig)
        
        if not isinstance(orig, StellarSpectrum):
            self.Fe_H = np.nan
            self.Fe_H_err = np.nan
            self.M_H = np.nan
            self.M_H_err = np.nan
            self.C_M = np.nan
            self.C_M_err = np.nan
            self.O_M = np.nan
            self.O_M_err = np.nan
            self.a_M = np.nan
            self.a_M_err = np.nan
            
            self.T_eff = np.nan
            self.T_eff_err = np.nan
            self.log_g = np.nan
            self.log_g_err = np.nan
        else:
            self.Fe_H = orig.Fe_H
            self.Fe_H_err = orig.Fe_H_err
            self.M_H = orig.M_H
            self.M_H_err = orig.M_H_err
            self.a_M = orig.a_M
            self.a_M_err = orig.a_M_err
            self.C_M = orig.C_M
            self.C_M_err = orig.C_M_err
            self.O_M = orig.O_M
            self.O_M_err = orig.O_M_err
            self.T_eff = orig.T_eff
            self.T_eff_err = orig.T_eff_err
            self.log_g = orig.log_g
            self.log_g_err = orig.log_g_err

    def get_param_names(self):
        params = super(StellarSpectrum, self).get_param_names()
        params = params + [
            'Fe_H', 'Fe_H_err',
            'M_H', 'M_H_err',
            'a_M', 'a_M_err',
            'C_M', 'C_M_err',
            'O_M', 'O_M_err',
            'T_eff', 'T_eff_err',
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
        return Physics.stellar_radius(log_L, log_T_eff)

