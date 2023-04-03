import numpy as np

from .stellarspectrum import StellarSpectrum

class ModelSpectrum(StellarSpectrum):
    # TODO: Make StellarSpectrum a mixin

    def __init__(self, orig=None):
        super().__init__(orig=orig)
        
        if isinstance(orig, ModelSpectrum):
            self.N_He = orig.N_He
            self.v_turb = orig.v_turb
            self.L_H = orig.L_H
            self.interp_param = orig.interp_param
        else:
            self.N_He = np.nan
            self.v_turb = np.nan
            self.L_H = np.nan
            self.interp_param = ''

    def get_param_names(self):
        params = super().get_param_names()
        params = params + ['N_He',
                           'v_turb',
                           'L_H',
                           'interp_param']
        return params
