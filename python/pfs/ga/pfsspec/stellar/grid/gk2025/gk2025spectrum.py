import numpy as np
from scipy.integrate import simpson

from pfs.ga.pfsspec.stellar import ModelSpectrum

class GK2025Spectrum(ModelSpectrum):
    def __init__(self, orig=None):
        super(GK2025Spectrum, self).__init__(orig=orig)

        if not isinstance(orig, GK2025Spectrum):
            pass
        else:
            pass
            
        self.is_flux_calibrated = True