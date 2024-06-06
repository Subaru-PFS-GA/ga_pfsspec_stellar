import numpy as np
from scipy.integrate import simps

from pfs.ga.pfsspec.stellar import ModelSpectrum

class Grid7Spectrum(ModelSpectrum):
    def __init__(self, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, Grid7Spectrum):
            pass
        else:
            pass
            
    