import numpy as np

from pfsspec.stellar import ModelSpectrum

class KuruczSpectrum(ModelSpectrum):
    def __init__(self, orig=None):
        super(KuruczSpectrum, self).__init__(orig=orig)
        