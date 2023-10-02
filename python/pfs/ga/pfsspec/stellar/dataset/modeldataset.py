from pfs.ga.pfsspec.core.dataset import SpectrumDataset
from pfs.ga.pfsspec.stellar import ModelSpectrum

class ModelDataset(SpectrumDataset):
    def __init__(self, constant_wave=True, preload_arrays=False, orig=None):
        super(ModelDataset, self).__init__(constant_wave=constant_wave, preload_arrays=preload_arrays, orig=orig)

    def create_spectrum(self):
        return ModelSpectrum()