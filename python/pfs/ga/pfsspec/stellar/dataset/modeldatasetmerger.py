from pfs.ga.pfsspec.core.dataset import SpectrumDatasetMerger
from pfs.ga.pfsspec.stellar.dataset import ModelDataset

class ModelDatasetMerger(SpectrumDatasetMerger):
    def __init__(self, orig=None):
        super(ModelDatasetMerger, self).__init__(orig=orig)

    def greate_dataset(self, preload_arrays=False):
        return ModelDataset(preload_arrays=preload_arrays)