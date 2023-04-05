import numpy as np

from ..modelgridconfig import ModelGridConfig
from .boszspectrum import BoszSpectrum

class Bosz(ModelGridConfig):
    def __init__(self, orig=None, normalized=False, pca=False):
        super().__init__(orig=orig, normalized=normalized, pca=pca)

        if isinstance(orig, Bosz):
            pass
        else:
            pass

    def add_args(self, parser):
        super().add_args(parser)

    def init_from_args(self, args):
        super().init_from_args(args)

    def init_axes(self, grid):
        grid.init_axis('M_H', np.arange(-2.5, 1.0, 0.25))
        grid.init_axis('T_eff', np.hstack((np.arange(3500.0, 12250.0, 250.0),
                                             np.arange(12500.0, 20000.0, 500.0),
                                             np.arange(20000.0, 36000.0, 1000.0))))
        grid.init_axis('log_g', np.arange(0, 5.5, 0.5))
        grid.init_axis('C_M', np.arange(-0.75, 0.75, 0.25))
        grid.init_axis('a_M', np.arange(-0.25, 0.75, 0.25))

        grid.build_axis_indexes()

    def init_values(self, grid):
        super().init_values(grid)
      
    def allocate_values(self, grid, wave, wave_edges=None):
        super().allocate_values(grid, wave, wave_edges=wave_edges)

    def create_spectrum(self):
        spec = BoszSpectrum()
        return spec
