import numpy as np

from ..modelgridconfig import ModelGridConfig
from .grid7spectrum import Grid7Spectrum

class Grid7(ModelGridConfig):
    def __init__(self, orig=None, normalized=False, pca=False):
        super().__init__(orig=orig, normalized=normalized, pca=pca)

        if isinstance(orig, Grid7):
            pass
        else:
            pass

    def add_args(self, parser, config):
        super().add_args(parser, config)

    def init_from_args(self, config, args):
        super().init_from_args(config, args)

    def init_axes(self, grid):
        eps = 0.0001
        grid.init_axis('M_H', np.arange(-5.0, 0.0, 0.1 + eps))
        grid.init_axis('T_eff', np.hstack((np.arange(3500.0, 5500.0 + eps, 100.0),
                                           np.arange(5600, 8000.0 + eps, 200.0))))
        grid.init_axis('log_g', np.arange(0, 5.0 + eps, 0.5))
        grid.init_axis('a_M', np.arange(-0.8, 1.2 + eps, 0.1))

        grid.build_axis_indexes()

    def init_values(self, grid):
        super().init_values(grid)
      
    def allocate_values(self, grid, wave, wave_edges=None):
        super().allocate_values(grid, wave, wave_edges=wave_edges)

    def create_spectrum(self):
        spec = Grid7Spectrum()
        return spec
