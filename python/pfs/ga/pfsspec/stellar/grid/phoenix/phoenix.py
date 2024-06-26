import numpy as np

from pfs.ga.pfsspec.stellar.grid import ModelGridConfig
from .phoenixspectrum import PhoenixSpectrum
    
class Phoenix(ModelGridConfig):
    def __init__(self, orig=None, normalized=False, pca=False):
        super().__init__(orig=orig, normalized=normalized, pca=pca)

        if isinstance(orig, Phoenix):
            pass
        else:
            pass

    def add_args(self, parser, config):
        super().add_args(parser, config)

    def init_from_args(self, config, args):
        super().init_from_args(config, args)
        
    def init_axes(self, grid):
        grid.init_axis('M_H', np.hstack((np.arange(-4.0, -2.0, 1),
                                          np.arange(-2.0, 1.5, 0.50))))
        grid.init_axis('T_eff', np.hstack((np.arange(2300.0, 7000.0, 100.0),
                                           np.arange(7000.0, 12200.0, 200.0))))
        grid.init_axis('log_g', np.arange(0, 6.5, 0.5))
        grid.init_axis('a_M', np.arange(-0.2, 1.4, 0.2))

        grid.build_axis_indexes()

    def init_values(self, grid):
        super().init_values(grid)
      
    def allocate_values(self, grid, wave, wave_edges=None):
        super().allocate_values(grid, wave, wave_edges=wave_edges)

    def create_spectrum(self):
        spec = PhoenixSpectrum()
        return spec
