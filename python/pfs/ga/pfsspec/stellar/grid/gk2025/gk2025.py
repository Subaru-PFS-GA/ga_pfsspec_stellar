import numpy as np

from ..modelgridconfig import ModelGridConfig
from .gk2025spectrum import GK2025Spectrum

class GK2025(ModelGridConfig):
    def __init__(self, orig=None, normalized=False, pca=False):
        super().__init__(orig=orig, normalized=normalized, pca=pca)

        if isinstance(orig, GK2025):
            pass
        else:
            pass

    def add_args(self, parser, config):
        super().add_args(parser, config)

    def init_from_args(self, config, args):
        super().init_from_args(config, args)

    def init_axes(self, grid):
        grid.init_axis('M_H', np.array([
            -5.0,
            -4.8, -4.6, -4.4, -4.2, -4.0,
            -3.8, -3.6, -3.4, -3.2, -3.0,
            -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0,
            -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0,
            -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,  0.0,
             0.1,  0.2,  0.3,  0.4,  0.5]))
        grid.init_axis('T_eff', np.array([
            3500, 3600, 3700, 3800, 3900,
            4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900,
            5000, 5100, 5200, 5300, 5400, 5500, 5600, 5800,
            6000, 6200, 6400, 6600, 6800,
            7000, 7200, 7400, 7600, 7800,
            8000]))
        grid.init_axis('log_g', np.array([
            0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]))
        grid.init_axis('a_M', np.array([
            -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,  0.0,
             0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,
             1.0,  1.1,  1.2]))
        grid.init_axis('C', np.array([
            -0.4, -0.2, 0.0, 0.2, 0.4]))

        grid.build_axis_indexes()

    def init_values(self, grid):
        super().init_values(grid)
        grid.init_value('line', pca=self.pca)
      
    def allocate_values(self, grid, wave, wave_edges=None):
        super().allocate_values(grid, wave, wave_edges=wave_edges)
        if self.pca is not None and self.pca:
            raise NotImplementedError()
        else:
            grid.allocate_value('line', wave.shape, dtype=float)

    def get_chunk_shape(self, grid, name, shape, s=None):
        return super().get_chunk_shape(grid, name, shape, s)

    def create_spectrum(self):
        spec = GK2025Spectrum()
        return spec
