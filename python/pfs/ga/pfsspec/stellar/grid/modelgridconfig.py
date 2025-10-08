import numpy as np

from pfs.ga.pfsspec.core.grid import GridConfig
from pfs.ga.pfsspec.stellar import ModelSpectrum
from pfs.ga.pfsspec.stellar.continuum.models import PiecewiseChebyshev
from pfs.ga.pfsspec.stellar.continuum.models import Planck
from pfs.ga.pfsspec.stellar.continuum.models import Alex
from pfs.ga.pfsspec.stellar.continuum.models import Log

class ModelGridConfig(GridConfig):
    # Implements functions to initialize a model grid. Inherited classes should
    # implement grid-specific functionality in overridden functions.

    CONTINUUM_MODEL_TYPES = {
        'planck': Planck,
        'alex': Alex,
        'chebyshev': PiecewiseChebyshev,
        'log': Log,
    }

    def __init__(self, normalized=False, pca=False, orig=None):
        super(ModelGridConfig, self).__init__(orig=orig)

        if isinstance(orig, ModelGridConfig):
            self.chunking = orig.chunking
            self.compression = orig.compression
            self.continuum_model_type = orig.continuum_model_type
            self.normalized = normalized if normalized is not None else orig.normalized
            self.pca = pca if pca is not None else orig.pca
        else:
            self.chunking = 'none'
            self.compression = 'none'
            self.continuum_model_type = None
            self.normalized = normalized
            self.pca = pca

    def add_args(self, parser, config):
        # TODO: remove
        # choices = [k for k in ModelGridConfig.CONTINUUM_MODEL_TYPES.keys()]
        # parser.add_argument('--continuum-model', type=str, choices=choices, help='Continuum model.\n')
        pass

    def init_from_args(self, config, args):
        if 'continuum_model' in args and args['continuum_model'] is not None:
            self.continuum_model_type = ModelGridConfig.CONTINUUM_MODEL_TYPES[args['continuum_model']]

    def init_axes(self, grid):
        grid.init_axis('Fe_H')
        grid.init_axis('T_eff')
        grid.init_axis('log_g')

    def init_values(self, grid):
        grid.init_value('flux', pca=self.pca)
        grid.init_value('cont', pca=self.pca)

    def allocate_values(self, grid, wave, wave_edges=None):
        if self.pca is not None and self.pca:
            raise NotImplementedError()
        else:
            grid.allocate_value('flux', wave.shape, dtype=float)
            grid.allocate_value('cont', wave.shape, dtype=float)

    def create_spectrum(self):
        return ModelSpectrum()

    def create_continuum_model(self):
        if self.continuum_model_type is not None:
            model = self.continuum_model_type()
            return model
        else:
            return None

    def is_value_valid(self, grid, name, value):
        return np.logical_not(np.any(np.isnan(value), axis=-1)) & ((value.max(axis=-1) != 0) | (value.min(axis=-1) != 0))

    def get_chunk_shape(self, grid, name, shape, s=None):
        # The chunking strategy for spectrum grids should observe the following
        # - we often need only parts of the wavelength coverage
        # - interpolation algorithms iterate over the wavelengths in the outer loop
        # - interpolation algorithms need nearby models, cubic splines require models
        #   in memory along the entire interpolation axis

        # The shape of the spectrum grid is (param1, param2, wave)
        if name in grid.values and name in ['flux', 'cont', 'line']:
            if self.chunking == 'granular':
                newshape = []
                # Keep neighboring 3 models together in every direction
                for i, k, ax in grid.enumerate_axes():
                    if k in ['log_g', 'Fe_H', 'M_H', 'T_eff']:
                        newshape.append(min(shape[i], 3))
                    else:
                        newshape.append(1)
                # Use small chunks along the wavelength direction
                newshape.append(min(256, shape[-1]))
                return tuple(newshape)
            elif self.chunking == 'spectrum':
                # Store entire spectra together
                newshape = [1] * (len(shape) - 1) + [shape[-1]]
                return tuple(newshape)

        return None

    def get_compression(self, grid, name, shape, s=None):
        if name in grid.values and name in ['flux', 'cont', 'line']:
            return self.compression
        
        return None

