import os
import logging
import itertools
import numpy as np
from random import choice
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.interpolate import interp1d, interpn

from pfsspec.core import PfsObject
from pfsspec.core.grid import ArrayGrid
from pfsspec.core.grid import RbfGrid
from pfsspec.core.grid import PcaGrid

class ModelGrid(PfsObject):
    """Wraps an array or RBF grid, optionally PCA-compressed, and implements logic
    to interpolate spectra."""

    PREFIX_MODELGRID = 'modelgrid'

    def __init__(self, config, grid_type, orig=None):
        super(ModelGrid, self).__init__(orig=orig)

        if not isinstance(orig, ModelGrid):
            self.config = config
            self.grid = self.create_grid(grid_type)
            self.continuum_model = None
            self.wave = None
            self.is_wave_regular = None
            self.is_wave_lin = None
            self.is_wave_log = None
            self.resolution = None

            self.wave_lim = None
            self.wave_slice = None      # Limits wavelength range when reading the grid
        else:
            self.config = config if config is not None else orig.config
            self.grid = self.create_grid(grid_type) if grid_type is not None else orig.grid
            self.continuum_model = orig.continuum_model
            self.wave = orig.wave
            self.is_wave_regular = orig.is_wave_regular
            self.is_wave_lin = orig.is_wave_lin
            self.is_wave_log = orig.is_wave_log
            self.resolution = orig.resolution

            self.wave_lim = orig.wave_lim
            self.wave_slice = orig.wave_slice

    @property
    def preload_arrays(self):
        return self.grid.preload_arrays

    @preload_arrays.setter
    def preload_arrays(self, value):
        self.grid.preload_arrays = value

    @property
    def array_grid(self):
        return self.grid.array_grid

    @property
    def rbf_grid(self):
        return self.grid.rbf_grid

    def create_grid(self, grid_type):
        grid = grid_type(config=self.config)

        # Wrap into a PCA grid
        if self.config.pca is not None and self.config.pca:
            grid = PcaGrid(grid)
          
        self.config.init_axes(grid)
        self.config.init_values(grid)

        return grid

    def add_args(self, parser):
        # TODO: it collides with --lambda from the spectrum reader classes
        #       make sure this parameter is registered when building datasets
        
        parser.add_argument('--lambda', type=float, nargs='*', default=None, help='Limit on lambda.')
        self.grid.add_args(parser)

    def init_from_args(self, args):
        self.wave_lim = self.get_arg('lambda', self.wave_lim, args)
        self.grid.init_from_args(args)

    def get_wave_slice(self):
        if self.wave is None:
            raise Exception("Cannot determine slice without an initialized wave vector.")

        if self.wave_lim is None:
            return slice(None)
        elif self.wave_slice is None:
            if len(self.wave_lim) == 2:
                idx = np.digitize([self.wave_lim[0], self.wave_lim[1]], self.wave)
                self.wave_slice = slice(max(0, idx[0] - 1), idx[1], None)
            elif len(self.wave_lim) == 1:
                idx = np.digitize([self.wave_lim[0]], self.wave)
                self.wave_slice = max(0, idx[0] - 1)
            else:
                raise Exception('Only two or one values are allowed for parameter lambda.')

        return self.wave_slice

    def get_constants(self):
        return self.grid.get_constants()

    def set_constants(self, constants):
        self.grid.set_constants(constants)

    def get_axes(self, squeeze=False):
        return self.grid.get_axes(squeeze=squeeze)

    def set_axes(self, axes):
        self.grid.set_axes(axes)

    def build_axis_indexes(self):
        self.grid.build_axis_indexes()

    def get_shape(self):
        return self.grid.get_shape()

    def allocate_values(self):
        self.config.allocate_values(self.grid, self.wave)
        if self.continuum_model is not None:
            self.continuum_model.allocate_values(self.grid)

    def set_continuum_model(self, continuum_model):
        self.continuum_model = continuum_model
        self.continuum_model.init_values(self.grid)

    def load(self, filename, s=None, format=None):
        super(ModelGrid, self).load(filename=filename, s=s, format=format)
        if self.config.normalized:
            self.continuum_model = self.config.create_continuum_model()
            self.continuum_model.init_wave(self.get_wave())
            self.continuum_model.init_values(self.grid)
        self.grid.load(filename, s=s, format=format)
        
    def save_items(self):
        self.grid.filename = self.filename
        self.grid.fileformat = self.fileformat
        self.grid.save_items()

        self.save_params()
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'wave')), self.wave)

    def save_params(self):
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'type')), type(self).__name__)
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'config')), type(self.config).__name__)
        if self.continuum_model is not None:
            self.save_item('/'.join((self.PREFIX_MODELGRID, 'continuum_model')), self.continuum_model.name)
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'is_wave_regular')), self.is_wave_regular)
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'is_wave_lin')), self.is_wave_lin)
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'is_wave_log')), self.is_wave_log)

    def load_items(self, s=None):
        self.wave = self.load_item('/'.join((self.PREFIX_MODELGRID, 'wave')), np.ndarray)

        self.load_params()

        name = self.load_item('/'.join((self.PREFIX_MODELGRID, 'continuum_model')), str)
        if name is not None:
            self.config.continuum_model_type = self.config.CONTINUUM_MODEL_TYPES[name]
            self.continuum_model = self.config.create_continuum_model()
            self.continuum_model.init_wave(self.wave)
            self.continuum_model.init_values(self.grid)

    def load_params(self):
        t = self.load_item('/'.join((self.PREFIX_MODELGRID, 'type')), str)
        if t != type(self).__name__:
            raise Exception("Grid type `{}` doesn't match type `{}` in data file.".format(type(self).__name__, t))

        t = self.load_item('/'.join((self.PREFIX_MODELGRID, 'config')), str)
        if t != type(self.config).__name__:
            raise Exception("Grid config type `{}` doesn't match config type `{}` in data file.".format(type(self.config).__name__, t))

        self.is_wave_regular = self.load_item('/'.join((self.PREFIX_MODELGRID, 'is_wave_regular')), bool)
        self.is_wave_lin = self.load_item('/'.join((self.PREFIX_MODELGRID, 'is_wave_lin')), bool)
        self.is_wave_log = self.load_item('/'.join((self.PREFIX_MODELGRID, 'is_wave_log')), bool)

    def get_nearest_index(self, **kwargs):
        return self.grid.get_nearest_index(**kwargs)

    def get_index(self, **kwargs):
        return self.grid.get_index(**kwargs)

    def get_wave(self):
        return self.wave[self.get_wave_slice() or slice(None)]

    def set_wave(self, wave):
        self.wave = wave

    def set_flux(self, flux, cont=None, **kwargs):
        self.grid.set_flux(flux, cont=cont, **kwargs)

    def set_flux_at(self, index, flux, cont=None):
        self.grid.set_flux_at(index, flux, cont=cont)

    def is_value_valid(self, name, value):
        raise NotImplementedError()
        return self.grid.is_value_valid(name, value)

    def get_value_sliced(self, name):
        if isinstance(self.grid, ArrayGrid):
            if self.grid.slice is not None:
                if name in ['flux', 'cont']:
                    # Slice in the wavelength direction as well
                    return self.grid.get_value_at(name, idx=self.grid.slice, s=self.get_wave_slice())
                else:
                    # Slice only in the stellar parameter directions
                    return self.grid.get_value_at(name, idx=self.grid.slice)
            else:
                return self.get_value_at(name, idx=None)
        else:
            raise NotImplementedError()

    def get_model_count(self, use_limits=False):
        return self.grid.get_valid_value_count('flux')

    def get_flux_shape(self):
        return self.get_shape() + self.wave[self.get_wave_slice() or slice(None)].shape

    def set_flux(self, flux, cont=None, **kwargs):
        """
        Sets the flux at a given point of the grid

        Parameters must exactly match grid coordinates.
        """
        self.set_value('flux', flux, **kwargs)
        if cont is not None:
            self.set_value('cont', cont, **kwargs)

    def set_flux_at(self, index, flux, cont=None):
        self.grid.set_value_at('flux', index, flux)
        if cont is not None:
            self.grid.set_value_at('cont', index, cont)

    def get_parameterized_spectrum(self, idx=None, s=None, **kwargs):
        spec = self.config.create_spectrum()
        self.grid.set_object_params(spec, idx=idx, **kwargs)
        spec.wave = self.get_wave()
        return spec

    def get_continuum_parameters(self, **kwargs):
        names = [p.name for p in self.continuum_model.get_model_parameters()]
        params = self.grid.get_values(names=names, **kwargs)
        return params

    def get_continuum_parameters_at(self, idx):
        names = [p.name for p in self.continuum_model.get_model_parameters()]
        params = self.grid.get_values_at(idx, names=names)
        return params

    def get_model(self, denormalize=False, **kwargs):
        spec = self.get_parameterized_spectrum(s=self.get_wave_slice(), **kwargs)
        spec.flux = np.array(self.grid.get_value('flux', s=self.get_wave_slice(), **kwargs), copy=True)
        if self.grid.has_value('cont'):
            spec.cont = np.array(self.grid.get_value('cont', s=self.get_wave_slice(), **kwargs), copy=True)

        if denormalize and self.continuum_model is not None:
            # Get the continuum parameters. This means interpolation,
            # in case the parameters are given with an RBF. Also allow
            # skipping parameters for those continuum models which
            # are calculated from the grid parameters (i.e. Planck)
            params = self.get_continuum_parameters(**kwargs)
            self.continuum_model.denormalize(spec, params)

        return spec

    def get_model_at(self, idx, denormalize=False):
        if self.grid.has_value_at('flux', idx):
            spec = self.get_parameterized_spectrum(idx, s=self.get_wave_slice())
            spec.flux = np.array(self.grid.get_value_at('flux', idx, s=self.get_wave_slice()), copy=True)
            if self.grid.has_value('cont'):
                spec.cont = np.array(self.grid.get_value_at('cont', idx, s=self.get_wave_slice()), copy=True)

            if denormalize and self.continuum_model is not None:
                params = self.get_continuum_parameters_at(idx)
                self.continuum_model.denormalize(spec, params)
            
            return spec
        else:
            return None

    def get_nearest_model(self, denormalize=False, **kwargs):
        """
        Finds grid point closest to the parameters specified
        """
        idx = self.grid.get_nearest_index(**kwargs)
        spec = self.get_model_at(idx, denormalize=denormalize)
        return spec

    def interpolate_model(self, interpolation=None, **kwargs):
        raise NotImplementedError()

    def interpolate_model_linear(self, **kwargs):
        r = self.grid.interpolate_value_linear('flux', **kwargs)
        if r is None:
            return None
        flux, kwargs = r

        if flux is not None:
            spec = self.get_parameterized_spectrum(**kwargs)
            spec.flux = flux
            if self.grid.has_value('cont'):
                spec.cont = self.grid.interpolate_value_linear('cont', **kwargs)
            return spec
        else:
            return None

    def interpolate_model_spline(self, free_param, **kwargs):
        r = self.grid.interpolate_value_spline('flux', free_param, **kwargs)
        if r is None:
            return None
        flux, bestargs = r

        if flux is not None:
            spec = self.get_parameterized_spectrum(**bestargs)
            spec.interp_param = free_param
            spec.flux = flux
            if self.grid.has_value('cont'):
                spec.cont, _ = self.grid.interpolate_value_spline('cont', free_param, **kwargs)
            return spec
        else:
            return None

    def interpolate_model_rbf(self, **kwargs):
        if isinstance(self.grid, RbfGrid) or \
           isinstance(self.grid, PcaGrid) and isinstance(self.grid.grid, RbfGrid):
                return self.get_nearest_model(**kwargs)
        else:
            raise Exception("Operation not supported.")
   