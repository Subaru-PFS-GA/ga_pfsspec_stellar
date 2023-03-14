import os
import logging
import itertools
import importlib
import numpy as np
import h5py
from random import choice
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.interpolate import interp1d, interpn

from pfs.ga.pfsspec.core import PfsObject
from pfs.ga.pfsspec.core.util.array import *
from pfs.ga.pfsspec.core.grid import ArrayGrid, RbfGrid, PcaGrid

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

    @classmethod
    def from_file(cls, filename, preload_arrays=False, args=None):
        """
        Initializes a model grid from an HDF5 file by figuring out what configuration
        and grid type to be used. This includes the config class, PCA and RBF.
        """

        logging.info(f'Inferring model grid config and type from HDF5 file {filename}.')

        # Peek into the HDF5 file
        with h5py.File(filename, 'r') as f:
            grid_type = f[ArrayGrid.PREFIX_GRID].attrs['type']
            modelgrid_config = f[cls.PREFIX_MODELGRID].attrs['config']
            modelgrid_type = f[cls.PREFIX_MODELGRID].attrs['type']

            # This is an PCA grid if /grid/arrays/flux/pca exists
            try:
                _ = f['grid']['arrays']['flux']['pca']
                is_pca = True
            except KeyError:
                is_pca = False

        # These classes are imported into the namespace
        grid_type = globals()[grid_type]
        modelgrid_type = globals()[modelgrid_type]

        # The grid config should be loaded manually
        module = importlib.import_module(f'.{modelgrid_config.lower()}', 'pfs.ga.pfsspec.stellar.grid')
        modelgrid_config_type = getattr(module, modelgrid_config)

        logging.info(f'Inferred model grid type {modelgrid_type}({modelgrid_config}), pca={is_pca}')

        # Instantiate the class
        grid = modelgrid_type(modelgrid_config_type(pca=is_pca), grid_type)
        grid.preload_arrays = preload_arrays
        if args is not None:
            grid.init_from_args(args)
        grid.load(filename, format='h5')

        return grid


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

    @property
    def pca_grid(self):
        return self.grid.pca_grid

    def create_grid(self, grid_type):
        grid = grid_type(config=self.config)

        # Wrap into a PCA grid
        if self.config.pca is not None and self.config.pca:
            grid = PcaGrid(grid)
          
        self.config.init_axes(grid)
        self.config.init_values(grid)

        return grid

    def add_args(self, parser):
        parser.add_argument('--wave-lim', type=float, nargs='*', default=None, help='Limit on lambda.')
        self.grid.add_args(parser)

    def init_from_args(self, args):
        self.wave_lim = self.get_arg('wave_lim', self.wave_lim, args)
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
        constants = self.grid.get_constants()
        if self.continuum_model is not None:
            constants.update(self.continuum_model.get_constants(self.get_wave()))
        return constants

    def set_constants(self, constants):
        self.grid.set_constants(constants)

    def get_slice(self):
        return self.grid.get_slice()

    def set_axes(self, axes):
        self.grid.set_axes(axes)

    def get_axis(self, key):
        return self.grid.get_axis(key)

    def enumerate_axes(self, s=None, squeeze=False):
        return self.grid.enumerate_axes(s=s, squeeze=squeeze)

    def build_axis_indexes(self):
        self.grid.build_axis_indexes()

    def get_shape(self, s=None, squeeze=False):
        return self.grid.get_shape(s=s, squeeze=squeeze)

    def allocate_values(self):
        self.config.allocate_values(self.grid, self.wave)
        if self.continuum_model is not None:
            self.continuum_model.allocate_values(self.grid)

    def set_continuum_model(self, continuum_model):
        self.continuum_model = continuum_model
        self.continuum_model.init_values(self.grid)

    def load(self, filename, s=None, format=None):
        super(ModelGrid, self).load(filename=filename, s=s, format=format)

        # We need to initialize the continuum model here because calling
        # `init_values` will configure the underlying grid to load the fitted
        # parameter values. We use all defauls here because the actual continuum
        # model will be loaded at a later step.
        if self.config.normalized:
            self.continuum_model = self.config.create_continuum_model()
            self.continuum_model.init_wave(self.get_wave())
            self.continuum_model.init_values(self.grid)

        self.grid.load(filename, s=s, format=format)

    def save_items(self):
        self.grid.filename = self.filename
        self.grid.fileformat = self.fileformat
        self.grid.save_items()

        if self.continuum_model is not None:
            self.continuum_model.filename = self.filename
            self.continuum_model.fileformat = self.fileformat
            self.continuum_model.save_items()

        self.save_params()
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'wave')), self.wave)

    def save_params(self):
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'type')), type(self).__name__)
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'config')), type(self.config).__name__)
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'config_pca')), self.config.pca)
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'config_rbf')), self.rbf_grid is not None)
        
        if self.continuum_model is not None:
            self.save_item('/'.join((self.PREFIX_MODELGRID, 'continuum_model')), self.continuum_model.name)
        
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'is_wave_regular')), self.is_wave_regular)
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'is_wave_lin')), self.is_wave_lin)
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'is_wave_log')), self.is_wave_log)

    def load_items(self, s=None):
        # We need to load the underlying grid here because continuum_model.init_values will
        # update the list of value arrays so we need the axis values
        self.grid.filename = self.filename
        self.grid.fileformat = self.fileformat
        self.grid.load_items()

        self.load_params()
        self.wave = self.load_item('/'.join((self.PREFIX_MODELGRID, 'wave')), np.ndarray)

        name = self.load_item('/'.join((self.PREFIX_MODELGRID, 'continuum_model')), str)
        if name is not None:
            self.config.continuum_model_type = self.config.CONTINUUM_MODEL_TYPES[name]
            self.continuum_model = self.config.create_continuum_model()
            self.continuum_model.init_wave(self.get_wave())
            
            self.continuum_model.filename = self.filename
            self.continuum_model.fileformat = self.fileformat
            self.continuum_model.load_items()

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
                    return self.grid.get_value_at(name, idx=self.grid.get_slice(), s=self.get_wave_slice())
                else:
                    # Slice only in the stellar parameter directions
                    return self.grid.get_value_at(name, idx=self.grid.get_slice())
            else:
                return self.get_value_at(name, idx=None)
        else:
            raise NotImplementedError()

    def get_model_count(self):
        """
        Returns the number of valid models, i.e. where the flux array
        has a valid value bases on the value index.
        """
        return self.grid.get_valid_value_count('flux', s=self.grid.get_slice())

    def get_flux_shape(self):
        return self.get_shape(s=self.get_slice(), squeeze=False) + self.wave[self.get_wave_slice() or slice(None)].shape

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
        spec.resolution = self.resolution
        return spec

    def get_continuum_parameters(self, **kwargs):
        names = [p.name for p in self.continuum_model.get_interpolated_params()]
        params = self.grid.get_values(names=names, **kwargs)
        return params

    def get_continuum_parameters_at(self, idx):
        names = [p.name for p in self.continuum_model.get_interpolated_params()]
        params = self.grid.get_values_at(idx, names=names)
        return params

    def get_model(self, denormalize=True, **kwargs):
        flux = self.grid.get_value('flux', s=self.get_wave_slice(), **kwargs)
        
        # In case the interpolation didn't work
        if flux is not None:
            s = self.get_wave_slice()

            spec = self.get_parameterized_spectrum(s=self, **kwargs)
            spec.flux = copy_array(flux)

            if self.grid.has_error('flux'):
                flux_err = self.grid.get_error('flux', s=s, **kwargs)
                spec.flux_err = copy_array(flux_err)
            
            if self.grid.has_value('cont'):
                cont = self.grid.get_value('cont', s=s, **kwargs)
                spec.cont = copy_array(cont)

            if denormalize and self.continuum_model is not None:
                # Get the continuum parameters. This means interpolation,
                # in case the parameters are given with an RBF. Also allow
                # skipping parameters for those continuum models which
                # are calculated from the grid parameters (i.e. Planck)
                params = self.get_continuum_parameters(**kwargs)
                self.continuum_model.denormalize(spec, params, s=s)

            return spec
        else:
            return None

    def get_model_at(self, idx, denormalize=True):
        if self.grid.has_value_at('flux', idx):
            s = self.get_wave_slice()

            spec = self.get_parameterized_spectrum(idx, s=self.get_wave_slice())
            spec.flux = copy_array(self.grid.get_value_at('flux', idx, s=s))
            
            if self.grid.has_error('flux'):
                spec.flux_err = copy_array(self.grid.get_error_at('flux', idx, s=s))
            
            if self.grid.has_value('cont'):
                spec.cont = copy_array(self.grid.get_value_at('cont', idx, s=self.get_wave_slice()))

            if denormalize and self.continuum_model is not None:
                params = self.get_continuum_parameters_at(idx)
                self.continuum_model.denormalize(spec, params, s=s)
            
            return spec
        else:
            return None

    def get_nearest_model(self, denormalize=True, **kwargs):
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

            r = self.grid.interpolate_value_linear('cont', **kwargs)
            if r is not None:
                spec.cont, _ = r

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
   