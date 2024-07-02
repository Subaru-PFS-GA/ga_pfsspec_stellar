import importlib
import numpy as np
import h5py
from random import choice
from collections.abc import Iterable
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.interpolate import interp1d, interpn

from pfs.ga.pfsspec.core.setup_logger import logger
from pfs.ga.pfsspec.core import PfsObject
from pfs.ga.pfsspec.core import Physics
from pfs.ga.pfsspec.core.caching import ReadOnlyCache
from pfs.ga.pfsspec.core.util.array import *
from pfs.ga.pfsspec.core.grid import ArrayGrid, RbfGrid, PcaGrid

class ModelGrid(PfsObject):
    """
    Wraps an array or RBF grid, optionally PCA-compressed, and implements logic
    to interpolate synthetic stellar spectra.

    The model grid can be normalized using a parametric continuum model. The model
    constants and parameters are stored in the grid file and the latter can be
    interpolated over the grid to any combination of the stellar parameters.

    Variables
    ---------
    config : ModelGridConfig
        Model grid configuration. Used to initialize the grid axes, etc.
    grid : Grid
        Underlying grid implementation (Array, PCA, Rbf...)
    wave : np.ndarray
        Wavelength bin centers
    wave_edges : np.ndarray
        Wavelength bin edges
    is_wave_regular : bool
        True if the wavelength grid is regular
    is_wave_lin : bool
        True if the wavelength grid is linear
    is_wave_log : bool
        True if the wavelength grid is logarithmic
    is_wave_vacuum : bool
        True if the wavelength grid is in vacuum
    resolution : float
        Resolution of the spectra in R = lambda / dlambda
    wave_lim : list
        Wavelength limits then reading the grid
    wave_slice : slice
        Limits wavelength range when reading the grid
    continuum_model : ContinuumModel
        Continuum model used to normalize the spectra as stored in the grid.
        Can be used to denormalize flux to physical units.
    psf : PSF
        Convolution kernel, apply before returning the model.
    """

    PREFIX_MODELGRID = 'modelgrid'

    def __init__(self, config, grid_type, orig=None):
        """
        Initializes a new instance of ModelGrid. Use `from_file` instead to
        load a grid from an HDF5 file.

        Parameters
        ----------
        config : ModelGridConfig
            Model grid configuration
        grid_type : Grid
            Grid implementation (Array, PCA, Rbf...)
        orig : ModelGrid
            Original instance to copy from
        """

        super(ModelGrid, self).__init__(orig=orig)

        if not isinstance(orig, ModelGrid):
            self.config = config                        # Model grid configuration
            self.grid = self.create_grid(grid_type)     # Underlying grid implementation (Array, PCA, Rbf...)
            self.wave = None                            # Wavelength bin centers
            self.wave_edges = None                      # Wavelength bin edges
            self.is_wave_regular = None
            self.is_wave_lin = None
            self.is_wave_log = None
            self.is_wave_vacuum = None                  # When True, wavelength is in vacuum
            self.resolution = None

            self.wave_lim = None                        # Wavelength limits then reading the grid
            self.wave_slice = None                      # Limits wavelength range when reading the grid

            self.continuum_model = None                 # Continuum model
            self.psf = None                             # Convolution kernel, apply before returning the model
        else:
            self.config = config if config is not None else orig.config
            self.grid = self.create_grid(grid_type) if grid_type is not None else orig.grid
            self.wave = orig.wave
            self.wave_edges = orig.wave_edges
            self.is_wave_regular = orig.is_wave_regular
            self.is_wave_lin = orig.is_wave_lin
            self.is_wave_log = orig.is_wave_log
            self.is_wave_vacuum = orig.is_wave_vacuum
            self.resolution = orig.resolution

            self.wave_lim = orig.wave_lim
            self.wave_slice = orig.wave_slice

            self.continuum_model = orig.continuum_model
            self.psf = orig.psf

    @classmethod
    def from_file(cls, filename, preload_arrays=False, mmap_arrays=False, cache_values=True, args=None, slice_from_args=True):
        """
        Initializes a model grid from an HDF5 file by figuring out what configuration
        and grid type to be used. This includes the config class, PCA and RBF.

        Parameters
        ----------
        filename : str
            HDF5 file name
        preload_arrays : bool
            Preload arrays into memory
        mmap_arrays : bool
            Memory map arrays, requires a contiguous storage model of arrays within HDF5.
        cache_values : bool
            Cache values for faster access
        args : dict
            Arguments to initialize the grid
        slice_from_args : bool
            Use arguments to slice the grid

        Returns
        -------
        ModelGrid
            Model grid instance
        """

        logger.debug(f'Inferring model grid config and type from HDF5 file {filename}.')

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

        logger.debug(f'Inferred model grid type {modelgrid_type}({modelgrid_config}, pca={is_pca})')

        # Instantiate the class
        grid = modelgrid_type(modelgrid_config_type(pca=is_pca), grid_type)
        grid.preload_arrays = preload_arrays
        grid.mmap_arrays = mmap_arrays
        if cache_values and grid.array_grid is not None:
            grid.array_grid.value_cache = ReadOnlyCache()
        if args is not None:
            grid.init_from_args(args, slice_from_args=slice_from_args)
        grid.load(filename, format='h5')

        return grid

    @property
    def preload_arrays(self):
        return self.grid.preload_arrays

    @preload_arrays.setter
    def preload_arrays(self, value):
        self.grid.preload_arrays = value

    @property
    def mmap_arrays(self):
        return self.grid.mmap_arrays
    
    @mmap_arrays.setter
    def mmap_arrays(self, value):
        self.grid.mmap_arrays = value

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
        """
        Creates a grid instance based on the configuration.
        """

        grid = grid_type(config=self.config)

        # Wrap into a PCA grid
        if self.config.pca is not None and self.config.pca:
            grid = PcaGrid(grid)
          
        self.config.init_axes(grid)
        self.config.init_values(grid)

        return grid

    def add_args(self, parser):
        """
        Registers command line arguments for the grid.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            Argument parser
        """

        parser.add_argument('--wave-lim', type=float, nargs='*', default=None, help='Limit on lambda.')
        self.grid.add_args(parser)

    def init_from_args(self, args, slice_from_args=True):
        """
        Initializes the grid from command line arguments.

        Parameters
        ----------
        args : dict
            Command line arguments
        slice_from_args : bool
            Use arguments to slice the grid along the axes
        """

        self.wave_lim = self.get_arg('wave_lim', self.wave_lim, args)
        self.grid.init_from_args(args, slice_from_args=slice_from_args)

    def get_wave_slice_impl(self, wlim):
        """
        Returns a slice for the wavelength vector based on the limits.

        Parameters
        ----------
        wlim : list
            Wavelength limits

        Returns
        -------
        slice
            Slice for the wavelength vector
        """

        if self.wave is None:
            raise Exception("Cannot determine slice without an initialized wave vector.")
        
        idx = np.digitize(wlim, self.wave)
        return slice(max(0, idx[0] - 1), idx[1], None)

    def get_wave_slice(self):
        """
        Returns a slice for the wavelength vector based on the limits.

        Returns
        -------
        slice
            Slice for the wavelength vector
        """

        if self.wave is None:
            raise Exception("Cannot determine slice without an initialized wave vector.")

        if self.wave_lim is None:
            return slice(None)
        elif self.wave_slice is None:
            if len(self.wave_lim) == 2:
                # Two values mean from and to wavelength
                self.wave_slice = self.get_wave_slice_impl(self.wave_lim)
            elif len(self.wave_lim) == 1:
                # A single value means a single wavelength bin
                idx = np.digitize([self.wave_lim[0]], self.wave)
                self.wave_slice = max(0, idx[0] - 1)
            else:
                raise Exception('Only one or true values are allowed for the variable `wave_lim`.')

        return self.wave_slice

    def get_constants(self):
        """
        Returns the constants of the continuum model, if any. These are stored in the grid file.

        Returns
        -------
        dict of ndarray
            Continuum model constants
        """

        constants = self.grid.get_constants()
        if self.continuum_model is not None:
            wave, _, _ = self.get_wave()
            constants.update(self.continuum_model.get_constants(wave=wave))
        return constants

    def set_constants(self, constants):
        """
        Sets the constants of the continuum model. These are stored in the grid file.

        Parameters
        ----------
        constants : dict of ndarray
            Continuum model constants
        """
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
        """
        Allocates the value arrays in the data file.

        This is called after the grid is initialized and the axes are set.
        """
        self.config.allocate_values(self.grid, self.wave, wave_edges=self.wave_edges)
        if self.continuum_model is not None:
            self.continuum_model.allocate_values(self.grid)

    def set_continuum_model(self, continuum_model):
        """
        Sets the continuum model to be used for normalization. When the continuum model
        is set, the grid is updated and the arrays for the continuum model are created.
        """
        self.continuum_model = continuum_model
        self.continuum_model.init_values(self.grid)

    def load(self, filename, s=None, format=None):
        """
        Loads the grid from an HDF5 file. This doesn't neccessarily mean loading the
        value arrays into memory.

        Parameters
        ----------
        filename : str
            HDF5 file name
        s : slice
            Slice of the grid to load, when loading the value arrays into memory.
        format : str
            File format. Use 'h5' for HDF5 files.
        """

        super().load(filename=filename, s=s, format=format)

        # We need to initialize the continuum model here because calling
        # `init_values` will configure the underlying grid to load the fitted
        # parameter values. We use all defauls here because the actual continuum
        # model will be loaded at a later step.
        if self.config.normalized:
            self.continuum_model = self.config.create_continuum_model()
            self.continuum_model.init_wave(self.get_wave()[0])
            self.continuum_model.init_values(self.grid)

        self.grid.load(filename, s=s, format=format)

    def save_items(self):
        """
        Save the grid items to the HDF5 file. This includes the grid axes,
        continuum model constants etc. but not the value arrays themselves.
        """

        self.grid.filename = self.filename
        self.grid.fileformat = self.fileformat
        self.grid.save_items()

        if self.continuum_model is not None:
            self.continuum_model.filename = self.filename
            self.continuum_model.fileformat = self.fileformat
            self.continuum_model.save_items()

        self.save_params()
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'wave')), self.wave)
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'wave_edges')), self.wave_edges)

    def save_params(self):
        """
        Save the grid parameters to the HDF5 file. This includes the grid type,
        config type, PCA and RBF flags etc.
        """

        self.save_item('/'.join((self.PREFIX_MODELGRID, 'type')), type(self).__name__)
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'config')), type(self.config).__name__)
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'config_pca')), self.config.pca)
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'config_rbf')), self.rbf_grid is not None)
        
        if self.continuum_model is not None:
            self.save_item('/'.join((self.PREFIX_MODELGRID, 'continuum_model')), self.continuum_model.name)
        
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'is_wave_regular')), self.is_wave_regular)
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'is_wave_lin')), self.is_wave_lin)
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'is_wave_log')), self.is_wave_log)
        self.save_item('/'.join((self.PREFIX_MODELGRID, 'is_wave_vacuum')), self.is_wave_vacuum)

    def load_items(self, s=None):
        """
        Load the grid items from the HDF5 file. This includes the grid axes, continuum
        model constants etc. but not the value arrays themselves.

        Parameters
        ----------
        s : slice
            Slice of the grid to load, when loading the value arrays into memory.
        """

        # We need to load the underlying grid here because continuum_model.init_values will
        # update the list of value arrays so we need the axis values
        self.grid.filename = self.filename
        self.grid.fileformat = self.fileformat
        self.grid.load_items()

        self.load_params()
        self.wave = self.load_item('/'.join((self.PREFIX_MODELGRID, 'wave')), np.ndarray)
        self.wave_edges = self.load_item('/'.join((self.PREFIX_MODELGRID, 'wave_edges')), np.ndarray)

        name = self.load_item('/'.join((self.PREFIX_MODELGRID, 'continuum_model')), str)
        if name is not None:
            wave = self.get_wave()[0]
            self.config.continuum_model_type = self.config.CONTINUUM_MODEL_TYPES[name]
            self.continuum_model = self.config.create_continuum_model()
            self.continuum_model.init_wave(wave)
            
            self.continuum_model.filename = self.filename
            self.continuum_model.fileformat = self.fileformat
            self.continuum_model.load_items()

            self.continuum_model.init_values(self.grid)

    def load_params(self):
        """
        Load the grid parameters from the HDF5 file. This includes the grid type, config
        type, PCA and RBF flags etc.
        """

        t = self.load_item('/'.join((self.PREFIX_MODELGRID, 'type')), str)
        if t != type(self).__name__:
            raise Exception("Grid type `{}` doesn't match type `{}` in data file.".format(type(self).__name__, t))

        t = self.load_item('/'.join((self.PREFIX_MODELGRID, 'config')), str)
        if t != type(self.config).__name__:
            raise Exception("Grid config type `{}` doesn't match config type `{}` in data file.".format(type(self.config).__name__, t))

        self.is_wave_regular = self.load_item('/'.join((self.PREFIX_MODELGRID, 'is_wave_regular')), bool)
        self.is_wave_lin = self.load_item('/'.join((self.PREFIX_MODELGRID, 'is_wave_lin')), bool)
        self.is_wave_log = self.load_item('/'.join((self.PREFIX_MODELGRID, 'is_wave_log')), bool)
        self.is_wave_vacuum = self.load_item('/'.join((self.PREFIX_MODELGRID, 'is_wave_vacuum')), bool)

    def get_nearest_index(self, **kwargs):
        """
        Find the nearest grid index to the parameters specified in `kwargs`.

        The indexes returned are always within the grid bounds and are integer values that
        can be used to index into the value arrays.

        Parameters
        ----------
        kwargs : dict
            Stellar parameters specific to the grid.

        Returns
        -------
        tuple
            Grid index
        """

        return self.grid.get_nearest_index(**kwargs)

    def get_index(self, **kwargs):
        """
        Find the grid index corresponding to the parameters specified in `kwargs`.

        The indexes returned are always within the grid bounds and can be float values
        when the stellar parameters fall between grid points. These indexes cannot be
        used to index into the value arrays directly.

        Parameters
        ----------
        kwargs : dict
            Stellar parameters specific to the grid.

        Returns
        -------
        tuple
            Grid index
        """
        
        return self.grid.get_index(**kwargs)

    def get_wave(self, wave_vacuum=True, s=None):
        """
        Gets the wavelength vector and optionally the edges and the wavelength mask.

        Parameters
        ----------
        wave_vacuum : bool
            Request wavelength in wave_vacuum
        s : slice
            Slice of the wavelength vector
        """

        # If no slice is provided, use the default slice
        if s is None:
            s = self.get_wave_slice() or slice(None)
        
        wave = self.wave[s]
        
        if self.wave_edges is not None:
            if s != slice(None):
                # TODO: Implement slicing of wave edges
                wave_edges = None
            else:
                wave_edges = self.wave_edges
        else:
            wave_edges = None

        # Convert wavelength to vacuum. Wavelength is condired in vacuum by default and
        # conversion happend only if the wavelength is specifically defined in air.
        if wave_vacuum and self.is_wave_vacuum is not None and not self.is_wave_vacuum:
            # Convert wavelength to vacuum
            wave = Physics.air_to_vac(wave)

        return wave, wave_edges, None

    def set_wave(self, wave, wave_edges=None):
        """
        Sets the wavelength vector and optionally the edges.

        Parameters
        ----------
        wave : np.ndarray
            Wavelength bin centers
        wave_edges : np.ndarray
            Wavelength bin edges
        """

        self.wave = wave
        self.wave_edges = wave_edges

    def is_value_valid(self, name, value):
        raise NotImplementedError()
        return self.grid.is_value_valid(name, value)

    def get_value_sliced(self, name):
        """
        Returns the value array sliced along the stellar parameter axes. The slice
        is taken from the grid object and determined by the arguments passed to
        the grid object during load.

        Parameters
        ----------
        name : str
            Name of the value array
        """

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

        Returns
        -------
        int
            Number of valid models
        """

        return self.grid.get_valid_value_count('flux', s=self.grid.get_slice())

    def get_flux_shape(self):
        """
        Returns the shape of the flux array.

        The shape is determined by taking the slicing of the grid and the wavelength vector into account.

        Returns
        -------
        tuple
            Shape of the flux array
        """

        return self.get_shape(s=self.get_slice(), squeeze=False) + self.wave[self.get_wave_slice() or slice(None)].shape

    def set_flux(self, flux, cont=None, **kwargs):
        """
        Sets the flux at a given point of the grid. The location must be specified
        with the stellar parameters.

        Parameters
        ----------
        flux : np.ndarray
            Flux array
        cont : np.ndarray
            Continuum model array (optional)
        kwargs : dict
            Stellar parameters
        """

        self.set_value('flux', flux, **kwargs)
        if cont is not None:
            self.set_value('cont', cont, **kwargs)

    def set_flux_at(self, index, flux, cont=None):
        """
        Sets the flux at a given point of the grid. The location must be specified
        with the grid index.

        Parameters
        ----------
        index : tuple
            Grid index
        flux : np.ndarray
            Flux array
        cont : np.ndarray
            Continuum model array (optional)
        """

        self.grid.set_value_at('flux', index, flux)
        if cont is not None:
            self.grid.set_value_at('cont', index, cont)

    def get_parameterized_spectrum(self, idx=None, wlim=None, wave_vacuum=True, **kwargs):
        """
        Returns a spectrum object with all parameters set based on the grid definition
        and the parameters specified by `idx` or in `kwargs`.

        The returned object is a parametrized spectrum object that has the wave vector set
        but not the flux or continuum model.

        Parameters
        ----------
        idx : tuple
            Grid index
        wlim : slice
            Slice of the wavelength vector
        wave_vacuum : bool
            Request wavelength in wave_vacuum
        kwargs : dict
            Stellar parameters

        Returns
        -------
        Spectrum
            Parametrized spectrum object
        """

        spec = self.config.create_spectrum()
        self.grid.set_object_params(spec, idx=idx, **kwargs)
        # TODO: deal with mask
        spec.wave, spec.wave_edges, wave_mask = self.get_wave(s=wlim, wave_vacuum=wave_vacuum)
        spec.resolution = self.resolution

        # Process history
        spec.append_history(f'Spectrum object of `{type(spec).__name__}` created.')
        spec.append_history(f'Nearest model taken from `{type(self.grid).__name__}` with config: `{type(self.grid.config).__name__}`, filename: `{self.filename}`')
                
        if wlim is not None and wlim != slice(None):
            spec.append_history(f'Wavelength range limited by request to {spec.wave[0], spec.wave[-1]}.')
        else:
            spec.append_history(f'Wavelength range limited by grid coverage to {spec.wave[0], spec.wave[-1]}.')

        if spec.wave is not None:
            spec.append_history(f'Wave vector assigned from grid.')
        
        if spec.wave_edges is not None:
            spec.append_history(f'Wave_edges vector assigned from grid.')
        
        if spec.resolution is not None:
            spec.append_history(f'Resolution is assumed as R={spec.resolution} from grid.')
        else:
            spec.append_history(f'Resolution is not assumed from grid.')

        return spec

    def get_continuum_parameters(self, **kwargs):
        """
        Returns the continuum model parameters at the specified stellar parameters.
        These parameters are transparently interpolated to the stellar parameters.

        Parameters
        ----------
        kwargs : dict
            Stellar parameters

        Returns
        -------
        dict
            Continuum model parameters
        """

        names = [p.name for p in self.continuum_model.get_interpolated_params()]
        params = self.grid.get_values(names=names, **kwargs)
        return params

    def get_continuum_parameters_at(self, idx):
        """
        Returns the continuum model parameters at the specified grid index.

        Parameters
        ----------
        idx : tuple
            Grid index

        Returns
        -------
        dict
            Continuum model parameters
        """

        names = [p.name for p in self.continuum_model.get_interpolated_params()]
        params = self.grid.get_values_at(idx, names=names)
        return params
    
    #region Model accessor functions

    def get_model_post_process(self, wlim, psf=None):
        """
        Returns a function to be executed before interpolation. The results of the
        function are cached by the grid object and reused in further interpolations.

        This particular function convolves the model with a PSF.

        Parameters
        ----------
        wlim : slice
            Wavelength limits for the convolution. Specify this for model grids with
            a large range of wavelengths to avoid unnecessary convolution.
        psf : PSF
            Convolution kernel

        Returns
        -------
        tuple
            Cache key prefix
        function
            Post-process function
        """

        # Perform the kernel convolution in a post-process step. This way the results will
        # be cached by the ArrayGrid and reused in further interpolations

        psf = psf if psf is not None else self.psf
        if psf is not None:
            cache_key_prefix = (psf,)
            def post_process(value):
                # TODO: Use wave limits
                _, vv, _, shift = psf.convolve(self.wave[wlim], values=[ value ], errors=[], size=None, normalize=True)
                value[shift:-shift] = vv[0]
                return value
        else:
            cache_key_prefix = ()
            post_process = None

        return cache_key_prefix, post_process

    def get_model(self, denormalize=True, wlim=None, psf=None, **kwargs):
        """
        Returns the model nearest to the specified stellar parameters in case of an ArrayGrid
        and the interpolated model in case of an RbfGrid.

        Parameters
        ----------
        denormalize : bool
            Denormalize the flux to physical units using the continuum model, if any
        wlim : slice
            Wavelength limits for the returned spectrum
        psf : PSF
            Convolution kernel
        kwargs : dict
            Stellar parameters

        Returns
        -------
        Spectrum
            Model spectrum
        """

        if self.array_grid is not None:
            return self.get_nearest_model(denormalize=denormalize, wlim=wlim, psf=psf, **kwargs)
        elif self.rbf_grid is not None:
            return self.interpolate_model_rbf(denormalize=denormalize, wlim=wlim, psf=psf, **kwargs)
        else:
            raise NotImplementedError()

    def get_model_at(self, idx, denormalize=True, wlim=None, psf=None):
        """
        Return the model at exact grid coordinates.

        Parameters
        ----------
        idx : tuple
            Grid index
        denormalize : bool
            Denormalize the flux to physical units using the continuum model, if any
        wlim : slice
            Wavelength limits for the returned spectrum
        psf : PSF
            Convolution kernel

        Returns
        -------
        Spectrum
            Model spectrum
        """

        def interp_fun(name, post_process, cache_key_prefix, wlim, idx, **kwargs):
            params = self.grid.get_params_at(idx, **kwargs)
            return self.grid.get_value_at(name, idx, post_process=post_process, cache_key_prefix=cache_key_prefix, s=wlim), params
        
        return self.interpolate_model_impl('grid', interp_fun, denormalize=denormalize, wlim=wlim, psf=psf, idx=idx)
    
    def interpolate_model(self, denormalize=True, interpolation=None, wlim=None, psf=None, **kwargs):
        """
        Interpolates the model to the specified stellar parameters.

        Parameters
        ----------
        denormalize : bool
            Denormalize the flux to physical units using the continuum model, if any
        interpolation : str
            Interpolation method: 'linear', 'spline', 'rbf', depending on what the grid supports
        wlim : slice
            Wavelength limits for the returned spectrum
        psf : PSF
            Convolution kernel
        kwargs : dict
            Stellar parameters

        Returns
        -------
        Spectrum
            Model spectrum
        """

        if self.array_grid is not None:
            return self.interpolate_model_linear(denormalize=denormalize, wlim=wlim, psf=psf, **kwargs)
        elif self.rbf_grid is not None:
            return self.interpolate_model_rbf(denormalize=denormalize, wlim=wlim, psf=psf, **kwargs)
        else:
            return NotImplementedError()

    def get_nearest_model(self, denormalize=True, wlim=None, psf=None, **kwargs):
        """
        Returns the stellar model on a grid point closest to the parameters specified.

        Parameters
        ----------
        denormalize : bool
            Denormalize the flux to physical units using the continuum model, if any
        wlim : slice
            Wavelength limits for the returned spectrum
        psf : PSF
            Convolution kernel
        kwargs : dict
            Stellar parameters

        Returns
        -------
        Spectrum
            Model spectrum
        """

        def interp_fun(name, post_process, cache_key_prefix, wlim, idx, **kwargs):
            idx = self.grid.get_nearest_index(**kwargs)
            params = self.grid.get_params_at(idx, **kwargs)
            return self.grid.get_value_at(name, idx, post_process=post_process, cache_key_prefix=cache_key_prefix, s=wlim), params
        
        return self.interpolate_model_impl('nearest', interp_fun, denormalize=denormalize, wlim=wlim, psf=psf, **kwargs)
        
    def interpolate_model_linear(self, denormalize=True, wlim=None, psf=None, **kwargs):
        """
        Interpolates the model using multivariate linear interpolation to the given
        stellar parameters.

        Parameters
        ----------
        denormalize : bool
            Denormalize the flux to physical units using the continuum model, if any
        wlim : slice
            Wavelength limits for the returned spectrum
        psf : PSF
            Convolution kernel
        kwargs : dict
            Stellar parameters

        Returns
        -------
        Spectrum
            Model spectrum        
        """

        def interp_fun(name, post_process, cache_key_prefix, wlim, idx, **kwargs):
            return self.grid.interpolate_value_linear(name, post_process=post_process, cache_key_prefix=cache_key_prefix, s=wlim, **kwargs)

        return self.interpolate_model_impl('linear', interp_fun, denormalize=denormalize, wlim=wlim, psf=psf, **kwargs)

    def interpolate_model_spline(self, free_param, denormalize=True, wlim=None, psf=None, **kwargs):
        """
        Interpolates the model using 1D cubic spline interpolation to the given stellar parameters.

        Parameters
        ----------
        free_param : str
            Name of the free parameter in which direction the interpolation happens. The rest
            of the parameters will be fixed at their nearest grid values.
        denormalize : bool
            Denormalize the flux to physical units using the continuum model, if any
        wlim : slice
            Wavelength limits for the returned spectrum
        psf : PSF
            Convolution kernel
        kwargs : dict
            Stellar parameters

        Returns
        -------
        Spectrum
            Model spectrum
        """

        def interp_fun(name, post_process, cache_key_prefix, wlim, idx, **kwargs):
            return self.grid.interpolate_value_spline(name, free_param, post_process=post_process, cache_key_prefix=cache_key_prefix, s=wlim, **kwargs)
        
        spec = self.interpolate_model_impl('spline', interp_fun, denormalize=denormalize, wlim=wlim, psf=psf, **kwargs)
        spec.interp_param = free_param
        return spec
    
    def interpolate_model_rbf(self, denormalize=True, wlim=None, psf=None, **kwargs):
        """
        Interpolates the model using RFB, when available.

        Parameters
        ----------
        denormalize : bool
            Denormalize the flux to physical units using the continuum model, if any
        wlim : slice
            Wavelength limits for the returned spectrum
        psf : PSF
            Convolution kernel
        kwargs : dict
            Stellar parameters

        Returns
        -------
        Spectrum
            Model spectrum  
        """

        def interp_fun(name, post_process, cache_key_prefix, wlim, idx, **kwargs):
            return self.grid.interpolate_value_rbf(name, post_process=post_process, cache_key_prefix=cache_key_prefix, s=wlim, **kwargs)

        if isinstance(self.grid, RbfGrid) or \
           isinstance(self.grid, PcaGrid) and isinstance(self.grid.grid, RbfGrid):
           
            return self.interpolate_model_impl('rbf', interp_fun, denormalize=denormalize, wlim=wlim, psf=psf, **kwargs)
        else:
            raise Exception("Operation not supported.")
            
    def interpolate_model_impl(self, method, interp_fun, denormalize=True, wlim=None, psf=None, idx=None, **kwargs):
        """
        Generic model interpolation using any interpolation function supported by the
        underlying grid implementation. Optionally convolve with a kernel.
        Optional convolution is passed on to the grid class so results can be cached.

        Either the grid index `idx` or the stellar parameters `kwargs` must be specified but not both at the
        same time.

        Parameters
        ----------
        method : str
            Interpolation method. Supported methods are 'grid', 'nearest', 'linear', 'spline', 'rbf'.
        interp_fun : function
            Interpolation function to be called on the grid object.
        denormalize : bool
            Denormalize the flux to physical units using the continuum model, if any
        wlim : slice
            Wavelength limits for the returned spectrum
        psf : PSF
            Convolution kernel
        idx : tuple
            Grid index
        kwargs : dict
            Stellar parameters
        """
        
        if method in ['grid', 'nearest']:
            msg_method = 'assigned from grid'
        else:
            if self.array_grid is None and self.rbf_grid is None:
                raise NotImplementedError("General interpolation is supported on ArrayGrid and RbfGrid only.")
        
            msg_method = 'calculated from {method} interpolation'

        if psf is not None:
            msg_psf = f' and convolved with PSF of type `{type(psf).__name__}`'
        else:
            msg_psf = ''
        
        # Determine the wave limits, convert to slice from (wmin, wmax)
        wlim = wlim if wlim is not None else self.get_wave_slice()
        if isinstance(wlim, Iterable):
            wlim = self.get_wave_slice_impl(wlim)

        # Get post-processing function to be passed to the grid. The grid object will
        # execute this before interpolation and optinally cache the results for
        # further interpolations.
        cache_key_prefix, post_process = self.get_model_post_process(wlim=wlim, psf=psf)
    
        r = interp_fun('flux', post_process=post_process, cache_key_prefix=cache_key_prefix, wlim=wlim, idx=idx, **kwargs)
        if r is None:
            return None
        flux, params = r

        if flux is not None:
            spec = self.get_parameterized_spectrum(wlim=wlim, **params)
            spec.flux = flux
            spec.append_history(f'Flux vector {msg_method}{msg_psf}.')

            if self.grid.has_error('flux'):
                # TODO: how to interpolate and convolve the error?
                raise NotImplementedError()
            else:
                spec.append_history('Flux error vector is not found in grid.')

            if self.grid.has_value('cont'):
                r = interp_fun('cont', post_process=post_process, cache_key_prefix=cache_key_prefix, wlim=wlim, idx=idx, **kwargs)
                if r is not None:
                    spec.cont, _ = r
                    spec.append_history(f'Continuum model vector {msg_method}{msg_psf}.')
            else:
                spec.append_history('Continuum model vector is not found in grid.')

            if denormalize and self.continuum_model is not None:
                # TODO: Use same interpolation here as above!
                raise NotImplementedError()
                cont_params = self.get_continuum_parameters(**params)
                self.continuum_model.denormalize(spec, cont_params, s=wlim)

            act_params = { k: params[k] for _, k, _ in self.grid.enumerate_axes() }
            if idx is not None:
                spec.append_history(f'Interpolated model with actual model parameters: {act_params}, requested model index: {idx}')
            else:
                req_params = { k: kwargs[k] for _, k, _ in self.grid.enumerate_axes() if k in kwargs }
                spec.append_history(f'Interpolated model with actual model parameters: {act_params}, requested model parameters: {req_params}')

            return spec
        else:
            logger.warning(f'Spectrum cannot be interpolated to parameters {kwargs} using {method} interpolation on grid of type `{type(self.grid).__name__}` with config `{type(self.config).__name__}`')
            return None
   
    #endregion