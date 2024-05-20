import os
import numpy as np

from pfs.ga.pfsspec.core.setup_logger import logger
from pfs.ga.pfsspec.core.grid import ArrayGrid
from pfs.ga.pfsspec.core.grid import RbfGrid
from pfs.ga.pfsspec.core.grid import GridBuilder
from pfs.ga.pfsspec.stellar.grid import ModelGrid

class ModelGridBuilder():
    # Mixin class for spectrum model specific grid building operations
    #
    # - params_grid holds a reference to the grid from which model parameters
    #   are taken. These can be continuum model fit parameters or else
    # - params_grid_index keeps track of the model grid locations that hold
    #   valid data. This index is combined with the index on input_grid

    def __init__(self, grid_config, orig=None):
        if isinstance(orig, ModelGridBuilder):
            self.params_grid = orig.params_grid
            self.params_grid_index = None

            self.pca = orig.pca
            self.rbf = orig.rbf
            self.grid_config = grid_config if grid_config is not None else orig.grid_config
            self.continuum_model = orig.continuum_model
        else:
            self.params_grid = None
            self.params_grid_index = None

            self.pca = None
            self.rbf = None
            self.grid_config = grid_config
            self.continuum_model = None

    def add_args(self, parser, config):
        self.grid_config.add_args(parser, config)
        
        parser.add_argument('--pca', action='store_true', help='Run on a PCA input grid.')
        parser.add_argument('--rbf', action='store_true', help='Run on an RBF params grid.')

    def init_from_args(self, script, config, args):
        self.pca = self.get_arg('pca', self.pca, args)
        self.rbf = self.get_arg('rbf', self.rbf, args)
        
        self.grid_config.init_from_args(config, args)
        
        # Create the default continuum model and initialize based on command-line arguments
        self.continuum_model = self.grid_config.create_continuum_model()
        if self.continuum_model is not None:
            self.continuum_model.init_from_args(args)

    def create_params_grid(self):
        if self.rbf is not None and self.rbf:
            t = RbfGrid
        else:
            t = ArrayGrid
        grid = ModelGrid(type(self.grid_config)(orig=self.grid_config, normalized=True), t)
        return grid

    def create_input_grid(self):
        # TODO: add support for RBF grid

        config = self.grid_config
        if self.pca is not None and self.pca:
            config = type(config)(pca=True)
        grid = ModelGrid(config, ArrayGrid)
        return grid

    def create_output_grid(self):
        return ModelGrid(self.grid_config, ArrayGrid)

    def open_params_grid(self, params_path):
        fn = os.path.join(params_path, 'spectra') + '.h5'
        self.params_grid.load(fn)

    def open_input_grid(self, input_path):
        fn = os.path.join(input_path, 'spectra') + '.h5'
        self.input_grid.load(fn)

    def open_output_grid(self, output_path):
        # Copy data from the input grid
        g = self.input_grid or self.params_grid

        input_slice = g.get_slice()
        output_axes = { p: axis for i, p, axis in g.enumerate_axes(s=input_slice, squeeze=False) }
        self.output_grid.set_axes(output_axes)

        wave, wave_edges, wave_mask = g.get_wave()
        self.output_grid.set_wave(wave, wave_edges=wave_edges)
        self.output_grid.build_axis_indexes()
        
        # DEBUG
        # self.output_grid.preload_arrays = True
        # END DEBUG

    def open_data(self, args, input_path, output_path, params_path):
        if params_path is not None:
            self.params_grid = self.create_params_grid()
            self.open_params_grid(params_path)
            self.params_grid.init_from_args(args)
            self.params_grid.build_axis_indexes()

            self.grid_shape = self.params_grid.get_shape(s=self.params_grid.get_slice(), squeeze=False)

            # Initialize continuum model
            if self.continuum_model is None:
                self.continuum_model = self.params_grid.continuum_model

            if self.continuum_model.wave is None:
                wave, _, _ = self.params_grid.get_wave()
                self.continuum_model.init_wave(wave)
        else:
            self.params_grid = None
            
        GridBuilder.open_data(self, args, input_path, output_path)

        if self.continuum_model is None:
            self.continuum_model = self.input_grid.continuum_model
        
        if self.continuum_model is not None and self.continuum_model.wave is None:
            wave, _, _ = self.input_grid.get_wave()
            self.continuum_model.init_wave(wave)

        # This has to happen after loading the input grid because params_index
        # is combined with the input index with logical and
        if self.params_grid is not None:
            self.build_params_index()

        # Set continuum model of the output grid
        if self.continuum_model is not None:
            self.output_grid.set_continuum_model(self.continuum_model)
        
        # Force creating output file for direct hdf5 writing
        fn = os.path.join(output_path, 'spectra') + '.h5'
        self.output_grid.save(fn, format='h5')

    def build_data_index(self):
        # Source indexes that are used to get the input values
        # The shape matches the shape of the input array before slicing with
        # axis bounds.
        slice = self.input_grid.get_slice()
        index = self.input_grid.array_grid.get_value_index_unsliced('flux', s=slice)
        self.input_grid_index = np.array(np.where(index))

        # Target indexes that are used to set the output values
        # The shape matches the shape out the output array which is the same
        # as the shape of the input array after slicing with axis bounds.
        index = self.input_grid.array_grid.get_value_index('flux', s=slice)
        self.output_grid_index = np.array(np.where(index))

    def build_params_index(self):
        # Build and index on the continuum fit parametes. This is a logical AND
        # combination

        if isinstance(self.params_grid.grid, ArrayGrid):
            params_index = None

            for p in self.continuum_model.get_interpolated_params():
                pi, _ = self.get_params_index(p.name)
                params_index = pi if params_index is None else params_index & pi

            params_index = np.array(np.where(params_index))
            self.params_grid_index = params_index
        else:
            self.params_grid_index = None

    def get_params_index(self, params_name):
        # Whan we have to combine data from an input grid and a params grid it might
        # happen that the param grid is sliced down to a smaller shape than the input grid.
        
        # Make sure that the params grid index and the input grid
        # index are combined to avoid all holes in the grid.
        
        # We have to do a bit of trickery here since params index and input index 
        # can have different shapes, although they must slice down to the same
        # shape.

        params_slice = self.params_grid.array_grid.slice
        params_index = self.params_grid.array_grid.get_value_index_unsliced(params_name, s=params_slice)

        if self.input_grid is not None and self.input_grid.array_grid.slice is not None:
            input_slice = self.input_grid.array_grid.slice
            input_index = self.input_grid.array_grid.get_value_index_unsliced('flux', s=input_slice)
            
            ii = input_index[input_slice or ()]
            pi = params_index[params_slice or ()]
            iis = ii.shape
            pis = pi.shape

            ii = ii.flatten() & pi.flatten()

            input_index[input_slice or ()] = ii.reshape(iis)
            params_index[params_slice or ()] = ii.reshape(pis)
        else:
            input_index = None

        return params_index, input_index

    def verify_data_index(self):
        # Make sure all data indices have the same shape
        # TODO: verify this here
        GridBuilder.verify_data_index(self)
        if self.params_grid is not None:
            assert(self.params_grid_index.shape[-1] == self.output_grid_index.shape[-1])

    def get_gridpoint_model(self, i):
        input_idx = tuple(self.input_grid_index[:, i])
        output_idx = tuple(self.output_grid_index[:, i])
        spec = self.input_grid.get_model_at(input_idx)
        return input_idx, output_idx, spec

    def get_gridpoint_params(self, i):
        # Get all parameters of the continuum model at a gridpoint
        params_idx = tuple(self.params_grid_index[:, i])
        output_idx = tuple(self.output_grid_index[:, i])

        params = {}
        for p in self.continuum_model.get_interpolated_params():
            params[p.name] = self.params_grid.grid.get_value_at(p.name, params_idx)

        return params_idx, output_idx, params

    def get_interpolated_params(self, **kwargs):
        # Interpolate the params grid to a location defined by kwargs

        params = {}
        for p in self.continuum_model.get_interpolated_params():
            v = self.params_grid.grid.get_value(p.name, **kwargs)
            params[p.name] = v

        return params

    def copy_value(self, input_grid, output_grid, name):
        logger.info('Copying value array `{}`'.format(name))
        raise NotImplementedError()

    def copy_rbf(self, input_grid, output_grid, name):
        logger.info('Copying RBF array `{}`'.format(name))
        rbf = input_grid.values[name]
        output_grid.set_value(name, rbf)

    def copy_wave(self, params_grid, output_grid):
        wave, wave_edges, wave_mask = params_grid.get_wave()
        output_grid.set_wave(wave, wave_edges=wave_edges)

    def copy_constants(self, params_grid, output_grid):
        output_grid.set_constants(params_grid.get_constants())