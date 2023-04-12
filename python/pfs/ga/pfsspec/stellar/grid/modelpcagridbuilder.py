import os
import numpy as np
import time
from tqdm import tqdm

from pfs.ga.pfsspec.core import Physics
from pfs.ga.pfsspec.core.grid import ArrayGrid
from pfs.ga.pfsspec.core.grid import RbfGrid
from pfs.ga.pfsspec.core.grid import PcaGridBuilder
from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid import ModelGridBuilder

class ModelPcaGridBuilder(PcaGridBuilder, ModelGridBuilder):

    def __init__(self, grid_config, orig=None):
        PcaGridBuilder.__init__(self, orig=orig)
        ModelGridBuilder.__init__(self, grid_config, orig=orig)

        if isinstance(orig, ModelPcaGridBuilder):
            self.grid_config = grid_config if grid_config is not None else orig.grid_config
            
            self.weights_grid = orig.weights_grid
            self.weights_grid_index = None
        else:
            self.grid_config = grid_config
            
            self.weights_grid = None
            self.weights_grid_index = None

    def add_subparsers(self, configurations, parser):
        return None

        # TODO: add continuum model parameters

    def add_args(self, parser, config):
        PcaGridBuilder.add_args(self, parser, config)
        ModelGridBuilder.add_args(self, parser, config)

    def init_from_args(self, config, args):
        PcaGridBuilder.init_from_args(self, config, args)
        ModelGridBuilder.init_from_args(self, config, args)
    
    def create_input_grid(self):
        # Input should not be a PCA grid
        return ModelGridBuilder.create_input_grid(self)

    def create_output_grid(self):
        # Output is always a PCA grid
        config = type(self.grid_config)(orig=self.grid_config, pca=True)
        return ModelGrid(config, ArrayGrid)

    def create_weights_grid(self):
        # It should match the input grid but not necessarily the same file
        return ModelGridBuilder.create_input_grid(self)

    def open_input_grid(self, input_path):
        ModelGridBuilder.open_input_grid(self, input_path)

    def open_output_grid(self, output_path):
        ModelGridBuilder.open_output_grid(self, output_path)

    def open_weights_grid(self, weights_path):
        if os.path.isfile(weights_path):
            fn = weights_path
        else:
            fn = os.path.join(weights_path, 'weights') + '.h5'
        self.weights_grid.load(fn)

    def open_data(self, args, input_path, output_path, params_path=None, weights_path=None):
        if weights_path is not None:
            self.weights_grid = self.create_weights_grid()
            self.open_weights_grid(weights_path)
            self.weights_grid.init_from_args(args)
            self.weights_grid.build_axis_indexes()
        else:
            self.weights_grid = None

        ModelGridBuilder.open_data(self, args, input_path, output_path, params_path=params_path)

    def build_data_index(self):
        ModelGridBuilder.build_data_index(self)

        # TODO: try to merge some of it with ModelGridBuild.build_data_index because
        #       weights mask just need to be &-d to the data masks

        if self.weights_grid is not None:
            # TODO: something might need to be done here with slicing of the weights

            slice = self.input_grid.get_slice()
            index = self.input_grid.array_grid.get_value_index_unsliced('flux', s=slice)
            slice = self.weights_grid.get_slice()
            index &= self.weights_grid.array_grid.get_value_index_unsliced('weight', s=slice)
            index &= np.squeeze((self.weights_grid.array_grid.get_value('weight') > 0), axis=-1)
            self.weights_grid_index = np.array(np.where(index))
       
    def get_vector_shape(self):
        return self.input_grid.get_wave()[0].shape

    def get_vector(self, i):
        # When fitting, the output fluxes will be already normalized, so
        # here we return the flux field only
        idx = tuple(self.input_grid_index[:, i])
        
        # Spectrum
        spec = self.input_grid.get_model_at(idx)
        
        # Weight, if available
        if self.weights_grid is not None:
            w = self.weights_grid.grid.get_value_at('weight', idx)
        elif self.input_grid.grid.has_value('weight'):
            w = self.input_grid.grid.get_value_at('weight', idx)
        else:
            w = 1.0

        return spec.flux, w

    def run(self):
        super(ModelPcaGridBuilder, self).run()

        # Copy continuum fit parameters, if possible. The input continuum fit parameters
        # can be either in an ArrayGrid or an RbfGrid. Copying is only possible when the
        # input is an ArrayGrid since output is always an ArrayGrid.
        cmgrid = self.params_grid or self.input_grid
        if cmgrid is not None and cmgrid.continuum_model is not None:
            self.logger.info('Copying continuum parameters from input grid.')
            if isinstance(cmgrid.grid, ArrayGrid):
                for p in cmgrid.continuum_model.get_interpolated_params():
                    slice = cmgrid.get_slice()
                    index = cmgrid.grid.get_value_index(p.name, s=slice)
                    params = cmgrid.get_value_sliced(p.name)
                    self.output_grid.grid.grid.allocate_value(p.name, shape=(params.shape[-1],))
                    self.output_grid.grid.grid.set_value(p.name, params)
                    self.output_grid.grid.grid.value_indexes[p.name] = index
            elif isinstance(cmgrid.grid, RbfGrid):
                self.logger.info('Interpolating continuum parameters from input RBF grid.')

                output_axes = { p: ax for _, p, ax in self.output_grid.enumerate_axes(squeeze=False) }
                points = ArrayGrid.get_meshgrid_points(output_axes, interpolation='xyz', squeeze=False, indexing='ij')
                
                # Interpolate the values from the RBF grid and save them as an array grid
                for p in cmgrid.continuum_model.get_interpolated_params():
                    params = cmgrid.rbf_grid.get_value(p.name, **points)

                    self.output_grid.grid.grid.allocate_value(p.name, shape=(params.shape[-1],))
                    self.output_grid.grid.grid.set_value(p.name, params)
                    self.output_grid.grid.grid.value_indexes[p.name] = np.full(params.shape[:-1], True)
            else:
                raise NotImplementedError("Cannot copy continuum fit parameters.")
        
        # TODO: this is not used, verify if correct
        if self.input_grid.grid.has_constant('constants'):
            self.output_grid.grid.set_constant('constants', self.input_grid.grid.get_constant('constants'))

        # Save principal components to a grid
        input_slice = self.input_grid.get_slice()
        input_shape = self.input_grid.get_shape(s=input_slice, squeeze=False)
        coeffs = np.full(input_shape + (self.PC.shape[1],), np.nan)
        input_count = self.get_input_count()
        for i in range(input_count):
            idx = tuple(self.output_grid_index[:, i])
            coeffs[idx] = self.PC[i, :]

        # Calculate the residual variance, this represents the reconstruction error
        flux_err = np.std(self.R, axis=0)

        self.output_grid.grid.allocate_value('flux', shape=coeffs.shape, pca=True)
        self.output_grid.grid.set_value('flux', coeffs, 
                                        eigs=self.S, eigv=self.V, mean=self.M, error=flux_err,
                                        transform=self.pca_transform)

        # DEBUG

        # import h5py
        # f = h5py.File(self.output_grid.filename, 'a')
        # f.create_dataset('w', data=self.W)
        # f.create_dataset('x', data=self.X)
        # f.close()
        
        # END DEBUG

        
    