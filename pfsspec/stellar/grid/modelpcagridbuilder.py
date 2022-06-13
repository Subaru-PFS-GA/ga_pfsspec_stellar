import os
import numpy as np
import time
from tqdm import tqdm

from pfsspec.core import Physics
from pfsspec.core.grid import ArrayGrid
from pfsspec.core.grid import RbfGrid
from pfsspec.core.grid import PcaGridBuilder
from pfsspec.stellar.grid import ModelGrid
from pfsspec.stellar.grid import ModelGridBuilder

class ModelPcaGridBuilder(PcaGridBuilder, ModelGridBuilder):

    def __init__(self, grid_config, orig=None):
        PcaGridBuilder.__init__(self, orig=orig)
        ModelGridBuilder.__init__(self, grid_config, orig=orig)

        if isinstance(orig, ModelPcaGridBuilder):
            self.grid_config = grid_config if grid_config is not None else orig.grid_config
        else:
            self.grid_config = grid_config

    def add_subparsers(self, configurations, parser):
        return None

        # TODO: add continuum model parameters

    def add_args(self, parser):
        PcaGridBuilder.add_args(self, parser)
        ModelGridBuilder.add_args(self, parser)

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

    def open_input_grid(self, input_path):
        ModelGridBuilder.open_input_grid(self, input_path)

    def open_output_grid(self, output_path):
        ModelGridBuilder.open_output_grid(self, output_path)

    def open_data(self, args, input_path, output_path, params_path=None):
        return ModelGridBuilder.open_data(self, args, input_path, output_path, params_path=params_path)

    def build_data_index(self):
        return ModelGridBuilder.build_data_index(self)

    def get_vector_shape(self):
        return self.input_grid.get_wave().shape

    def get_vector(self, i):
        # When fitting, the output fluxes will be already normalized, so
        # here we return the flux field only
        idx = tuple(self.input_grid_index[:, i])
        spec = self.input_grid.get_model_at(idx)

        return spec.flux

    def run(self):
        super(ModelPcaGridBuilder, self).run()

        # Copy continuum fit parameters, if possible. The input continuum fit parameters
        # can be either in an ArrayGrid or an RbfGrid. Copying is only possible when the
        # input is an ArrayGrid since output is always an ArrayGrid.
        cmgrid = self.params_grid or self.input_grid
        if cmgrid is not None and cmgrid.continuum_model is not None:
            self.logger.info('Copying continuum parameters from input grid.')
            if isinstance(cmgrid.grid, ArrayGrid):
                for p in cmgrid.continuum_model.get_model_parameters():
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
                for p in cmgrid.continuum_model.get_model_parameters():
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

        self.output_grid.grid.allocate_value('flux', shape=coeffs.shape, pca=True)
        self.output_grid.grid.set_value('flux', (coeffs, self.S, self.V, self.M), pca=True)

        
    