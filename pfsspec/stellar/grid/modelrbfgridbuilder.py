import logging
import os
import gc
import numpy as np
import time
from tqdm import tqdm

from pfsspec.core.util.array_filters import *
from pfsspec.core.grid import ArrayGrid
from pfsspec.core.grid import RbfGrid
from pfsspec.core.grid import PcaGrid
from pfsspec.core.grid import RbfGridBuilder
from pfsspec.stellar.grid import ModelGrid
from pfsspec.stellar.grid import ModelGridBuilder

class ModelRbfGridBuilder(RbfGridBuilder, ModelGridBuilder):

    STEPS = ['flux', 'fit', 'pca']

    def __init__(self, grid_config, grid=None, orig=None):
        RbfGridBuilder.__init__(self, orig=orig)
        ModelGridBuilder.__init__(self, grid_config, orig=orig)

        if isinstance(orig, ModelRbfGridBuilder):
            self.step = orig.step
        else:
            self.step = None

    def add_subparsers(self, configurations, parser):
        return None

        # TODO: add continuum model parameters

    def add_args(self, parser):
        RbfGridBuilder.add_args(self, parser)
        ModelGridBuilder.add_args(self, parser)

        parser.add_argument('--step', type=str, choices=ModelRbfGridBuilder.STEPS, help='RBF step to perform.\n')

    def init_from_args(self, config, args):
        self.step = self.get_arg('step', self.step, args)

        self.pca = (self.step == 'pca')

        RbfGridBuilder.init_from_args(self, config, args)
        ModelGridBuilder.init_from_args(self, config, args)
      
    def create_input_grid(self):
        # It doesn't really matter if the input is already a PCA grid or just a direct
        # array because RBF interpolation is the same. On the other hand,
        # when we want to slice a PCA grid in wavelength, we have to load the
        # eigenvectors so this needs to be extended here.
        
        return ModelGridBuilder.create_input_grid(self)

    def open_input_grid(self, input_path):
        return ModelGridBuilder.open_input_grid(self, input_path)

    def create_output_grid(self):
        config = self.grid_config
        if self.step == 'pca':
            config = type(config)(pca=True)
        grid = ModelGrid(config, RbfGrid)
        return grid

    def open_data(self, input_path, output_path, params_path=None):
        return ModelGridBuilder.open_data(self, input_path, output_path, params_path=params_path)

    def build_data_index(self):
        return ModelGridBuilder.build_data_index(self)

    def open_output_grid(self, output_path):
        fn = os.path.join(output_path, 'spectra') + '.h5'
        
        # Pad the output axes. This automatically takes the parameter ranges into
        # account since the grid is sliced.
        orig_axes = self.input_grid.get_axes()
        if self.padding:
            padded_axes = ArrayGrid.pad_axes(orig_axes)
            self.output_grid.set_axes(padded_axes)
        else:
            self.output_grid.set_axes(orig_axes)

        # DEBUG
        # self.output_grid.preload_arrays = True
        # self.output_grid.grid.preload_arrays = True
        # END DEBUG

        self.output_grid.filename = fn
        self.output_grid.fileformat = 'h5'

    def build_rbf(self, input_grid, output_grid, name,
        method=None, function=None, epsilon=None, s=None):

        value = input_grid.get_value(name, s=s)

        if value is None:
            self.logger.warning('Skipping RBF fit to array `{}` since it is empty.'.format(name))
        else:
            mask = input_grid.get_value_index(name)
            axes = input_grid.get_axes()

            if self.padding:
                value, axes, mask = pad_array(axes, value, mask=mask)
                self.logger.info('Array `{}` padded to shape {}'.format(name, value.shape))

            self.logger.info('Fitting RBF to array `{}` of size {} using {} with {} and eps={}'.format(name, value.shape, method, function, epsilon))
            rbf = self.fit_rbf(value, axes, mask=mask, 
                method=method, function=function, epsilon=epsilon)
            output_grid.set_value(name, rbf)

            # Do explicit garbage collection
            del value
            del mask
            gc.collect()


    def fit_params(self, params_grid, output_grid):
        # Calculate RBF interpolation of continuum fit parameters
        # This is done parameter by parameter so continuum models which cannot
        # be fitted everywhere are still interpolated to as many grid positions
        # as possible. Unfortunately, because of this, we cannot reuse the RBF
        # distance and kernel matrix.

        for p in params_grid.continuum_model.get_model_parameters():
            # TODO: can we run this with a PcaGrid output?
            if params_grid.array_grid.has_value(p.name):
                self.build_rbf(params_grid.array_grid, output_grid.rbf_grid,
                p.name, method=p.rbf_method, function=p.rbf_function, epsilon=p.rbf_epsilon)

    def copy_params(self, params_grid, output_grid):
        # Copy RBF interpolation of continuum fit parameters from an existing grid

        output_grid.set_constants(params_grid.get_constants())
        output_grid.set_wave(params_grid.get_wave())

        for p in self.continuum_model.get_model_parameters():
            self.copy_rbf(params_grid.grid, output_grid.grid.grid, p.name)

    def fit_flux(self, input_grid, output_grid):
        # Calculate RBF interpolation in the flux vector directly

        output_grid.set_wave(input_grid.get_wave())

        for name in ['flux', 'cont']:
            if input_grid.grid.has_value(name):
                self.build_rbf(input_grid.array_grid, output_grid.rbf_grid, name,
                    s=input_grid.get_wave_slice())

    def run_step_fit(self):
        # Fit RBF to continuum parameters or/and the flux directly
    
        if self.params_grid is not None:
            params_grid = self.params_grid
        elif self.input_grid.continuum_model is not None:
            params_grid = self.input_grid
        else:
            params_grid = None

        # Copy continuum fit parameters
        if params_grid is not None:
            self.copy_wave(params_grid, self.output_grid)
            self.copy_constants(params_grid, self.output_grid)
            self.fit_params(params_grid, self.output_grid)
        
    def run_step_flux(self):
        self.run_step_fit()
        self.copy_wave(self.input_grid, self.output_grid)
        self.fit_flux(self.input_grid, self.output_grid)

    def run_step_pca(self):
        # Calculate the RBF interpolation of the principal components. Optionally,
        # if a params grid is specified, copy the existing RBF interpolation of
        # those to the output grid.

        if self.params_grid is not None:
            self.copy_constants(self.params_grid, self.output_grid)

            if self.rbf:
                # Copy RBF interpolation of continuum parameters
                self.copy_params(self.params_grid, self.output_grid)
            else:
                # Run interpolation of continuum parameters
                self.fit_params(self.params_grid, self.output_grid)
        else:
            # Run interpolation of continuum parameters taken from the PCA grid
            self.copy_constants(self.input_grid, self.output_grid)

            self.fit_params(self.input_grid, self.output_grid)

        # Calculate RBF interpolation of principal components
        grid = self.input_grid.grid
        for name in ['flux', 'cont']:
            if self.input_grid.grid.has_value(name):
                # Dig down to the innermost grid (Array/RBF within PcaGrid within ModelGrid)
                self.build_rbf(self.input_grid.grid.grid, self.output_grid.grid.grid, name,
                    method='sparse', function='gaussian', epsilon=1.0)

        # Copy wave vector, eigenvalues and eigenvectors
        self.copy_wave(self.input_grid, self.output_grid)
        for name in ['flux', 'cont']:
            self.output_grid.grid.eigs[name] = self.input_grid.grid.eigs[name]
            self.output_grid.grid.eigv[name] = self.input_grid.grid.eigv[name]

    def run(self):
        if self.step == 'flux':
            self.run_step_flux()
        elif self.step == 'fit':
            self.run_step_fit()
        elif self.step == 'pca':
            self.run_step_pca()
        else:
            raise NotImplementedError()

#endregion