import os
import logging
from tqdm import tqdm
import numpy as np

from test.core import TestBase
from pfsspec.core.grid import ArrayGrid
from pfsspec.stellar.grid import ModelGrid
from pfsspec.stellar.grid.bosz import Bosz

class TestModelGrid_Fit(TestBase):
    def get_test_grid(self, args):
        #file = os.path.join(self.PFSSPEC_DATA_PATH, '/scratch/ceph/dobos/temp/test072/spectra.h5')
        #file = '/scratch/ceph/dobos/data/pfsspec/import/stellar/rbf/bosz_5000_full/fitrbf_3/spectra.h5'
        file = '/datascope/subaru/data/pfsspec/models/stellar/rbf/bosz/bosz_5000_full_3/fit/spectra.h5'
        grid = ModelGrid(Bosz(), ArrayGrid)
        grid.load(file, format='h5')
        grid.init_from_args(args)

        return grid

    def test_init_from_args(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual(slice(None), grid.get_wave_slice())

    def test_get_axes(self):
        args = {}
        grid = self.get_test_grid(args)
        axes = grid.get_axes()
        self.assertEqual(3, len(axes))

    def test_get_nearest_model(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(Fe_H=0., T_eff=4500, log_g=4, C_M=0, O_M=0)
        self.assertIsNotNone(spec)
        
    def test_get_continuum_parameters(self):
        args = {}
        grid = self.get_test_grid(args)
        params = grid.get_continuum_parameters(Fe_H=-1.2, T_eff=4250, log_g=4.0, O_M=0.0, C_M=-0.0)

    def test_get_continuum_parameters_at(self):
        args = {}
        grid = self.get_test_grid(args)
        idx = grid.get_nearest_index(Fe_H=-1.2, T_eff=4125, log_g=4.3, O_M=0.1, C_M=-0.1)
        params = grid.get_continuum_parameters_at(idx)

    def test_get_denormalized_model(self):
        args = {}
        grid = self.get_test_grid(args)
        grid.grid.value_indexes['flux'] = \
            grid.grid.value_indexes['legendre'] & \
            grid.grid.value_indexes['blended_0'] & \
            grid.grid.value_indexes['blended_1'] & \
            grid.grid.value_indexes['blended_2']
        idx = [ ix[100] for ix in np.where(grid.grid.value_indexes['flux']) ]
        spec = grid.get_model_at(idx, denormalize=True)
        pass