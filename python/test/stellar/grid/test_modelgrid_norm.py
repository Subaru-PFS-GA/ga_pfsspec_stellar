import os
import logging
from tqdm import tqdm
import numpy as np

from test.core import TestBase
from pfs.ga.pfsspec.core.grid import ArrayGrid
from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid.bosz import Bosz
from pfs.ga.pfsspec.stellar.grid.phoenix import Phoenix

class TestModelGrid_Fit(TestBase):
    def get_test_grid(self, args):
        # Run these tests on a normalized grid that has the flux array set.
        # file = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/rbf/bosz/bosz_5000_GF/norm/spectra.h5')
        # grid = ModelGrid(Bosz(), ArrayGrid)

        file = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/rbf/phoenix/phoenix_HiRes_GK/fit/spectra.h5')
        grid = ModelGrid(Phoenix(normalized=True), ArrayGrid)

        grid.load(file, format='h5')
        grid.init_from_args(args)

        return grid

    def test_init_from_args(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual(slice(None), grid.get_wave_slice())

    def test_enumerate_axes(self):
        args = {}
        grid = self.get_test_grid(args)
        axes = grid.enumerate_axes()
        self.assertEqual(5, len(list(axes)))

    def test_get_nearest_model(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=5500, log_g=4, C_M=0, a_M=0)
        self.assertIsNotNone(spec)
        
    def test_get_continuum_parameters(self):
        args = {}
        grid = self.get_test_grid(args)
        params = grid.get_continuum_parameters(Fe_H=-1.2, T_eff=5500, log_g=4.0, O_M=0.0, C_M=-0.0)

    def test_get_continuum_parameters_at(self):
        args = {}
        grid = self.get_test_grid(args)
        idx = grid.get_nearest_index(Fe_H=-1.2, T_eff=5125, log_g=4.3, O_M=0.1, C_M=-0.1)
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