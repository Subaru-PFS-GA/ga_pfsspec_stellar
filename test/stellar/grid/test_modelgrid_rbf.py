import os
import logging
from tqdm import tqdm
import numpy as np

from test.core import TestBase
from pfsspec.core.grid import RbfGrid
from pfsspec.stellar.grid import ModelGrid
from pfsspec.stellar.grid.bosz import Bosz

class TestModelGrid_Rbf(TestBase):
    def get_test_grid(self, args):
        file = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/rbf/bosz/bosz_5000_GF/flux/spectra.h5')
        grid = ModelGrid(Bosz(), RbfGrid)
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
        spec = grid.get_nearest_model(M_H=0., T_eff=4500, log_g=4, C_M=0, a_M=0)
        self.assertIsNotNone(spec)
        
    def test_interpolate_model_rbf(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.interpolate_model_rbf(M_H=-1.2, T_eff=4125, log_g=4.3, a_M=0.1, C_M=-0.1)
        self.assertIsNotNone(spec)

    def test_interpolate_model_rbf_performance(self):
        # logging.basicConfig(level=logging.DEBUG)

        args = {}
        grid = self.get_test_grid(args)
        rbf_raw = grid.grid

        for i in tqdm(range(100)):
            # Draw a value for each parameters
            # Note that these are indices and not physical values!
            params = {}
            for k in rbf_raw.axes:
                v = rbf_raw.axes['M_H'].values
                params[k] = np.random.uniform(v[0], v[-1])
            
            rbf_raw.get_values(names=['blended_0', 'blended_1', 'blended_2'], **params)