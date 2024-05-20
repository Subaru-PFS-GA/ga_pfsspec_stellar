import os
from tqdm import tqdm
import numpy as np

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.grid import RbfGrid
from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid.bosz import Bosz

class TestModelGrid_Pca_Rbf(TestBase):
    def get_test_grid(self, args):
        file = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/rbf/bosz/bosz_50000_FGK/pca-rbf/spectra.h5')
        grid = ModelGrid(Bosz(pca=True), RbfGrid)
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

    def test_get_model(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_model(M_H=0., T_eff=5500, log_g=4, C_M=0, a_M=0)
        self.assertIsNotNone(spec)
        
    def test_interpolate_model_rbf(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.interpolate_model_rbf(M_H=-1.2, T_eff=5125, log_g=4.3, a_M=0.1, C_M=-0.1)
        self.assertIsNotNone(spec)

    def test_get_continuum_parameters(self):
        args = {}
        grid = self.get_test_grid(args)
        params = grid.get_continuum_parameters(M_H=-1.2, T_eff=5125, log_g=4.3, a_M=0.1, C_M=-0.1)

    def test_get_continuum_parameters_at(self):
        args = {}
        grid = self.get_test_grid(args)
        idx = grid.get_index(M_H=-1.2, T_eff=5125, log_g=4.3, a_M=0.1, C_M=-0.1)
        params = grid.get_continuum_parameters_at(idx)

    def test_interpolate_model_rbf_performance(self):
        args = {}
        grid = self.get_test_grid(args)
        rbf_raw = grid.grid.grid

        for i in tqdm(range(100)):
            # Draw a value for each parameters
            # Note that these are indices and not physical values!
            params = {}
            for k in rbf_raw.axes:
                v = rbf_raw.axes['M_H'].values
                params[k] = np.random.uniform(v[0], v[-1])
            
            rbf_raw.get_values(names=['blended_0', 'blended_1', 'blended_2'], **params)