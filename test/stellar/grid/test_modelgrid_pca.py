import os

from test.core import TestBase
from pfsspec.core.grid import ArrayGrid
from pfsspec.stellar.grid import ModelGrid
from pfsspec.stellar.grid.bosz import Bosz

class TestModelGrid_Pca(TestBase):
    def get_test_grid(self, args, preload_arrays=False):
        file = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/rbf/bosz/bosz_5000_GF/pca/spectra.h5')
        grid = ModelGrid(Bosz(pca=True), ArrayGrid)
        grid.preload_arrays = preload_arrays
        grid.load(file, format='h5')
        grid.init_from_args(args)

        return grid

    def test_init_from_args(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual(slice(None), grid.get_wave_slice())

    def test_preload_arrays(self):
        args = {}
        grid = self.get_test_grid(args, preload_arrays=True)

    def test_enumerate_axes(self):
        args = {}
        grid = self.get_test_grid(args)
        axes = grid.enumerate_axes()
        self.assertEqual(5, len(list(axes)))

    def test_get_shape(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual((14, 11, 11, 6, 4), grid.get_shape())

    def test_get_model_count(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual(36620, grid.get_model_count())

    def test_get_flux_shape(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual((14, 11, 11, 6, 4, 15404), grid.get_flux_shape())

        args = { 'T_eff': [5800, 7000] }
        grid = self.get_test_grid(args)
        self.assertEqual((14, 6, 11, 6, 4, 15404), grid.get_flux_shape())

        args = { 'T_eff': [5000] }
        grid = self.get_test_grid(args)
        self.assertEqual((14, 1, 11, 6, 4, 15404), grid.get_flux_shape())

        args = { 'T_eff': [5800, 7000], 'wave_lim': [4800, 5600] }
        grid = self.get_test_grid(args)
        self.assertEqual((14, 6, 11, 6, 4, 1542), grid.get_flux_shape())

    def test_get_nearest_model(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=4500, log_g=4, C_M=0, a_M=0)
        self.assertIsNotNone(spec)
