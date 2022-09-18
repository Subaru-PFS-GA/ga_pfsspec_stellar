import os
from timeit import default_timer as timer
import cProfile
import numpy as np

from test.core import TestBase
from pfs.ga.pfsspec.core.grid import ArrayGrid
from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid.bosz import Bosz
from pfs.ga.pfsspec.stellar.grid.phoenix import Phoenix

class TestModelGrid_Pca(TestBase):
    def get_test_grid(self, args, preload_arrays=False):
        #file = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/rbf/bosz/bosz_5000_GF/pca/spectra.h5')
        #grid = ModelGrid(Bosz(pca=True), ArrayGrid)
        
        file = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/rbf/phoenix/phoenix_HiRes_GK/pca_none_weights_3/spectra.h5')
        grid = ModelGrid(Phoenix(pca=True), ArrayGrid)

        #file = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/rbf/phoenix/phoenix_HiRes_GK/norm/spectra.h5')
        #grid = ModelGrid(Phoenix(pca=False), ArrayGrid)
        
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
        spec = grid.get_nearest_model(M_H=0., T_eff=4500, log_g=4, C_M=0, a_M=0, denormalize=True)
        self.assertIsNotNone(spec)

    def test_get_model_at_perf(self):

        args = {}
        grid = self.get_test_grid(args, preload_arrays=False)
        grid.pca_grid.k = 2000

        idxs = np.stack(np.where(grid.array_grid.value_indexes['flux']), axis=-1)

        pr = cProfile.Profile()
        pr.enable()

        start = timer()
        q = 0
        for idx in idxs[:100]:
            spec = grid.get_model_at(idx, denormalize=True)
            q += 1
        end = timer()

        t_iter = (end - start) / q
        # hdf5: 0.30
        # no denorm: 0.26
        
        # preload: 0.27
        # no denorm: 0.23

        pr.disable()
        pr.print_stats(sort='cumtime')

        print(t_iter)