import os
import numpy as np

from test.core import TestBase
from pfs.ga.pfsspec.core.grid import ArrayGrid
from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid.bosz import Bosz

class TestBoszModelGrid(TestBase):
    def get_grid(self, normalized=False):
        if not normalized:
            fn = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/bosz/bosz_50000/spectra.h5')
        else:
            fn = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/bosz/bosz_50000_rbf/fit/spectra.h5')

        grid = ModelGrid(Bosz(normalized=normalized), ArrayGrid)
        grid.preload_arrays = False
        grid.load(fn, format='h5')

        return grid

    def test_get_nearest_model(self):
        grid = self.get_grid()
        spec = grid.get_nearest_model(M_H=-1.5, T_eff=5000, log_g=1.5, a_M=0.0, C_M=0.0)
        self.assertIsNotNone(spec)

    def test_load_normalized(self):
        grid = self.get_grid(True)
        spec = grid.get_nearest_model(M_H=0, T_eff=4500, log_g=1, a_M=0, C_M=0)
        self.assertIsNotNone(spec)

    def test_get_slice_rbf(self):
        grid = self.get_grid()

        wl_idx = np.digitize([6565], grid.wave)
        flux, cont, axes = grid.get_slice_rbf(s=wl_idx, O_M=0, C_M=0)
        self.assertEqual((3, 8752), flux.xi.shape)
        self.assertEqual((8752,), flux.nodes.shape)

        wl_idx = np.digitize([6565, 6575], grid.wave)
        flux, cont, axes = grid.get_slice_rbf(s=slice(wl_idx[0], wl_idx[1]), O_M=0, C_M=0)
        self.assertEqual((3, 8752), flux.xi.shape)
        self.assertEqual((8752, 152), flux.nodes.shape)

        pass

    def test_get_slice_rbn_nopadding(self):
        grid = self.get_grid()

        wl_idx = np.digitize([6565], grid.wave)
        flux, cont, axes = grid.get_slice_rbf(padding=False, s=wl_idx, O_M=0, C_M=0)
        self.assertEqual((3, 6336), flux.xi.shape)
        self.assertEqual((6336,), flux.nodes.shape)