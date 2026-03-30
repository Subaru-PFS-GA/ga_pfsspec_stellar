import os

from pfs.ga.pfsspec.core.grid import ArrayGrid
from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid.bosz import Bosz
from pfs.ga.pfsspec.stellar.grid.phoenix import Phoenix
from pfs.ga.pfsspec.stellar.grid.gk2025 import GK2025

from ..core.test_base import TestBase

class StellarTestBase(TestBase):
    def setUp(self):
        super().setUp()

        self.kurucz_grid = None
        self.bosz_grid = None

        self.phoenix_grid = None
        self.phoenix_pca_grid = None

        self.gk2025_grid = None

        self.normalized_grids = None

    def get_kurucz_grid(self):
        # TODO: file location is broken
        raise NotImplementedError()

        if self.kurucz_grid is None:
            file = os.path.join(self.PFSSPEC_DATA_PATH, 'stellar/compressed/kurucz.h5')
            self.kurucz_grid = KuruczGrid(model='test')
            self.kurucz_grid.load(file, s=None, format='h5')

        return self.kurucz_grid

    def get_bosz_grid(self):
        if self.bosz_grid is None:
            file = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/bosz/bosz_5000_full/spectra.h5')
            self.bosz_grid = ModelGrid.from_file(file)

        return self.bosz_grid

    def get_phoenix_grid(self):
        if self.phoenix_grid is None:
            file = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/phoenix/phoenix_HiRes/spectra.h5')
            self.phoenix_grid = ModelGrid(Phoenix(), ArrayGrid)
            self.phoenix_grid.load(file, s=None, format='h5')

        return self.phoenix_grid

    def get_phoenix_pca_grid(self):
        if self.phoenix_pca_grid is None:
            file = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/rbf/phoenix/phoenix_HiRes_GK/pca_none_weights_3/spectra.h5')
            self.phoenix_pca_grid = ModelGrid(Phoenix(pca=True), ArrayGrid)
            self.phoenix_pca_grid.load(file, s=None, format='h5')

        return self.phoenix_pca_grid

    def get_gk2025_grid(self):
        if self.gk2025_grid is None:
            file = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/gk2025/gk2025_binned_compressed/spectra.h5')
            self.gk2025_grid = ModelGrid.from_file(file)

        return self.gk2025_grid
    
    def get_normalized_grids(self):
        if self.normalized_grids is None:
            self.normalized_grids = {}
            # file = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/roman/gridie/spectra.h5')
            file = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/gk2025/gk2025_binned_compressed/spectra.h5')
            self.normalized_grids['b'] = ModelGrid.from_file(file)

            # file = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/roman/grid7/spectra.h5')
            file = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/gk2025/gk2025_binned_compressed/spectra.h5')
            self.normalized_grids['mr'] = ModelGrid.from_file(file)
            self.normalized_grids['r'] = ModelGrid.from_file(file)

        return self.normalized_grids