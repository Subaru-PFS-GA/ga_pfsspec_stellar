import os

from test.core import TestBase

from pfsspec.core.grid import ArrayGrid
from pfsspec.stellar.grid import ModelGrid
from pfsspec.stellar.grid.bosz import Bosz
from pfsspec.stellar.grid.phoenix import Phoenix

class StellarTestBase(TestBase):
    def setUp(self):
        super().setUp()

        self.kurucz_grid = None
        self.bosz_grid = None
        self.phoenix_grid = None

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
            self.bosz_grid = ModelGrid(Bosz(), ArrayGrid)
            self.bosz_grid.load(file, s=None, format='h5')

        return self.bosz_grid

    def get_phoenix_grid(self):
        if self.phoenix_grid is None:
            file = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/phoenix/phoenix_HiRes/spectra.h5')
            self.phoenix_grid = ModelGrid(Phoenix(), ArrayGrid)
            self.phoenix_grid.load(file, s=None, format='h5')

        return self.phoenix_grid