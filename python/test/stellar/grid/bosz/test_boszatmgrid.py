import os

from test.core import TestBase
from pfs.ga.pfsspec.stellar.grid.bosz import BoszAtmGrid

class TestBoszAtmGrid(TestBase):
    def test_load(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, '/scratch/ceph/dobos/data/pfsspec/import/stellar/atm/bosz/atm.h5')
        grid = BoszAtmGrid()
        grid.preload_arrays = True
        grid.load(filename, format='h5')

