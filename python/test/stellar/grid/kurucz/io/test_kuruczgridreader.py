import os

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.stellar.grid.kurucz.io import KuruczSpectrumReader
from pfs.ga.pfsspec.stellar.grid.kurucz.io import KuruczGridReader
from pfs.ga.pfsspec.stellar.grid.kurucz import KuruczGrid

class TestKuruczGridReader(TestBase):
    def test_read_grid_kurucz(self):
        path = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/atlas9/')
        grid = KuruczGrid('test')
        grid.preload_arrays = True
        reader = KuruczGridReader(grid, path)
        reader.read_grid()
        self.assertEqual((2, 61, 11, 1221), grid.values['flux'].shape)

    def test_get_filename(self):
        self.skipTest()