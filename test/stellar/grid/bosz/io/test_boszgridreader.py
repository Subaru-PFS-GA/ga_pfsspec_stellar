import os
import numpy as np

from test.core import TestBase
from pfsspec.core.grid import ArrayGrid
from pfsspec.stellar.grid import ModelGrid
from pfsspec.stellar.grid.bosz import Bosz
from pfsspec.stellar.grid.bosz.io import BoszGridReader, BoszSpectrumReader

class TestBoszGridReader(TestBase):
    def create_grid_reader(self):
        path = os.path.join(self.PFSSPEC_DATA_PATH, 'download/models/stellar/grid/bosz/bosz_5000/')

        gridreader = BoszGridReader()
        gridreader.top = 10
        gridreader.parallel = False

        reader = gridreader.reader = BoszSpectrumReader(path, wave_lim=[3000, 9000], resolution=5000, format='ascii')
        fn = reader.get_filename(M_H=0.0, T_eff=5000.0, log_g=1.0, a_M=0.0, C_M=0.0, R=5000)
        fn = os.path.join(path, fn)
        spec = reader.read(fn)
        
        grid = gridreader.grid = gridreader.create_grid()
        grid.preload_arrays = True
        grid.set_wave(spec.wave)
        grid.grid.init_values()
        grid.build_axis_indexes()

        return grid, gridreader

    def test_get_example_filename(self):
        grid, gridreader = self.create_grid_reader()
        fn = gridreader.get_example_filename()

        self.assertEqual('amp00cp00op00t5000g10v20modrt0b5000rs.asc.bz2', fn)

    def test_read_grid(self):
        grid, gridreader = self.create_grid_reader()
        gridreader.read_grid()
        self.assertEqual((14, 66, 11, 6, 4, 10986), grid.get_flux_shape())
        self.assertEqual(11, np.sum(grid.grid.value_indexes['flux']))
