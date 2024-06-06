import os
import numpy as np

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.grid import ArrayGrid
from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid.grid7 import Grid7
from pfs.ga.pfsspec.stellar.grid.grid7.io import Grid7GridReader, Grid7SpectrumReader

class TestGrid7GridReader(TestBase):
    def create_grid_reader(self):
        path = os.path.join(self.PFSSPEC_DATA_PATH, 'download/models/stellar/grid/roman/grid7')

        gridreader = Grid7GridReader()
        gridreader.top = 10
        gridreader.parallel = False

        reader = gridreader.reader = Grid7SpectrumReader(path, wave_lim=[3000, 9000], resolution=50000, format='bin')
        fn = reader.get_filename(M_H=0.0, T_eff=5000.0, log_g=1.0, a_M=0.0, R=5000)
        fn = os.path.join(path, fn)
        spec = reader.read(fn)
        
        grid = gridreader.grid = gridreader.create_grid()
        grid.preload_arrays = True
        grid.set_wave(spec.wave, wave_edges=spec.wave_edges)
        grid.grid.init_values()
        grid.build_axis_indexes()

        return grid, gridreader, path

    def test_get_example_filename(self):
        grid, gridreader, path = self.create_grid_reader()
        fn = gridreader.get_example_filename()

        self.assertEqual('bin/t5000/g_10/t5000g_10f_00a_00.bin.gz', fn)

    def test_get_grid_axes(self):
        grid, gridreader, path = self.create_grid_reader()

        axes = gridreader.get_grid_axes(path)
        pass

    def test_read_grid(self):
        grid, gridreader, path = self.create_grid_reader()
        gridreader.read_grid()
        self.assertEqual((50, 34, 11, 21, 20001), grid.get_flux_shape())
        self.assertEqual(10, np.sum(grid.grid.value_indexes['flux']))
