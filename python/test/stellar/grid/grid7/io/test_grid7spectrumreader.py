import os

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.grid import ArrayGrid
from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid.grid7.io import Grid7SpectrumReader

class TestBoszSpectrumReader(TestBase):
    def test_get_filename(self):
        reader = Grid7SpectrumReader(format='bin', resolution=50000)
        fn = reader.get_filename(M_H=-0.5, T_eff=4000, log_g=3, a_M=0)
        self.assertEqual("bin/t4000/g_30/t4000g_30f-05a_00.bin.gz", fn)

    def test_parse_filename(self):
        fn = 't4000g_30f-05a_00.bin.gz'
        p = Grid7SpectrumReader.parse_filename(fn)
        self.assertEqual(-0.5, p['M_H'])
        self.assertEqual(0.0, p['a_M'])
        self.assertEqual(4000, p['T_eff'])
        self.assertEqual(3.0, p['log_g'])

    def test_read_bin(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'download/models/stellar/grid/roman/grid7/bin/t4000/g_30/t4000g_30f-05a_00.bin.gz')
        r = Grid7SpectrumReader(format='bin', resolution=50000)
        spec = r.read(filename)
