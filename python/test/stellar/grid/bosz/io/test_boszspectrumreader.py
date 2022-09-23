import os

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core.grid import ArrayGrid
from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid.bosz.io import BoszSpectrumReader

class TestBoszSpectrumReader(TestBase):
    def test_get_filename(self):
        reader = BoszSpectrumReader(format='fits', resolution=5000)
        fn = reader.get_filename(M_H=-0.5, T_eff=4000, log_g=3, a_M=0, C_M=0)
        self.assertEqual("amm05cp00op00t4000g30v20modrt0b5000rs.fits", fn)

    def test_get_url(self):
        reader = BoszSpectrumReader(format='fits', resolution=5000)
        url = reader.get_url(M_H=-0.5, T_eff=4000, log_g=3, a_M=0, C_M=0)
        self.assertEqual("https://archive.stsci.edu/missions/hlsp/bosz/fits/insbroad_005000/metal_-0.50/carbon_+0.00/alpha_+0.00/amm05cp00op00t4000g30v20modrt0b5000rs.fits", url)

    def test_parse_filename(self):
        fn = 'amm03cm03om03t3500g25v20modrt0b5000rs.asc.bz2'
        p = BoszSpectrumReader.parse_filename(fn)
        self.assertEqual(-0.25, p['M_H'])
        self.assertEqual(-0.25, p['C_M'])
        self.assertEqual(-0.25, p['a_M'])
        self.assertEqual(3500, p['T_eff'])
        self.assertEqual(2.5, p['log_g'])

    def test_read(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'download/models/stellar/grid/bosz/bosz_5000/amm03cm03om03t3500g25v20modrt0b5000rs.asc')
        r = BoszSpectrumReader(filename, format='ascii', resolution=5000)
        spec = r.read()

    def test_read_bz2(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'download/models/stellar/grid/bosz/bosz_5000/amm03cm03om03t3500g25v20modrt0b5000rs.asc.bz2')
        r = BoszSpectrumReader(filename, format='ascii', resolution=5000)
        spec = r.read()

    def test_read_fits(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'download/models/stellar/grid/bosz/bosz_50000_fits/amm03cm03om03t3500g25v20modrt0b50000rs.fits')
        r = BoszSpectrumReader(filename, format='fits', resolution=50000)
        spec = r.read()