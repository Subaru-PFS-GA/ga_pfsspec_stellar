import os

from test.pfs.ga.pfsspec.core import TestBase

from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid.phoenix.io import PhoenixSpectrumReader

class TestPhoenixSpectrumReader(TestBase):
    def test_read(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'download/models/stellar/grid/phoenix/phoenix_HiRes/Z-0.0/lte11600-2.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')

        r = PhoenixSpectrumReader(filename)
        spec = r.read()
        self.assertEqual(spec.wave.shape, spec.flux.shape)

        r = PhoenixSpectrumReader(filename, wave_lim=(4000, 6000))
        spec = r.read()
        self.assertEqual(spec.wave.shape, spec.flux.shape)

    def test_get_filename(self):
        fn = PhoenixSpectrumReader().get_filename(M_H=-2, T_eff=9600, log_g=4, a_M=0.0)
        self.assertEqual('Z-2.0/lte09600-4.00-2.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits', fn)

    def test_parse_filename(self):
        fn = 'lte09600-4.00-2.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
        p = PhoenixSpectrumReader.parse_filename(fn)
        self.assertEqual(-2.0, p['M_H'])
        self.assertEqual(9600, p['T_eff'])
        self.assertEqual(4.0, p['log_g'])
