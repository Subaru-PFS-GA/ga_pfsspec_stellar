import os

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core import Filter
from pfs.ga.pfsspec.core.grid import ArrayGrid
from pfs.ga.pfsspec.core.grid import RbfGrid
from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid.bosz import Bosz

class TestBoszSpectrum(TestBase):
    
    def test_synthmag_carrie(self):
        filter_hsc_i = Filter()
        filter_hsc_i.read(os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/hsc/filters', 'fHSC-i.txt'))

        fn = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/bosz/bosz_5000_aM0_CM0_rbf/flux', 'spectra.h5')
        bosz = ModelGrid(Bosz(pca=False, normalized=False), RbfGrid)
        bosz.preload_arrays = False
        bosz.load(fn, format='h5')

        #spec = bosz.get_model(Fe_H=-0.5, T_eff=3289.274329992684, log_g=5.307799816131592)
        spec = bosz.get_model(M_H=-0.5, T_eff=3589.274329992684, log_g=4.307799816131592)
        mag_hsc_i_bosz = spec.synthmag_carrie(filter_hsc_i, -2.808000087738037)

        self.assertNotEqual(0, mag_hsc_i_bosz)

    def test_extrapolate(self):
        fn = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/bosz/bosz_5000_aM0_CM0_rbf/flux', 'spectra.h5')
        bosz = ModelGrid(Bosz(pca=False, normalized=False), RbfGrid)
        bosz.preload_arrays = False
        bosz.load(fn, format='h5')

        spec = bosz.get_model(M_H=-0.5, T_eff=3289.274329992684, log_g=5.307799816131592, a_M=0, C_M=0)
        self.assertIsNone(spec)

        spec = bosz.get_model(M_H=-0.5, T_eff=3589.274329992684, log_g=4.307799816131592, a_M=0, C_M=0)
        self.assertIsNotNone(spec)