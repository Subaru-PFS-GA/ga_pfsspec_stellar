import os

from test.core import TestBase
from pfsspec.core import Filter
from pfsspec.core.grid import ArrayGrid
from pfsspec.core.grid import RbfGrid
from pfsspec.stellar.grid import ModelGrid
from pfsspec.stellar.grid.bosz import Bosz
from pfsspec.stellar.grid.phoenix import Phoenix

class TestPhoenixSpectrum(TestBase):
    
    def test_synthmag_carrie(self):
        filter_hsc_i = Filter()
        filter_hsc_i.read(os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/hsc/filters', 'wHSC-i.txt'))

        fn = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/phoenix/phoenix_HiRes_RGB_rbf/flux', 'spectra.h5')
        phoenix = ModelGrid(Phoenix(pca=False, normalized=False), RbfGrid)
        phoenix.preload_arrays = False
        phoenix.load(fn, format='h5')

        #spec = phoenix.get_model(M_H=-0.5, T_eff=3289.274329992684, log_g=5.307799816131592)
        spec = phoenix.get_model(M_H=-0.5, T_eff=4870.274329992684, log_g=5.307799816131592)
        mag_hsc_i_pho = spec.synthmag_carrie(filter_hsc_i, -2.808000087738037)

        self.assertNotEqual(0, mag_hsc_i_pho)

    def test_normalize_to_mag(self):
        filter_hsc_i = Filter()
        filter_hsc_i.read(os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/hsc/filters', 'wHSC-i.txt'))

        fn = os.path.join(self.PFSSPEC_DATA_PATH, '/datascope/subaru/data/pfsspec/models/stellar/grid/phoenix/phoenix_HiRes', 'spectra.h5')
        phoenix = ModelGrid(Phoenix(pca=False, normalized=False), ArrayGrid)
        phoenix.preload_arrays = False
        phoenix.load(fn, format='h5')

        spec = phoenix.get_nearest_model(M_H=-0.5, T_eff=3289.274329992684, log_g=5.307799816131592, a_M=0)
        spec.normalize_to_mag(filter_hsc_i, 23)

    def test_interpolate_model_linear(self):
        fn = os.path.join(self.PFSSPEC_DATA_PATH, '/datascope/subaru/data/pfsspec/models/stellar/grid/phoenix/phoenix_HiRes', 'spectra.h5')
        phoenix = ModelGrid(Phoenix(pca=False, normalized=False), ArrayGrid)
        phoenix.preload_arrays = False
        phoenix.load(fn, format='h5')

        spec = phoenix.interpolate_model_linear(M_H=-0.5, T_eff=3289.274329992684, log_g=5.307799816131592, a_M=0)

        pass

    def test_get_nearest_model(self):
        fn = os.path.join(self.PFSSPEC_DATA_PATH, '/datascope/subaru/data/pfsspec/models/stellar/grid/phoenix/phoenix_HiRes', 'spectra.h5')
        phoenix = ModelGrid(Phoenix(pca=False, normalized=False), ArrayGrid)
        phoenix.preload_arrays = False
        phoenix.load(fn, format='h5')

        spec = phoenix.get_nearest_model(M_H=-0.5, T_eff=3289.274329992684, log_g=5.307799816131592, a_M=0)

        pass