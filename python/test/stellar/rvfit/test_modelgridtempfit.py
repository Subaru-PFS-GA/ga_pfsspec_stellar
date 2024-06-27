import os
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from pfs.ga.pfsspec.core.sampling import Parameter, NormalDistribution, UniformDistribution
from pfs.ga.pfsspec.core.obsmod.resampling import FluxConservingResampler
from pfs.ga.pfsspec.stellar.rvfit import ModelGridRVFit, ModelGridRVFitTrace

from .tempfittestbase import TempFitTestBase

class TestModelGridTempFit(TempFitTestBase):
    
    def get_tempfit(self,
                    flux_correction=False,
                    use_priors=False,
                    **kwargs):
        
        trace = ModelGridRVFitTrace()

        tempfit = ModelGridRVFit(trace=trace, correction_model=None)
        
        tempfit.mcmc_burnin = 5       # Just a few MCMC steps to make it fast
        tempfit.mcmc_samples = 5
        tempfit.params_0 = kwargs
        tempfit.template_resampler = FluxConservingResampler()
        tempfit.template_grids = {
            'b': self.get_bosz_grid(),
            'mr': self.get_bosz_grid()
        }

        if use_priors:
            tempfit.rv_prior = lambda rv: -(rv - self.rv_real)**2 / 100**2
            tempfit.params_priors = {
                'T_eff': lambda T_eff: -(T_eff - 4500)**2 / 300**2
            }
        else:
            tempfit.rv_prior = None
            tempfit.params_priors = None

        tempfit.params_fixed = {
            'C_M': 0,
            'a_M': 0
        }

        return tempfit
    
    def test_get_normalization(self):
        rvfit = self.get_tempfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        # Provide the templates
        spec_norm, temp_norm = rvfit.get_normalization({'mr': spec}, {'mr': temp})

        # Provide the template parameters
        spec_norm, temp_norm = rvfit.get_normalization({'mr': spec}, M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
    
    def test_get_param_packing_functions_params(self):
        tempfit = self.get_tempfit()

        params_scalar = { 'T_eff': 5000.0, 'M_H': -1.0, 'log_g': 1.0 }
        params_vector = { 'T_eff': np.array([5000.0, 5200.0]), 'M_H': np.array([-1.0, -2.0]), 'log_g': np.array([1.0, 1.5]) }
        params_free = [ 'T_eff', 'M_H', 'log_g' ]
        params_fixed = { 'a_M': 0.0 }

        pack_params, unpack_params, pack_bounds = tempfit._get_param_packing_functions_params(params_free, mode='params')

        pp = pack_params(params_scalar)
        params = unpack_params(pp)
        self.assertEqual((1, 3), pp.shape)
        self.assertEqual(params_scalar, params)

        pp = pack_params(params_vector)
        params = unpack_params(pp)
        self.assertEqual((2, 3), pp.shape)
        for k in params:
            npt.assert_equal(params_vector[k], params[k])

    def test_get_param_packing_functions(self):
        tempfit = self.get_tempfit()

        rv_scalar = 100.0                                   # Single RV
        rv_vector = np.array([100.0, 90.0])                 # Vector or RVs

        a_scalar = np.array(1.0)                            # Single amplitude
        a_vector = np.array([0.9, 0.8])                     # Two amplitudes
        aa_scalar = np.array([[1.0, 0.5, 0.2, 0.1]])        # Vector of four amplitudes
        aa_vector = np.array([[0.9, 0.8, 0.7, 0.6, 0.5], [1.0, 0.5, 0.2, 0.1, 0.0]])

        params_scalar = { 'T_eff': 5000.0, 'M_H': -1.0, 'log_g': 1.0 }
        params_vector = { 'T_eff': np.array([5000.0, 5200.0]), 'M_H': np.array([-1.0, -2.0]), 'log_g': np.array([1.0, 1.5]) }
        params_free = [ 'T_eff', 'M_H', 'log_g' ]
        params_fixed = { 'a_M': 0.0 }

        pack_params, unpack_params, pack_bounds = tempfit.get_param_packing_functions(params_free=params_free, mode='a_params_rv')

        pp = pack_params(a_scalar, params_scalar, rv_scalar)
        a, params, rv = unpack_params(pp)
        self.assertEqual((1, 5), pp.shape)
        self.assertEqual(rv_scalar, rv)
        for k in params:
            self.assertEqual(params_scalar[k], params[k])
        self.assertEqual(a_scalar, a)

        pp = pack_params(aa_vector, params_vector, rv_vector)
        a, params, rv = unpack_params(pp)
        self.assertEqual((2, 9), pp.shape)
        npt.assert_equal(rv_vector, rv)
        for k in params:
            npt.assert_equal(params_vector[k], params[k])
        npt.assert_equal(aa_vector, a)

        pp = pack_params(aa_scalar, params_scalar, rv_scalar)
        a, params, rv = unpack_params(pp)
        self.assertEqual((1, 8), pp.shape)
        npt.assert_equal(rv_scalar, rv)
        for k in params:
            npt.assert_equal(params_scalar[k], params[k])
        npt.assert_equal(aa_scalar[0], a)
