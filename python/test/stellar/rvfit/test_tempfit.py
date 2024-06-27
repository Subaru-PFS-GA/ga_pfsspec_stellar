import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from pfs.ga.pfsspec.core.obsmod.resampling import FluxConservingResampler
from pfs.ga.pfsspec.stellar.rvfit import RVFit, TempFitTrace

from .tempfittestbase import TempFitTestBase

class TestTempFit(TempFitTestBase):

    def get_tempfit(self,
                    flux_correction=False,
                    use_priors=False,
                    **kwargs):
        
        trace = TempFitTrace()

        tempfit = RVFit(trace=trace, correction_model=None)
        tempfit.mcmc_burnin = 5       # Just a few MCMC steps to make it fast
        tempfit.mcmc_samples = 5
        tempfit.template_resampler = FluxConservingResampler()

        if use_priors:
            tempfit.rv_prior = lambda rv: -(rv - self.rv_real)**2 / 100**2
        else:
            tempfit.rv_prior = None

        return tempfit
    
    def test_get_param_packing_functions_rv(self):
        tempfit = self.get_tempfit()

        rv_scalar = 100.0                                   # Single RV
        rv_vector = np.array([100.0, 90.0])                 # Vector of RVs

        pack_params, unpack_params, pack_bounds = tempfit._get_param_packing_functions_rv(mode='rv')

        pp = pack_params(rv_scalar)
        rv = unpack_params(pp)
        self.assertEqual((1, 1), pp.shape)
        self.assertEqual(rv_scalar, rv)

        pp = pack_params(rv_vector)
        rv = unpack_params(pp)
        self.assertEqual((2, 1), pp.shape)
        # npt.assert_equal(rv_vector, rv)       # This test is false because rv_vector is actually in the wrong format

    def test_get_param_packing_functions_a(self):
        tempfit = self.get_tempfit()

        a_scalar = np.array(1.0)                            # Single amplitude
        a_vector = np.array([0.9, 0.8])                     # Two amplitudes
        aa_scalar = np.array([[1.0, 0.5, 0.2, 0.1]])        # Vector of four amplitudes
        aa_vector = np.array([[0.9, 0.8, 0.7, 0.6, 0.5], [1.0, 0.5, 0.2, 0.1, 0.0]])

        pack_params, unpack_params, pack_bounds = tempfit._get_param_packing_functions_a(mode='a')

        pp = pack_params(a_scalar)
        a = unpack_params(pp)
        self.assertEqual((1, 1), pp.shape)
        self.assertEqual(a_scalar, a)

        pp = pack_params(a_vector)
        a = unpack_params(pp)
        self.assertEqual((1, 2), pp.shape)
        npt.assert_equal(a_vector, a)

        pp = pack_params(aa_scalar)
        a = unpack_params(pp)
        self.assertEqual((1, 4), pp.shape)
        npt.assert_equal(aa_scalar[0], a)

        pp = pack_params(aa_vector)
        a = unpack_params(pp)
        self.assertEqual((2, 5), pp.shape)
        npt.assert_equal(aa_vector, a)

    def test_get_param_packing_functions(self):
        tempfit = self.get_tempfit()

        rv_scalar = 100.0                                   # Single RV
        rv_vector = np.array([100.0, 90.0])                 # Vector or RVs

        a_scalar = np.array(1.0)                            # Single amplitude
        a_vector = np.array([0.9, 0.8])                     # Two amplitudes
        aa_scalar = np.array([[1.0, 0.5, 0.2, 0.1]])        # Vector of four amplitudes
        aa_vector = np.array([[0.9, 0.8, 0.7, 0.6, 0.5], [1.0, 0.5, 0.2, 0.1, 0.0]])

        pack_params, unpack_params, pack_bounds = tempfit.get_param_packing_functions(mode='a_rv')

        pp = pack_params(a_scalar, rv_scalar)
        a, rv = unpack_params(pp)
        self.assertEqual((1, 2,), pp.shape)
        self.assertEqual(rv_scalar, rv)
        self.assertEqual(a_scalar, a)

        pp = pack_params(aa_vector, rv_vector)
        a, rv = unpack_params(pp)
        self.assertEqual((2, 6), pp.shape)
        npt.assert_equal(rv_vector, rv)
        npt.assert_equal(aa_vector, a)

        pp = pack_params(aa_scalar, rv_scalar)
        a, rv = unpack_params(pp)
        self.assertEqual((1, 5), pp.shape)
        npt.assert_equal(rv_scalar, rv)
        npt.assert_equal(aa_scalar[0], a)

        pp = pack_params(aa_vector, rv_vector)
        a, rv = unpack_params(pp)
        self.assertEqual((2, 6), pp.shape)
        npt.assert_equal(rv_vector, rv)
        npt.assert_equal(aa_vector, a)

        pack_params, unpack_params, pack_bounds = tempfit.get_param_packing_functions(mode='a')

        pp = pack_params(a_scalar)
        a = unpack_params(pp)
        self.assertEqual((1, 1), np.shape(pp))
        self.assertEqual((), np.shape(a))
        npt.assert_equal(a_scalar, a)

        pp = pack_params(aa_scalar)
        a = unpack_params(pp)
        self.assertEqual((1, 4,), pp.shape)
        npt.assert_equal(aa_scalar[0], a)

        pp = pack_params(a_vector)
        a = unpack_params(pp)
        self.assertEqual((1, 2), pp.shape)
        npt.assert_equal(a_vector, a)

        pp = pack_params(aa_vector)
        a = unpack_params(pp)
        self.assertEqual((2, 5), pp.shape)
        npt.assert_equal(aa_vector, a)

        pack_params, unpack_params, pack_bounds = tempfit.get_param_packing_functions(mode='rv')

        pp = pack_params(rv_scalar)
        rv = unpack_params(pp)
        self.assertEqual((1, 1), pp.shape)
        self.assertEqual(rv_scalar, rv)

        pp = pack_params(rv_vector)
        rv = unpack_params(pp)
        self.assertEqual((2, 1), pp.shape)
        # npt.assert_equal(rv_vector, rv)           # This test is false because rv_vector is actually in the wrong format
    
    def test_get_observation(self):
        spec = self.get_observation()
        
        spec.plot(xlim=(7000, 9000))
        self.save_fig()

    def test_get_normalization(self):
        tempfit = self.get_tempfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        spec_norm, temp_norm = tempfit.get_normalization({'mr': spec}, {'mr': temp})

    def test_process_spectrum(self):
        tempfit = self.get_tempfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation()

        sp = tempfit.process_spectrum('', 0, spec)

        tempfit.spec_norm, tempfit.temp_norm = tempfit.get_normalization({'mr': spec}, {'mr': temp})
        sp = tempfit.process_spectrum('', 0, spec)

    def test_process_template(self):
        spec = self.get_observation(arm='mr')
        
        tempfit = self.get_tempfit()
        tempfit.determine_wlim({ 'mr': spec }, (-300, 300))
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        tm = tempfit.process_template('mr', temp, spec, 100)

        psf = self.get_test_psf(arm='mr')
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        tm = tempfit.process_template('mr', temp, spec, 100, psf=psf)

        tempfit.spec_norm, tempfit.temp_norm = tempfit.get_normalization({'mr': spec}, {'mr': temp})
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        tm = tempfit.process_template('mr', temp, spec, 100, psf=psf)

    def test_diff_template(self):
        tempfit = self.get_tempfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)

        wave, dfdl = tempfit.diff_template(temp)
        wave, dfdl = tempfit.diff_template(temp, np.linspace(3000, 9000, 6000))

    def test_log_diff_template(self):
        tempfit = self.get_tempfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)

        wave, dfdl = tempfit.log_diff_template(temp)
        wave, dfdl = tempfit.log_diff_template(temp, np.linspace(3000, 9000, 6000))

    def test_determine_wlim(self):
        tempfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = \
            self.get_initialized_tempfit(
                flux_correction=True,
                normalize=True,
                convolve_template=True,
                multiple_arms=True,
                multiple_exp=True,
                use_priors=False)
        
        # Test different types of freedom
        wlim = tempfit.determine_wlim(specs, rv_bounds=(-500, 500), per_arm=False, per_exp=False)
        self.assertIsInstance(wlim, tuple)

        wlim = tempfit.determine_wlim(specs, rv_bounds=(-500, 500), per_arm=False, per_exp=True)
        self.assertIsInstance(wlim, list)
        self.assertIsInstance(wlim[0], tuple)

        wlim = tempfit.determine_wlim(specs, rv_bounds=(-500, 500), per_arm=True, per_exp=False)
        self.assertIsInstance(wlim, dict)
        self.assertIsInstance(wlim['b'], tuple)

        wlim = tempfit.determine_wlim(specs, rv_bounds=(-500, 500), per_arm=True, per_exp=True)
        self.assertIsInstance(wlim, dict)
        self.assertIsInstance(wlim['b'], list)
        self.assertIsInstance(wlim['b'][0], tuple)