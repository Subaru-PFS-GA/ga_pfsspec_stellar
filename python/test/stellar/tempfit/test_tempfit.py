import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from pfs.ga.pfsspec.core import Spectrum
from pfs.ga.pfsspec.core.obsmod.resampling import FluxConservingResampler
from pfs.ga.pfsspec.stellar.tempfit import TempFit, TempFitTrace

from .tempfittestbase import TempFitTestBase

class TestTempFit(TempFitTestBase):

    def get_tempfit(self,
                    flux_correction=False,
                    use_priors=False,
                    **kwargs):
        
        trace = TempFitTrace()

        tempfit = TempFit(trace=trace, correction_model=None)
        tempfit.mcmc_burnin = 5       # Just a few MCMC steps to make it fast
        tempfit.mcmc_samples = 5
        tempfit.template_resampler = FluxConservingResampler()

        if use_priors:
            tempfit.rv_prior = lambda rv: -(rv - self.rv_real)**2 / 100**2
        else:
            tempfit.rv_prior = None

        return tempfit
    
    def get_dummy_spectrum(self, arm, use_mask=True, all_masked=False):
        # Generate fake spectra for functional testing
        if arm == 'b':
            wave = np.linspace(3000, 6000, 3000)
        elif arm == 'r':
            wave = np.linspace(6000, 9000, 3000)
        flux = np.ones_like(wave)

        if use_mask:
            if all_masked:
                mask = np.full_like(wave, False, dtype=bool)
            else:
                mask = np.full_like(wave, True, dtype=bool)

        spec = Spectrum()
        spec.wave = wave
        spec.flux = flux

        return spec

    def get_dummy_spectra(self, single_exp, single_exp_list, missing_exp, missing_all_exp, missing_arm):
        """
        Generate fake spectra for functional testing

        Parameters
        ----------
        single_exp : bool
            If True, generate a single exposure with two arms
        single_exp_list : bool
            If True, generate a single exposure but returns as a list
        missing_exp : bool
            If True, generate a missing exposure
        missing_all_exp : bool
            If True, generate a missing exposure for all arms
        missing_arm : str
            If not None, generate a missing arm
        """

        if single_exp:
            specs = {
                'b': self.get_dummy_spectrum('b'),
                'r': self.get_dummy_spectrum('r')
            }
        elif single_exp_list:
            specs = {
                'b': [ self.get_dummy_spectrum('b') ],
                'r': [ self.get_dummy_spectrum('r') ]
            }
        elif missing_exp:
            specs = {
                'b': [ self.get_dummy_spectrum('b'), self.get_dummy_spectrum('b') ],
                'r': [ None, self.get_dummy_spectrum('r') ]
            }
        elif missing_all_exp:
            specs = {
                'b': [ self.get_dummy_spectrum('b'), None, None ],
                'r': [ None, self.get_dummy_spectrum('r'), None ]
            }
        elif missing_arm:
            specs = {
                'b': [ None, None ],
                'r': [ self.get_dummy_spectrum('r'), self.get_dummy_spectrum('r') ]
            }
        else:
            specs = {
                'b': [ self.get_dummy_spectrum('b'), self.get_dummy_spectrum('b') ],
                'r': [ self.get_dummy_spectrum('r'), self.get_dummy_spectrum('r') ]
            }

        return specs
    
    def enumerate_spectra_helper(self, tempfit, ground_truth, include_none=False):
        for k1 in ground_truth:
            for k2, gt in ground_truth[k1].items():
                per_arm, per_exp = k1
                single_exp, single_exp_list, missing_exp, missing_all_exp, missing_arm = k2
                
                specs = self.get_dummy_spectra(single_exp, single_exp_list, missing_exp, missing_all_exp, missing_arm)
                spec_list = list(tempfit.enumerate_spectra(specs, per_arm=per_arm, per_exp=per_exp, include_none=include_none))

                self.assertEqual(gt['count'], len(spec_list))
                for i, (arm, ei, mi, spec) in enumerate(spec_list):
                    self.assertEqual(gt['arm'][i], arm)
                    self.assertEqual(gt['ei'][i], ei)
                    self.assertEqual(gt['mi'][i], mi)
    
    def test_enumerate_spectra(self):
        tempfit = self.get_tempfit()

        ground_truth = {
            # per_arm, per_exp
            (True, True,): 
            {
                # All good
                (False, False, False, False, False): {
                    'count': 4,
                    'arm': [ 'b', 'b', 'r', 'r' ],
                    'ei': [ 0, 1, 0, 1 ],
                    'mi': [ 0, 1, 2, 3 ]
                },
                # Single exposure
                (True, False, False, False, False): {
                    'count': 2,
                    'arm': [ 'b', 'r' ],
                    'ei': [ 0, 0 ],
                    'mi': [ 0, 1 ]
                },
                # Single exposure as list
                (False, True, False, False, False): {
                    'count': 2,
                    'arm': [ 'b', 'r' ],
                    'ei': [ 0, 0 ],
                    'mi': [ 0, 1 ]
                },
                # Missing exposure
                (False, False, True, False, False): {
                    'count': 3,
                    'arm': [ 'b', 'b', 'r' ],
                    'ei': [ 0, 1, 1 ],
                    'mi': [ 0, 1, 2 ]
                },
                # Missing all exposures
                (False, False, False, True, False): {
                    'count': 2,
                    'arm': [ 'b', 'r' ],
                    'ei': [ 0, 1 ],
                    'mi': [ 0, 1 ]
                },
                # Missing arm
                (False, False, False, False, True): {
                    'count': 2,
                    'arm': [ 'r', 'r' ],
                    'ei': [ 0, 1 ],
                    'mi': [ 0, 1 ]
                },
            },
            # per_arm, per_exp
            (True, False): 
            {
                # All good
                (False, False, False, False, False): {
                    'count': 4,
                    'arm': [ 'b', 'b', 'r', 'r' ],
                    'ei': [ 0, 1, 0, 1 ],
                    'mi': [ 0, 0, 1, 1 ]
                },
                # Single exposure
                (True, False, False, False, False): {
                    'count': 2,
                    'arm': [ 'b', 'r' ],
                    'ei': [ 0, 0 ],
                    'mi': [ 0, 1 ]
                },
                # Single exposure as list
                (False, True, False, False, False): {
                    'count': 2,
                    'arm': [ 'b', 'r' ],
                    'ei': [ 0, 0 ],
                    'mi': [ 0, 1 ]
                },
                # Missing exposure
                (False, False, True, False, False): {
                    'count': 3,
                    'arm': [ 'b', 'b', 'r' ],
                    'ei': [ 0, 1, 1 ],
                    'mi': [ 0, 0, 1 ]
                },
                # Missing all exposures
                (False, False, False, True, False): {
                    'count': 2,
                    'arm': [ 'b', 'r' ],
                    'ei': [ 0, 1 ],
                    'mi': [ 0, 1 ]
                },
                # Missing arm
                (False, False, False, False, True): {
                    'count': 2,
                    'arm': [ 'r', 'r' ],
                    'ei': [ 0, 1 ],
                    'mi': [ 0, 0 ]
                },
            },
            # per_arm, per_exp
            (False, True): 
            {
                # All good
                (False, False, False, False, False): {
                    'count': 4,
                    'arm': [ 'b', 'b', 'r', 'r' ],
                    'ei': [ 0, 1, 0, 1 ],
                    'mi': [ 0, 1, 0, 1 ]
                },
                # Single exposure
                (True, False, False, False, False): {
                    'count': 2,
                    'arm': [ 'b', 'r' ],
                    'ei': [ 0, 0 ],
                    'mi': [ 0, 0 ]
                },
                # Single exposure as list
                (False, True, False, False, False): {
                    'count': 2,
                    'arm': [ 'b', 'r' ],
                    'ei': [ 0, 0 ],
                    'mi': [ 0, 0 ]
                },
                # Missing exposure
                (False, False, True, False, False): {
                    'count': 3,
                    'arm': [ 'b', 'b', 'r' ],
                    'ei': [ 0, 1, 1 ],
                    'mi': [ 0, 1, 1 ]
                },
                # Missing all exposures
                (False, False, False, True, False): {
                    'count': 2,
                    'arm': [ 'b', 'r' ],
                    'ei': [ 0, 1 ],
                    'mi': [ 0, 1 ]
                },
                # Missing arm
                (False, False, False, False, True): {
                    'count': 2,
                    'arm': [ 'r', 'r' ],
                    'ei': [ 0, 1 ],
                    'mi': [ 0, 1 ]
                },
            },
            # per_arm, per_exp
            (False, False): 
            {
                # All good
                (False, False, False, False, False): {
                    'count': 4,
                    'arm': [ 'b', 'b', 'r', 'r' ],
                    'ei': [ 0, 1, 0, 1 ],
                    'mi': [ 0, 0, 0, 0 ]
                },
                # Single exposure
                (True, False, False, False, False): {
                    'count': 2,
                    'arm': [ 'b', 'r' ],
                    'ei': [ 0, 0 ],
                    'mi': [ 0, 0 ]
                },
                # Single exposure as list
                (False, True, False, False, False): {
                    'count': 2,
                    'arm': [ 'b', 'r' ],
                    'ei': [ 0, 0 ],
                    'mi': [ 0, 0 ]
                },
                # Missing exposure
                (False, False, True, False, False): {
                    'count': 3,
                    'arm': [ 'b', 'b', 'r' ],
                    'ei': [ 0, 1, 1 ],
                    'mi': [ 0, 0, 0 ]
                },
                # Missing all exposures
                (False, False, False, True, False): {
                    'count': 2,
                    'arm': [ 'b', 'r' ],
                    'ei': [ 0, 1 ],
                    'mi': [ 0, 0 ]
                },
                # Missing arm
                (False, False, False, False, True): {
                    'count': 2,
                    'arm': [ 'r', 'r' ],
                    'ei': [ 0, 1 ],
                    'mi': [ 0, 0 ]
                },
            },
        }

        self.enumerate_spectra_helper(tempfit, ground_truth)

    def test_enumerate_spectra_include_none(self):
        tempfit = self.get_tempfit()

        ground_truth = {
            # per_arm, per_exp
            (True, True,): 
            {
                # All good
                (False, False, False, False, False): {
                    'count': 4,
                    'arm': [ 'b', 'b', 'r', 'r' ],
                    'ei': [ 0, 1, 0, 1 ],
                    'mi': [ 0, 1, 2, 3 ]
                },
                # Single exposure
                (True, False, False, False, False): {
                    'count': 2,
                    'arm': [ 'b', 'r' ],
                    'ei': [ 0, 0 ],
                    'mi': [ 0, 1 ]
                },
                # Single exposure as list
                (False, True, False, False, False): {
                    'count': 2,
                    'arm': [ 'b', 'r' ],
                    'ei': [ 0, 0 ],
                    'mi': [ 0, 1 ]
                },
                # Missing exposure
                (False, False, True, False, False): {
                    'count': 4,
                    'arm': [ 'b', 'b', 'r', 'r' ],
                    'ei': [ 0, 1, 0, 1 ],
                    'mi': [ 0, 1, None, 2 ]
                },
                # Missing all exposures
                (False, False, False, True, False): {
                    'count': 6,
                    'arm': [ 'b', 'b', 'b', 'r', 'r', 'r' ],
                    'ei': [ 0, 1, 2, 0, 1, 2 ],
                    'mi': [ 0, None, None, None, 1, None ]
                },
                # Missing arm
                (False, False, False, False, True): {
                    'count': 4,
                    'arm': [ 'b', 'b', 'r', 'r' ],
                    'ei': [ 0, 1, 0, 1 ],
                    'mi': [ None, None, 0, 1 ]
                },
            },
            # per_arm, per_exp
            (True, False): 
            {
                # All good
                (False, False, False, False, False): {
                    'count': 4,
                    'arm': [ 'b', 'b', 'r', 'r' ],
                    'ei': [ 0, 1, 0, 1 ],
                    'mi': [ 0, 0, 1, 1 ]
                },
                # Single exposure
                (True, False, False, False, False): {
                    'count': 2,
                    'arm': [ 'b', 'r' ],
                    'ei': [ 0, 0 ],
                    'mi': [ 0, 1 ]
                },
                # Single exposure as list
                (False, True, False, False, False): {
                    'count': 2,
                    'arm': [ 'b', 'r' ],
                    'ei': [ 0, 0 ],
                    'mi': [ 0, 1 ]
                },
                # Missing exposure
                (False, False, True, False, False): {
                    'count': 4,
                    'arm': [ 'b', 'b', 'r', 'r' ],
                    'ei': [ 0, 1, 0, 1 ],
                    'mi': [ 0, 0, None, 1 ]
                },
                # Missing all exposures
                (False, False, False, True, False): {
                    'count': 6,
                    'arm': [ 'b', 'b', 'b', 'r', 'r', 'r' ],
                    'ei': [ 0, 1, 2, 0, 1, 2 ],
                    'mi': [ 0, None, None, None, 1, None ]
                },
                # Missing arm
                (False, False, False, False, True): {
                    'count': 4,
                    'arm': [ 'b', 'b', 'r', 'r' ],
                    'ei': [ 0, 1, 0, 1 ],
                    'mi': [ None, None, 0, 0 ]
                },
            },
            # per_arm, per_exp
            (False, True): 
            {
                # All good
                (False, False, False, False, False): {
                    'count': 4,
                    'arm': [ 'b', 'b', 'r', 'r' ],
                    'ei': [ 0, 1, 0, 1 ],
                    'mi': [ 0, 1, 0, 1 ]
                },
                # Single exposure
                (True, False, False, False, False): {
                    'count': 2,
                    'arm': [ 'b', 'r' ],
                    'ei': [ 0, 0 ],
                    'mi': [ 0, 0 ]
                },
                # Single exposure as list
                (False, True, False, False, False): {
                    'count': 2,
                    'arm': [ 'b', 'r' ],
                    'ei': [ 0, 0 ],
                    'mi': [ 0, 0 ]
                },
                # Missing exposure
                (False, False, True, False, False): {
                    'count': 4,
                    'arm': [ 'b', 'b', 'r', 'r' ],
                    'ei': [ 0, 1, 0, 1 ],
                    'mi': [ 0, 1, None, 1 ]
                },
                # Missing all exposures
                (False, False, False, True, False): {
                    'count': 6,
                    'arm': [ 'b', 'b', 'b', 'r', 'r', 'r' ],
                    'ei': [ 0, 1, 2, 0, 1, 2 ],
                    'mi': [ 0, None, None, None, 1, None ]
                },
                # Missing arm
                (False, False, False, False, True): {
                    'count': 4,
                    'arm': [ 'b', 'b', 'r', 'r' ],
                    'ei': [ 0, 1, 0, 1 ],
                    'mi': [ None, None, 0, 1 ]
                },
            },
            # per_arm, per_exp
            (False, False): 
            {
                # All good
                (False, False, False, False, False): {
                    'count': 4,
                    'arm': [ 'b', 'b', 'r', 'r' ],
                    'ei': [ 0, 1, 0, 1 ],
                    'mi': [ 0, 0, 0, 0 ]
                },
                # Single exposure
                (True, False, False, False, False): {
                    'count': 2,
                    'arm': [ 'b', 'r' ],
                    'ei': [ 0, 0 ],
                    'mi': [ 0, 0 ]
                },
                # Single exposure as list
                (False, True, False, False, False): {
                    'count': 2,
                    'arm': [ 'b', 'r' ],
                    'ei': [ 0, 0 ],
                    'mi': [ 0, 0 ]
                },
                # Missing exposure
                (False, False, True, False, False): {
                    'count': 4,
                    'arm': [ 'b', 'b', 'r', 'r' ],
                    'ei': [ 0, 1, 0, 1 ],
                    'mi': [ 0, 0, None, 0 ]
                },
                # Missing all exposures
                (False, False, False, True, False): {
                    'count': 6,
                    'arm': [ 'b', 'b', 'b', 'r', 'r', 'r' ],
                    'ei': [ 0, 1, 2, 0, 1, 2 ],
                    'mi': [ 0, None, None, None, 0, None ]
                },
                # Missing arm
                (False, False, False, False, True): {
                    'count': 4,
                    'arm': [ 'b', 'b', 'r', 'r' ],
                    'ei': [ 0, 1, 0, 1 ],
                    'mi': [ None, None, 0, 0 ]
                },
            },
        }

        self.enumerate_spectra_helper(tempfit, ground_truth, include_none=True)
    
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

        # Single model fitted to all arms and exposures
        wlim = tempfit.determine_wlim(specs, rv_bounds=(-500, 500), per_arm=False, per_exp=False)
        self.assertEqual(len(wlim), 1)

        # Single model fitted to all arms, but per exposure
        wlim = tempfit.determine_wlim(specs, rv_bounds=(-500, 500), per_arm=False, per_exp=True)
        self.assertEqual(len(wlim), 2)

        # Different model fitted to each arm but same for all exposures
        wlim = tempfit.determine_wlim(specs, rv_bounds=(-500, 500), per_arm=True, per_exp=False)
        self.assertEqual(len(wlim), 2)

        # Different model fitted to each arm and each exposure
        wlim = tempfit.determine_wlim(specs, rv_bounds=(-500, 500), per_arm=True, per_exp=True)
        self.assertEqual(len(wlim), 4)