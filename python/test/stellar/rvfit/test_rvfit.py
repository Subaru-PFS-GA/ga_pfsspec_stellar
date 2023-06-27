import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from pfs.ga.pfsspec.core.obsmod.resampling import FluxConservingResampler
from pfs.ga.pfsspec.stellar.rvfit import RVFit, RVFitTrace
from pfs.ga.pfsspec.core.obsmod.fluxcorr import PolynomialFluxCorrection
from pfs.ga.pfsspec.core.sampling import Parameter

from .rvfittestbase import RVFitTestBase

class TestRVFit(RVFitTestBase):
    
    def get_rvfit(self, flux_correction=False, use_priors=False, **kwargs):
        trace = RVFitTrace()
        rvfit = RVFit(trace=trace)
        rvfit.mcmc_burnin = 5       # Just a few MCMC steps to make it fast
        rvfit.mcmc_samples = 5
        rvfit.template_resampler = FluxConservingResampler()

        if flux_correction:
            rvfit.use_flux_corr = True
            rvfit.flux_corr = PolynomialFluxCorrection()

        if use_priors:
            rvfit.rv_prior = lambda rv: -(rv - self.rv_real)**2 / 100**2
        else:
            rvfit.rv_prior = None

        return rvfit
    
    def test_get_packing_functions(self):
        rvfit = self.get_rvfit()

        a_scalar = np.array(1.0)
        a_vector = np.array([0.9, 0.8]).T
        aa_scalar = np.array([1.0, 0.5, 0.2, 0.1])
        aa_vector = np.array([[0.9, 0.8, 0.7, 0.6, 0.5], [1.0, 0.5, 0.2, 0.1, 0.0]]).T
        rv_scalar = 100.0
        rv_vector = np.array([100.0, 90.0])

        pack_params, unpack_params, pack_bounds = rvfit.get_packing_functions(mode='a_rv')
        
        pp = pack_params(a_scalar, rv_scalar)
        a, rv = unpack_params(pp)
        self.assertEqual((2,), pp.shape)
        npt.assert_equal(a_scalar, a)
        self.assertEqual(rv_scalar, rv)

        pp = pack_params(aa_scalar, rv_scalar)
        a, rv = unpack_params(pp)
        self.assertEqual((5,), pp.shape)
        npt.assert_equal(aa_scalar, a)
        self.assertEqual(rv_scalar, rv)

        pp = pack_params(a_vector, rv_vector)
        a, rv = unpack_params(pp)
        self.assertEqual((2, 2), pp.shape)
        npt.assert_equal(a_vector, a)
        npt.assert_equal(rv_vector, rv)

        pp = pack_params(aa_vector, rv_vector)
        a, rv = unpack_params(pp)
        self.assertEqual((6, 2), pp.shape)
        npt.assert_equal(aa_vector, a)
        npt.assert_equal(rv_vector, rv)

        pack_params, unpack_params, pack_bounds = rvfit.get_packing_functions(mode='rv')

        pp = pack_params(rv_scalar)
        rv = unpack_params(pp)
        self.assertEqual((1,), pp.shape)
        self.assertEqual(rv_scalar, rv)

        pp = pack_params(rv_vector)
        rv = unpack_params(pp)
        self.assertEqual((2,), pp.shape)
        npt.assert_equal(rv_vector, rv)
    
    def test_get_observation(self):
        spec = self.get_observation()
        
        spec.plot(xlim=(7000, 9000))
        self.save_fig()

    def test_get_normalization(self):
        rvfit = self.get_rvfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        spec_norm, temp_norm = rvfit.get_normalization({'mr': spec}, {'mr': temp})

    def test_process_spectrum(self):
        rvfit = self.get_rvfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation()

        sp = rvfit.process_spectrum('', 0, spec)

        rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization({'mr': spec}, {'mr': temp})
        sp = rvfit.process_spectrum('', 0, spec)

    def test_process_template(self):
        spec = self.get_observation(arm='mr')
        
        rvfit = self.get_rvfit()
        rvfit.determine_wlim({ 'mr': spec }, (-300, 300))
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        tm = rvfit.process_template('mr', temp, spec, 100)

        psf = self.get_test_psf(arm='mr')
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        tm = rvfit.process_template('mr', temp, spec, 100, psf=psf)

        rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization({'mr': spec}, {'mr': temp})
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        tm = rvfit.process_template('mr', temp, spec, 100, psf=psf)

    def test_diff_template(self):
        rvfit = self.get_rvfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)

        wave, dfdl = rvfit.diff_template(temp)
        wave, dfdl = rvfit.diff_template(temp, np.linspace(3000, 9000, 6000))

    def test_log_diff_template(self):
        rvfit = self.get_rvfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)

        wave, dfdl = rvfit.log_diff_template(temp)
        wave, dfdl = rvfit.log_diff_template(temp, np.linspace(3000, 9000, 6000))

    def rvfit_test_helper(self, ax, flux_correction, normalize, convolve_template, multiple_arms, multiple_exp, use_priors=False, calculate_log_L=False, fit_lorentz=False, guess_rv=False, fit_rv=False, calculate_error=False):
        rvfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = self.get_initialized_rvfit(flux_correction, normalize, convolve_template, multiple_arms, multiple_exp, use_priors)

        ax.axvline(rv_real, color='r', label='rv real')

        if calculate_log_L:
            def calculate_log_L_helper(rv, shape):
                log_L, phi, chi, ndf = rvfit.calculate_log_L(specs, temps, rv)
                a = rvfit.eval_a(phi, chi)

                self.assertEqual(shape, log_L.shape)
                self.assertEqual(shape + phi_shape, phi.shape)
                self.assertEqual(shape + chi_shape, chi.shape)

                ax.plot(rv, log_L, 'o')

                log_L, _, _, _ = rvfit.calculate_log_L(specs, temps, rv, a=a)
        
            # Test with scalar
            calculate_log_L_helper(100, ())
            
            # Test with vector
            rv = np.linspace(-300, 300, 31)
            calculate_log_L_helper(rv, rv.shape)

        if fit_lorentz or guess_rv or fit_rv or calculate_error:
            rv = np.linspace(-300, 300, 31)
            log_L, phi, chi, ndf = rvfit.calculate_log_L(specs, temps, rv)
            pp, _ = rvfit.fit_lorentz(rv, log_L)

            y1 = rvfit.lorentz(rv, *pp)

            ax.plot(rv, log_L, 'o')
            ax.plot(rv, y1, '-')
  
        if guess_rv or fit_rv or calculate_error:
            _, _, _, rv0 = rvfit.guess_rv(specs, temps)
            ax.axvline(rv0, color='k', label='rv guess')

    def test_eval_flux_corr_basis(self):
        rvfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = self.get_initialized_rvfit(True, True, True, True, True, False)

        f, ax = plt.subplots(1, 1)
        
        basis, basis_size = rvfit.eval_flux_corr_basis(specs)
        for k in specs:
            for ei, ee in enumerate(specs[k] if isinstance(specs[k], list) else [specs[k]]):
                ax.plot(ee.wave, basis[k][ei])

        self.save_fig(f)

    def test_calculate_log_L(self):
        configs = [
            dict(flux_correction=False, use_priors=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(flux_correction=True, use_priors=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(flux_correction=True, use_priors=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        ]

        f, ax = plt.subplots(1, 1)

        for config in configs:
            self.rvfit_test_helper(ax, **config, calculate_log_L=True)

        self.save_fig(f)
        
    def test_fit_lorentz(self):
        configs = [
            dict(flux_correction=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        ]

        f, axs = plt.subplots(1, len(configs), squeeze=False)

        for ax, config in zip(axs[0], configs):
            self.rvfit_test_helper(ax, **config, fit_lorentz=True)

        self.save_fig(f)

    def test_guess_rv(self):
        configs = [
            dict(flux_correction=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        ]

        f, axs = plt.subplots(1, len(configs), squeeze=False)

        for ax, config in zip(axs[0], configs):
            self.rvfit_test_helper(ax, **configs[0], guess_rv=True)

        self.save_fig(f)

    def rvfit_fit_rv_test_helper(self, ax, flux_correction, normalize, convolve_template, multiple_arms, multiple_exp, use_priors):
        rvfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = self.get_initialized_rvfit(flux_correction, normalize, convolve_template, multiple_arms, multiple_exp, use_priors)
        res = rvfit.fit_rv(specs, temps)

        ax.axvline(rv_real, color='r', label='rv real')
        ax.axvline(res.rv_fit, color='b', label='rv fit')
        ax.axvline(res.rv_fit - res.rv_err, color='b')
        ax.axvline(res.rv_fit + res.rv_err, color='b')

        # rvv = np.linspace(rv_real - 10 * rv_err, rv_real + 10 * rv_err, 101)
        rvv = np.linspace(res.rv_fit - 0.001, res.rv_fit + 0.001, 101)
        log_L, phi, chi, ndf = rvfit.calculate_log_L(specs, temps, rvv)
        if rvfit.rv_prior is not None:
            log_L += rvfit.rv_prior(rvv)
        ax.plot(rvv, log_L, '.')
        ax.set_xlim(rvv[0], rvv[-1])

        ax.set_title(f'RV={rv_real:.2f}, RF_fit={res.rv_fit:.3f}+/-{res.rv_err:.3f}')
        # ax.set_xlim(rv_real - 50 * rv_err, rv_real + 50 * rv_err)

    def test_fit_rv(self):
        configs = [
            dict(flux_correction=False, use_priors=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(flux_correction=True, use_priors=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(flux_correction=True, use_priors=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        ]

        f, axs = plt.subplots(len(configs), 1, figsize=(6, 4 * len(configs)), squeeze=False)

        for ax, config in zip(axs[:, 0], configs):
            self.rvfit_fit_rv_test_helper(ax, **config)

        self.save_fig(f)

    def rvfit_run_mcmc_test_helper(self, ax, flux_correction, normalize, convolve_template, multiple_arms, multiple_exp, use_priors):
        rvfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = self.get_initialized_rvfit(flux_correction, normalize, convolve_template, multiple_arms, multiple_exp, use_priors)

        ax.axvline(rv_real, color='r', label='rv real')

        rvfit.mcmc_burnin = 5
        rvfit.mcmc_samples = 5
        res = rvfit.run_mcmc(specs, temps,
                                  rv_0=rv_real + 10,
                                  rv_bounds=(rv_real - 100, rv_real + 100),
                                  rv_step=2)

        ax.hist(res.rv_mcmc)

    def test_run_mcmc(self):
        configs = [
            dict(flux_correction=False, use_priors=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(flux_correction=True, use_priors=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(flux_correction=True, use_priors=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        ]

        f, axs = plt.subplots(len(configs), 1, figsize=(6, 4 * len(configs)), squeeze=False)

        for ax, config in zip(axs[:, 0], configs):
            self.rvfit_run_mcmc_test_helper(ax, **config)

        self.save_fig(f)

    def test_calculate_fisher(self):
        configs = [
            dict(flux_correction=False, use_priors=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(flux_correction=True, use_priors=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(flux_correction=True, use_priors=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        ]

        f, axs = plt.subplots(len(configs), 1, squeeze=False)

        for ax, config in zip(axs[:, 0], configs):
            rvfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = self.get_initialized_rvfit(**config)
            res = rvfit.fit_rv(specs, temps)
            F = {}
            C = {}
            err = {}
            for mode, method in zip([
                    'full',
                    'rv',
                    'full',
                    'rv',
                    'full',
                    'full'
                ], [
                    'hessian',
                    'hessian',
                    'emcee',
                    'sampling',
                    'phi_chi',
                    'alex'
                ]):
                FF, CC = rvfit.calculate_F(specs, temps, res.rv_fit, mode=mode, method=method)
                F[f'{mode}_{method}'] = FF
                C[f'{mode}_{method}'] = CC
                err[f'{mode}_{method}'] = np.sqrt(CC[-1, -1])

            pass

        self.save_fig(f)

    def test_calculate_rv_bouchy(self):
        configs = [
            dict(flux_correction=False, use_priors=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
        ]

        rvfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = self.get_initialized_rvfit(**configs[0])
        rvfit.calculate_rv_bouchy(specs, temps, rv_real)

# Run when profiling

# $ python -m cProfile -o tmp/tmp.prof python/test/pfs/ga/pfsspec/stellar/rvfit/test_rvfit.py
# $ python -m snakeviz --server tmp/tmp.prof

# t = TestRVFit()
# t.setUpClass()
# t.setUp()
# t.test_fit_rv()