import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from pfs.ga.pfsspec.core.obsmod.resampling import FluxConservingResampler
from pfs.ga.pfsspec.stellar.tempfit import TempFit, TempFitTrace
from pfs.ga.pfsspec.stellar.tempfit import FluxCorr
from pfs.ga.pfsspec.stellar.fluxcorr import PolynomialFluxCorrection

from .tempfittestbase import TempFitTestBase

class TestTempFitFluxCorr(TempFitTestBase):
    # RV fitting with flux correction
    
    def get_tempfit(self,
                    flux_correction=False,
                    use_priors=False,
                    **kwargs):

        trace = TempFitTrace()
        
        fluxcorr = FluxCorr()
        if flux_correction:
            fluxcorr.use_flux_corr = True
            fluxcorr.flux_corr_type = PolynomialFluxCorrection
        
        tempfit = TempFit(trace=trace, correction_model=fluxcorr)
        tempfit.mcmc_burnin = 5       # Just a few MCMC steps to make it fast
        tempfit.mcmc_samples = 5
        tempfit.template_resampler = FluxConservingResampler()

        if use_priors:
            tempfit.rv_prior = lambda rv: -(rv - self.rv_real)**2 / 100**2
        else:
            tempfit.rv_prior = None

        return tempfit

    def tempfit_test_helper(self, ax, flux_correction, normalize, convolve_template, multiple_arms, multiple_exp, use_priors=False, calculate_log_L=False, fit_lorentz=False, guess_rv=False, fit_rv=False, calculate_error=False):
        (tempfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0) = \
            self.get_initialized_tempfit(
                flux_correction=flux_correction,
                normalize=normalize,
                convolve_template=convolve_template,
                multiple_arms=multiple_arms,
                multiple_exp=multiple_exp,
                use_priors=use_priors)
        
        tempfit.init_correction_models(specs, rv_bounds=(-500, 500), force=True)

        ax.axvline(rv_real, color='r', label='rv real')

        if calculate_log_L:
            def calculate_log_L_helper(rv):
                if np.shape(rv) == ():
                    # Tests specific to FluxCorr correction model
                    pp_specs = tempfit.preprocess_spectra(specs)
                    pp_temps = tempfit.preprocess_templates(specs, temps, rv)
                    log_L, phi, chi, ndf = tempfit.correction_model.eval_log_L(pp_specs, pp_temps, return_phi_chi=True)
                    a = tempfit.correction_model.eval_a(phi, chi)

                    ax.plot(rv, log_L, 'o')

                    log_L = tempfit.correction_model.eval_log_L(pp_specs, pp_temps, a=a)

                tempfit.calculate_log_L(specs, temps, rv)
        
            # Test with scalar
            rv = 100
            calculate_log_L_helper(rv)
            
            # Test with vector
            rv = np.linspace(-300, 300, 31)
            calculate_log_L_helper(rv)

        if fit_lorentz or guess_rv or fit_rv or calculate_error:
            rv = np.linspace(-300, 300, 31)
            log_L = tempfit.calculate_log_L(specs, temps, rv)
            pp, _ = tempfit.fit_lorentz(rv, log_L)

            y1 = tempfit.lorentz(rv, *pp)

            ax.plot(rv, log_L, 'o')
            ax.plot(rv, y1, '-')
  
        if guess_rv or fit_rv or calculate_error:
            _, _, rv0, lL0 = tempfit.guess_rv(specs, temps)
            ax.axvline(rv0, color='k', label='rv guess')

    def test_init_correction_model(self):
        tempfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = \
            self.get_initialized_tempfit(
                flux_correction=True,
                normalize=True,
                convolve_template=True,
                multiple_arms=True,
                multiple_exp=True,
                use_priors=False)
        
        # Test different types of freedom
        models = tempfit.init_correction_model(specs, rv_bounds=(-500, 500), per_arm=False, per_exp=False,
                                              create_model_func=tempfit.correction_model.create_flux_corr)
        self.assertEqual(1, len(models))
        self.assertIsInstance(models[0], PolynomialFluxCorrection)

        models = tempfit.init_correction_model(specs, rv_bounds=(-500, 500), per_arm=False, per_exp=True,
                                              create_model_func=tempfit.correction_model.create_flux_corr)
        self.assertEqual(2, len(models))
        self.assertIsInstance(models[0], PolynomialFluxCorrection)

        models = tempfit.init_correction_model(specs, rv_bounds=(-500, 500), per_arm=True, per_exp=False,
                                              create_model_func=tempfit.correction_model.create_flux_corr)
        self.assertEqual(2, len(models))
        self.assertIsInstance(models[0], PolynomialFluxCorrection)

        models = tempfit.init_correction_model(specs, rv_bounds=(-500, 500), per_arm=True, per_exp=True,
                                              create_model_func=tempfit.correction_model.create_flux_corr)
        self.assertEqual(4, len(models))
        self.assertIsInstance(models[0], PolynomialFluxCorrection)

    def test_get_coeff_count(self):
        tempfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = \
            self.get_initialized_tempfit(
                flux_correction=True,
                normalize=True,
                convolve_template=True,
                multiple_arms=True,
                multiple_exp=True,
                use_priors=False)

        # arms: 2
        # exposures: 2
        # degree = 5

        gt = [
            [ False, False, False, False, 1, 5 ],
            [ False, False, False, True, 1, 10 ],
            [ False, False, True, False, 1, 10 ],
            [ False, False, True, True, 1, 20 ],
            [ False, True, False, False, 2, 5 ],
            [ False, True, False, True, 2, 10 ],
            [ False, True, True, False, 2, 10 ],
            [ False, True, True, True, 2, 20 ],
            [ True, False, False, False, 2, 5 ],
            [ True, False, False, True, 2, 10 ],
            [ True, False, True, False, 2, 10 ],
            [ True, False, True, True, 2, 20 ],
            [ True, True, False, False, 4, 5 ],
            [ True, True, False, True, 4, 10 ],
            [ True, True, True, False, 4, 10 ],
            [ True, True, True, True, 4, 20 ],
        ]

        # Test different types of freedom
        for i, [tempfit.amplitude_per_arm, tempfit.amplitude_per_exp,
                tempfit.correction_model.flux_corr_per_arm, tempfit.correction_model.flux_corr_per_exp,
                gt_amp_count, gt_coeff_count] in enumerate(gt):
            
            tempfit.init_correction_models(specs, rv_bounds=(-500, 500), force=True)
            amp_count = tempfit.get_amp_count(specs)
            coeff_count = tempfit.correction_model.get_coeff_count()

            self.assertEqual(gt_amp_count, amp_count)
            self.assertEqual(gt_coeff_count, coeff_count)

    def test_get_eval_flux_corr_basis(self):
        tempfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = \
            self.get_initialized_tempfit(
                flux_correction=True,
                normalize=True,
                convolve_template=True,
                multiple_arms=True,
                multiple_exp=True,
                use_priors=False)

        # arms: 2
        # exposures: 2
        # degree = 5

        gt = [
            [ False, False, False, False, 1, 5 ],
            [ False, False, False, True, 1, 10 ],
            [ False, False, True, False, 1, 10 ],
            [ False, False, True, True, 1, 20 ],
            [ False, True, False, False, 2, 5 ],
            [ False, True, False, True, 2, 10 ],
            [ False, True, True, False, 2, 10 ],
            [ False, True, True, True, 2, 20 ],
            [ True, False, False, False, 2, 5 ],
            [ True, False, False, True, 2, 10 ],
            [ True, False, True, False, 2, 10 ],
            [ True, False, True, True, 2, 20 ],
            [ True, True, False, False, 4, 5 ],
            [ True, True, False, True, 4, 10 ],
            [ True, True, True, False, 4, 10 ],
            [ True, True, True, True, 4, 20 ],
        ]

        # Test different types of freedom
        for i, [tempfit.amplitude_per_arm, tempfit.amplitude_per_exp,
                tempfit.correction_model.flux_corr_per_arm, tempfit.correction_model.flux_corr_per_exp,
                gt_amp_count, gt_coeff_count] in enumerate(gt):
            
            tempfit.init_correction_models(specs, rv_bounds=(-500, 500), force=True)

            basis, basis_size, model_mask, wave_mask = tempfit.correction_model.eval_flux_corr_basis(specs)

            self.assertEqual(gt_amp_count + gt_coeff_count, basis_size)
            for arm in basis:
                for b in basis[arm]:
                    self.assertEqual(b.ndim, 2)
                    self.assertEqual(b.shape[-1], gt_amp_count + gt_coeff_count)

    def test_eval_flux_corr_basis_with_plot(self):
        tempfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = \
            self.get_initialized_tempfit(
                flux_correction=True,
                normalize=True,
                convolve_template=True,
                multiple_arms=True,
                multiple_exp=True,
                use_priors=False
            )
        tempfit.init_correction_models(specs, rv_bounds=(-500, 500), force=True)

        f, ax = plt.subplots(1, 1)
        
        basis, basis_size, model_mask, wave_mask = tempfit.correction_model.eval_flux_corr_basis(specs)
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
            self.tempfit_test_helper(ax, **config, calculate_log_L=True)

        self.save_fig(f)
        
    def test_fit_lorentz(self):
        configs = [
            dict(flux_correction=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        ]

        f, axs = plt.subplots(1, len(configs), squeeze=False)

        for ax, config in zip(axs[0], configs):
            self.tempfit_test_helper(ax, **config, fit_lorentz=True)

        self.save_fig(f)

    def test_guess_rv(self):
        configs = [
            dict(flux_correction=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        ]

        f, axs = plt.subplots(1, len(configs), squeeze=False)

        for ax, config in zip(axs[0], configs):
            self.tempfit_test_helper(ax, **configs[0], guess_rv=True)

        self.save_fig(f)

    def fit_rv_test_helper(self, ax, flux_correction, normalize, convolve_template, multiple_arms, multiple_exp, use_priors):
        tempfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = \
            self.get_initialized_tempfit(
                flux_correction=flux_correction,
                normalize=normalize,
                convolve_template=convolve_template,
                multiple_arms=multiple_arms,
                multiple_exp=multiple_exp,
                use_priors=use_priors
            )
        res = tempfit.fit_rv(specs, temps)

        ax.axvline(rv_real, color='r', label='rv real')
        ax.axvline(res.rv_fit, color='b', label='rv fit')
        ax.axvline(res.rv_fit - res.rv_err, color='b')
        ax.axvline(res.rv_fit + res.rv_err, color='b')

        # rvv = np.linspace(rv_real - 10 * rv_err, rv_real + 10 * rv_err, 101)
        rvv = np.linspace(res.rv_fit - 0.001, res.rv_fit + 0.001, 101)
        log_L = tempfit.calculate_log_L(specs, temps, rvv)

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
            self.fit_rv_test_helper(ax, **config)

        self.save_fig(f)

    def test_fit_rv_flux_corr(self):
        # Test various types of flux correction

        f, ax = plt.subplots(1, 1, figsize=(6, 4), squeeze=True)

        tempfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = \
            self.get_initialized_tempfit(
                flux_correction=True,
                normalize=True,
                convolve_template=True,
                multiple_arms=True,
                multiple_exp=True,
                use_priors=True
            )

        tempfit.correction_model.amplitude_per_arm = True
        tempfit.correction_model.amplitude_per_exp = True
        tempfit.correction_model.flux_corr_per_arm = True
        tempfit.correction_model.flux_corr_per_exp = True

        tempfit.init_correction_models(specs, rv_bounds=(-500, 500))

        res = tempfit.fit_rv(specs, temps)

        ax.axvline(rv_real, color='r', label='rv real')
        ax.axvline(res.rv_fit, color='b', label='rv fit')
        ax.axvline(res.rv_fit - res.rv_err, color='b')
        ax.axvline(res.rv_fit + res.rv_err, color='b')

        # rvv = np.linspace(rv_real - 10 * rv_err, rv_real + 10 * rv_err, 101)
        rvv = np.linspace(res.rv_fit - 0.001, res.rv_fit + 0.001, 101)
        log_L = tempfit.calculate_log_L(specs, temps, rvv)
        ax.plot(rvv, log_L, '.')
        ax.set_xlim(rvv[0], rvv[-1])

        ax.set_title(f'RV={rv_real:.2f}, RF_fit={res.rv_fit:.3f}+/-{res.rv_err:.3f}')
        # ax.set_xlim(rv_real - 50 * rv_err, rv_real + 50 * rv_err)

        self.save_fig(f)

    def test_fit_rv_rv_fixed(self):
        f, ax = plt.subplots(1, 1, figsize=(6, 4), squeeze=True)

        tempfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = \
            self.get_initialized_tempfit(
                flux_correction=True,
                normalize=True,
                convolve_template=True,
                multiple_arms=True,
                multiple_exp=True,
                use_priors=True)

        tempfit.amplitude_per_arm = True
        tempfit.amplitude_per_exp = True
        tempfit.flux_corr_per_arm = True
        tempfit.flux_corr_per_exp = True

        tempfit.init_correction_models(specs, rv_bounds=(-500, 500))

        # Only fit the continuum correction
        res = tempfit.fit_rv(specs, temps, rv_fixed=True)

        ax.axvline(rv_real, color='r', label='rv real')
        ax.axvline(res.rv_fit, color='b', label='rv fit')
        ax.axvline(res.rv_fit - res.rv_err, color='b')
        ax.axvline(res.rv_fit + res.rv_err, color='b')

        # rvv = np.linspace(rv_real - 10 * rv_err, rv_real + 10 * rv_err, 101)
        rvv = np.linspace(res.rv_fit - 0.001, res.rv_fit + 0.001, 101)
        log_L = tempfit.calculate_log_L(specs, temps, rvv)
        ax.plot(rvv, log_L, '.')
        ax.set_xlim(rvv[0], rvv[-1])

        ax.set_title(f'RV={rv_real:.2f}, RF_fit={res.rv_fit:.3f}+/-{res.rv_err:.3f}')
        # ax.set_xlim(rv_real - 50 * rv_err, rv_real + 50 * rv_err)

        self.save_fig(f)

    def run_mcmc_test_helper(self, ax, flux_correction, normalize, convolve_template, multiple_arms, multiple_exp, use_priors):
        tempfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = \
            self.get_initialized_tempfit(
                flux_correction=flux_correction,
                normalize=normalize,
                convolve_template=convolve_template,
                multiple_arms=multiple_arms,
                multiple_exp=multiple_exp,
                use_priors=use_priors
            )

        ax.axvline(rv_real, color='r', label='rv real')

        tempfit.mcmc_burnin = 5
        tempfit.mcmc_samples = 5
        res = tempfit.run_mcmc(specs, temps,
                               rv_0=rv_real + 10,
                               rv_bounds=(rv_real - 100, rv_real + 100),
                               rv_step=2)

        ax.hist(res.rv_mcmc.flatten())

    def test_run_mcmc(self):
        configs = [
            dict(flux_correction=False, use_priors=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(flux_correction=True, use_priors=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(flux_correction=True, use_priors=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        ]

        f, axs = plt.subplots(len(configs), 1, figsize=(6, 4 * len(configs)), squeeze=False)

        for ax, config in zip(axs[:, 0], configs):
            self.run_mcmc_test_helper(ax, **config)

        self.save_fig(f)

    def test_calculate_fisher(self):
        configs = [
            dict(flux_correction=False, use_priors=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(flux_correction=True, use_priors=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(flux_correction=True, use_priors=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        ]

        f, axs = plt.subplots(len(configs), 1, squeeze=False)

        for ax, config in zip(axs[:, 0], configs):
            
            tempfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = \
                self.get_initialized_tempfit(**config)
            
            res = tempfit.fit_rv(specs, temps)
            F = {}
            C = {}
            err = {}

            for mode, method in zip([
                'full',
                'rv',
                # 'full',
                # 'rv',
                'full',
                'full'
            ], [
                'hessian',
                'hessian',
                # 'emcee',
                # 'sampling',
                'phi_chi',
                'alex'
            ]):

                FF, CC = tempfit.correction_model.calculate_F(specs, temps, res.rv_fit, mode=mode, method=method)
                F[f'{mode}_{method}'] = FF
                C[f'{mode}_{method}'] = CC
                err[f'{mode}_{method}'] = np.sqrt(CC[-1, -1])

            pass

        self.save_fig(f)

    def test_calculate_rv_bouchy(self):
        configs = [
            dict(flux_correction=False, use_priors=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
        ]

        tempfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = \
            self.get_initialized_tempfit(**configs[0])
        tempfit.init_correction_models(specs, rv_bounds=(-500, 500))
        tempfit.calculate_rv_bouchy(specs, temps, rv_real)

# Run when profiling

# $ python -m cProfile -o tmp/tmp.prof python/test/pfs/ga/pfsspec/stellar/rvfit/test_rvfit.py
# $ python -m snakeviz --server tmp/tmp.prof

# t = TestTempFit()
# t.setUpClass()
# t.setUp()
# t.test_fit_rv()