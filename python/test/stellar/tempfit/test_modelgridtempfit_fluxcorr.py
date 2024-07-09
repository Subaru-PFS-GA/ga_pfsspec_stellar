import os
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from pfs.ga.pfsspec.core.sampling import Parameter, NormalDistribution, UniformDistribution
from pfs.ga.pfsspec.core.obsmod.resampling import FluxConservingResampler
from pfs.ga.pfsspec.stellar.tempfit import ModelGridTempFit, ModelGridTempFitTrace
from pfs.ga.pfsspec.stellar.tempfit import FluxCorr
from pfs.ga.pfsspec.stellar.fluxcorr import PolynomialFluxCorrection

from .tempfittestbase import TempFitTestBase


class ModelGridTempFitTraceTest(ModelGridTempFitTrace):
    def __init__(self):
        super().__init__()


class TestModelGridTempFitFluxCorr(TempFitTestBase):
    
    def get_tempfit(self,
                    flux_correction=False,
                    use_priors=False,
                    **kwargs):
        
        trace = ModelGridTempFitTraceTest()

        fluxcorr = FluxCorr()
        if flux_correction:
            fluxcorr.use_flux_corr = True
            fluxcorr.flux_corr_type = PolynomialFluxCorrection

        tempfit = ModelGridTempFit(trace=trace, correction_model=fluxcorr)
        
        tempfit.mcmc_burnin = 5       # Just a few MCMC steps to make it fast
        tempfit.mcmc_samples = 5
        tempfit.params_0 = kwargs
        tempfit.template_resampler = FluxConservingResampler()
        tempfit.template_grids = {
            'b': self.get_bosz_grid(),
            'mr': self.get_bosz_grid()
        }

        if flux_correction:
            tempfit.use_flux_corr = True
            tempfit.flux_corr_type = PolynomialFluxCorrection

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

    def tempfit_test_helper(self,
                            ax,
                            flux_correction,
                            normalize,
                            convolve_template,
                            multiple_arms,
                            multiple_exp,
                            use_priors=False,
                            calculate_log_L=False,
                            fit_lorentz=False,
                            guess_rv=False,
                            fit_rv=False,
                            calculate_error=False):

        (tempfit, rv_real, specs, temps, psfs,
         phi_shape, chi_shape, params_0) = self.get_initialized_tempfit(
            flux_correction=flux_correction,
            normalize=normalize,
            convolve_template=convolve_template,
            multiple_arms=multiple_arms,
            multiple_exp=multiple_exp,
            use_priors=use_priors
        )

        tempfit.init_correction_models(specs, rv_bounds=(-500, 500), force=True)

        ax.axvline(rv_real, color='r', label='rv real')

        if calculate_log_L:
            def calculate_log_L_helper(rv, shape):
                if np.shape(rv) == ():
                    # Tests specific to FluxCorr correction model
                    pp_specs = tempfit.preprocess_spectra(specs)
                    pp_temps = tempfit.preprocess_templates(specs, temps, rv)
                    log_L, phi, chi, ndf = tempfit.correction_model.eval_log_L(pp_specs, pp_temps, return_phi_chi=True)
                    a = tempfit.correction_model.eval_a(phi, chi)

                    self.assertEqual(shape, np.shape(log_L))
                    self.assertEqual(shape + phi_shape, np.shape(phi))
                    self.assertEqual(shape + chi_shape, np.shape(chi))

                    # Do not specify the templates here
                    log_L = tempfit.calculate_log_L(specs, None, rv)
                    log_L = tempfit.calculate_log_L(specs, None, rv, params=params_0)
                    log_L = tempfit.calculate_log_L(specs, None, rv, a=a)
                    log_L = tempfit.calculate_log_L(specs, None, rv, a=a, params=params_0)

                tempfit.calculate_log_L(specs, temps, rv)
        
            # Test with scalar
            rv = 100
            calculate_log_L_helper(rv, ())
            
            # Test with vector
            rv = np.linspace(-300, 300, 31)
            calculate_log_L_helper(rv, rv.shape)
            
        if fit_lorentz or guess_rv or fit_rv or calculate_error:
            rv = np.linspace(-300, 300, 31)
            log_L, phi, chi, ndf = tempfit.calculate_log_L(specs, temps, rv)
            pp, _ = tempfit.fit_lorentz(rv, log_L)

            y1 = tempfit.lorentz(rv, *pp)

            ax.plot(rv, log_L, 'o')
            ax.plot(rv, y1, '-')
  
        if guess_rv or fit_rv or calculate_error:
            _, _, rv0 = tempfit.guess_rv(specs, temps)
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
        model = tempfit.init_correction_model(specs, rv_bounds=(-500, 500), per_arm=False, per_exp=False,
                                              create_model_func=tempfit.correction_model.create_flux_corr)
        self.assertEqual(1, len(model))
        self.assertIsInstance(model[0], PolynomialFluxCorrection)

        model = tempfit.init_correction_model(specs, rv_bounds=(-500, 500), per_arm=False, per_exp=True,
                                              create_model_func=tempfit.correction_model.create_flux_corr)
        self.assertEqual(2, len(model))
        self.assertIsInstance(model[0], PolynomialFluxCorrection)

        model = tempfit.init_correction_model(specs, rv_bounds=(-500, 500), per_arm=True, per_exp=False,
                                              create_model_func=tempfit.correction_model.create_flux_corr)
        self.assertEqual(2, len(model))
        self.assertIsInstance(model[0], PolynomialFluxCorrection)

        model = tempfit.init_correction_model(specs, rv_bounds=(-500, 500), per_arm=True, per_exp=True,
                                              create_model_func=tempfit.correction_model.create_flux_corr)
        self.assertEqual(4, len(model))
        self.assertIsInstance(model[0], PolynomialFluxCorrection)

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
            coeff_count = tempfit.correction_model.get_coeff_count(specs)

            self.assertEqual(gt_amp_count, amp_count)
            self.assertEqual(gt_coeff_count, coeff_count)        

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

    def rvfit_fit_rv_test_helper(self,
                                 ax,
                                 flux_correction, normalize, convolve_template,
                                 multiple_arms, multiple_exp,
                                 use_priors, rv_fixed=False):
        
        (rvfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0) = \
            self.get_initialized_tempfit(
                flux_correction=flux_correction,
                normalize=normalize,
                convolve_template=convolve_template,
                multiple_arms=multiple_arms,
                multiple_exp=multiple_exp,
                use_priors=use_priors
            )

        ax.axvline(rv_real, color='r', label='rv real')

        res = rvfit.fit_rv(specs,
                           rv_0=rv_real + 10,
                           rv_bounds=(rv_real - 100, rv_real + 100),
                           rv_fixed=rv_fixed,
                           params_0=params_0)
                
        ax.axvline(res.rv_fit, color='b', label='rv fit')
        ax.axvline(res.rv_fit - res.rv_err, color='b')
        ax.axvline(res.rv_fit + res.rv_err, color='b')

        # TODO: create a log_L map instead?
        # # rvv = np.linspace(rv_real - 10 * rv_err, rv_real + 10 * rv_err, 101)
        # rvv = np.linspace(rv - 0.001, rv + 0.001, 101)
        # log_L, phi, chi, ndf = rvfit.calculate_log_L(specs, temps, rvv)
        # ax.plot(rvv, log_L, '.')
        # ax.set_xlim(rvv[0], rvv[-1])

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

    def test_fit_rv_rv_fixed(self):
        configs = [
            dict(flux_correction=False, use_priors=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(flux_correction=True, use_priors=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(flux_correction=True, use_priors=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        ]

        f, axs = plt.subplots(len(configs), 1, figsize=(6, 4 * len(configs)), squeeze=False)

        for ax, config in zip(axs[:, 0], configs):
            self.rvfit_fit_rv_test_helper(ax, rv_fixed=True, **config)

        self.save_fig(f)

    def rvfit_run_mcmc_test_helper(self, ax, flux_correction, normalize, convolve_template, multiple_arms, multiple_exp, use_priors):
        
        (tempfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0) = \
            self.get_initialized_tempfit(
                flux_correction=flux_correction,
                normalize=normalize,
                convolve_template=convolve_template,
                multiple_arms=multiple_arms,
                multiple_exp=multiple_exp,
                use_priors=use_priors
            )

        params_steps = { p: v * 0.01 for p, v in params_0.items() }

        ax.axvline(rv_real, color='r', label='rv real')

        tempfit.mcmc_burnin = 5
        tempfit.mcmc_samples = 5
        res = tempfit.run_mcmc(specs,
                               rv_0=rv_real + 10,
                               rv_bounds=(rv_real - 100, rv_real + 100),
                               rv_step=2,
                               params_0=params_0,
                               params_steps=params_steps)

        ax.plot(res.rv_mcmc.ravel(), res.params_mcmc['T_eff'].ravel(), '.')

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
            rvfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = \
                self.get_initialized_tempfit(**config)
            
            res = rvfit.fit_rv(specs, rv_0=rv_real)
            
            F = {}
            C = {}
            err = {}
            for mode, method in zip([
                    'full',
                    'params_rv',
                    'rv',
                    # 'full',
                    # 'params_rv',
                    # 'rv',
                ], [
                    'hessian',
                    'hessian',
                    'hessian',
                    # 'emcee',
                    # 'emcee',
                    # 'sampling',
                ]):
                FF, CC = rvfit.calculate_F(specs, res.rv_fit, res.params_fit, mode=mode, method=method, step=0.01)
                F[f'{mode}_{method}'] = FF
                C[f'{mode}_{method}'] = CC
                err[f'{mode}_{method}'] = np.sqrt(CC[-1, -1])

            pass

        self.save_fig(f)
