import os
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from pfs.ga.pfsspec.core.sampling import Parameter, NormalDistribution, UniformDistribution
from pfs.ga.pfsspec.core.obsmod.resampling import FluxConservingResampler
from pfs.ga.pfsspec.stellar.tempfit import ModelGridTempFit, ModelGridTempFitTrace
from pfs.ga.pfsspec.stellar.tempfit import ContNorm
from pfs.ga.pfsspec.stellar.continuum.models import Spline

from .tempfittestbase import TempFitTestBase


class ModelGridTempFitTraceTest(ModelGridTempFitTrace):
    def __init__(self):
        super().__init__()

        self.mcmc_x = None
        self.mcmc_log_L = None


class TestModelGridTempFitContNorm(TempFitTestBase):
    
    def get_tempfit(self,
                    flux_correction=False,
                    continuum_fit=True,
                    use_priors=False,
                    **kwargs):
        
        trace = ModelGridTempFitTraceTest()

        contnorm = ContNorm()
        contnorm.continuum_fit = True

        tempfit = ModelGridTempFit(trace=trace, correction_model=contnorm)
        
        tempfit.mcmc_burnin = 5       # Just a few MCMC steps to make it fast
        tempfit.mcmc_samples = 5
        tempfit.params_0 = kwargs
        tempfit.template_resampler = FluxConservingResampler()
        tempfit.template_grids = self.get_normalized_grids()

        if use_priors:
            tempfit.rv_prior = lambda rv: -(rv - self.rv_real)**2 / 100**2
            tempfit.params_priors = {
                'T_eff': lambda T_eff: -(T_eff - 4500)**2 / 300**2
            }
        else:
            tempfit.rv_prior = None
            tempfit.params_priors = None

        tempfit.params_fixed = {
            'C_M': kwargs['C_M'] if 'C_M' in kwargs else 0.0,
            'a_M': kwargs['a_M'] if 'a_M' in kwargs else 0.0,
        }

        return tempfit

    def tempfit_test_helper(self,
                            ax,
                            continuum_fit,
                            continuum_per_arm,
                            continuum_per_exp,
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
            continuum_fit=continuum_fit,
            normalize=normalize,
            convolve_template=convolve_template,
            multiple_arms=multiple_arms,
            multiple_exp=multiple_exp,
            use_priors=use_priors
        )

        tempfit.correction_model.use_cont_norm = continuum_fit
        tempfit.correction_model.cont_per_arm = continuum_per_arm
        tempfit.correction_model.cont_per_exp = continuum_per_exp

        tempfit.init_correction_models(specs, rv_bounds=(-500, 500), force=True)

        ax.axvline(rv_real, color='r', label='rv real')

        if calculate_log_L:
            def calculate_log_L_helper(rv):
                # Tests specific to ContNorm correction model
                pp_specs = tempfit.preprocess_spectra(specs)
                pp_temps = tempfit.preprocess_templates(specs, temps, rv)
                log_L = tempfit.correction_model.eval_log_L(pp_specs, pp_temps)
                
                ax.plot(rv, log_L, 'o')
        
            # Test with scalar
            calculate_log_L_helper(100)
            tempfit.calculate_log_L(specs, temps, 100)
            
            # Test with vector
            rv = np.linspace(-300, 300, 31)
            tempfit.calculate_log_L(specs, temps, rv)
            
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
                continuum_fit=True,
                normalize=True,
                convolve_template=True,
                multiple_arms=True,
                multiple_exp=True,
                use_priors=False)
        
        # Test different types of freedom
        model = tempfit.init_correction_model(specs, rv_bounds=(-500, 500), per_arm=False, per_exp=False,
                                   create_model_func=tempfit.correction_model.create_continuum_model)
        self.assertEqual(1, len(model))
        self.assertIsInstance(model[0], Spline)

        model = tempfit.init_correction_model(specs, rv_bounds=(-500, 500), per_arm=False, per_exp=True,
                                   create_model_func=tempfit.correction_model.create_continuum_model)
        self.assertEqual(2, len(model))
        self.assertIsInstance(model[0], Spline)

        model = tempfit.init_correction_model(specs, rv_bounds=(-500, 500), per_arm=True, per_exp=False,
                                   create_model_func=tempfit.correction_model.create_continuum_model)
        self.assertEqual(2, len(model))
        self.assertIsInstance(model[0], Spline)

        model = tempfit.init_correction_model(specs, rv_bounds=(-500, 500), per_arm=True, per_exp=True,
                                              create_model_func=tempfit.correction_model.create_continuum_model)
        self.assertEqual(4, len(model))
        self.assertIsInstance(model[0], Spline)

    def test_get_coeff_count(self):
        tempfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = \
            self.get_initialized_tempfit(
                continuum_fit=True,
                normalize=True,
                convolve_template=True,
                multiple_arms=True,
                multiple_exp=True,
                use_priors=False)

        # arms: 2
        # exposures: 2
        # degree = 5

        gt = [
            [ False, False, False, False, 1, 0 ],
            [ False, False, False, True, 1, 0 ],
            [ False, False, True, False, 1, 0 ],
            [ False, False, True, True, 1, 0 ],
            [ False, True, False, False, 2, 0 ],
            [ False, True, False, True, 2, 0 ],
            [ False, True, True, False, 2, 0 ],
            [ False, True, True, True, 2, 0 ],
            [ True, False, False, False, 2, 0 ],
            [ True, False, False, True, 2, 0 ],
            [ True, False, True, False, 2, 0 ],
            [ True, False, True, True, 2, 0 ],
            [ True, True, False, False, 4, 0 ],
            [ True, True, False, True, 4, 0 ],
            [ True, True, True, False, 4, 0 ],
            [ True, True, True, True, 4, 0 ],
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

    def test_calculate_log_L(self):
        configs = [
            dict(continuum_fit=True, continuum_per_arm=True, continuum_per_exp=True, use_priors=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(continuum_fit=True, continuum_per_arm=True, continuum_per_exp=False, use_priors=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        ]

        f, ax = plt.subplots(1, 1)

        for config in configs:
            self.tempfit_test_helper(ax, **config, calculate_log_L=True)

        self.save_fig(f)

    def test_fit_lorentz(self):
        configs = [
            dict(continuum_fit=True, continuum_per_arm=True, continuum_per_exp=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(continuum_fit=True, continuum_per_arm=True, continuum_per_exp=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
        ]

        f, axs = plt.subplots(1, len(configs), squeeze=False)

        for ax, config in zip(axs[0], configs):
            self.tempfit_test_helper(ax, **config, fit_lorentz=True)

        self.save_fig(f)

    def test_guess_rv(self):
        configs = [
            dict(continuum_fit=True, continuum_per_arm=True, continuum_per_exp=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(continuum_fit=True, continuum_per_arm=True, continuum_per_exp=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        ]

        f, axs = plt.subplots(1, len(configs), squeeze=False)

        for ax, config in zip(axs[0], configs):
            self.tempfit_test_helper(ax, **configs[0], guess_rv=True)

        self.save_fig(f)

    def tempfit_fit_rv_test_helper(self,
                                   ax,
                                   continuum_fit, continuum_per_arm, continuum_per_exp,
                                   normalize,
                                   convolve_template,
                                   multiple_arms, multiple_exp,
                                   use_priors,
                                   rv_fixed=False):
        
        (tempfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0) = \
            self.get_initialized_tempfit(
                continuum_fit=continuum_fit,
                normalize=normalize,
                convolve_template=convolve_template,
                multiple_arms=multiple_arms,
                multiple_exp=multiple_exp,
                use_priors=use_priors,

                # Override to allow working with Roman's current grid
                # TODO: remove this
                M_H=-1.5, T_eff=4000, log_g=2.5, a_M=-0.5
            )
        
        tempfit.correction_model.use_cont_norm = continuum_fit
        tempfit.correction_model.cont_per_arm = continuum_per_arm
        tempfit.correction_model.cont_per_exp = continuum_per_exp

        tempfit.init_correction_models(specs, rv_bounds=(-500, 500), force=True)

        ax.axvline(rv_real, color='r', label='rv real')

        state = tempfit.init_state(specs,
                                    rv_0=rv_real + 10,
                                    rv_bounds=(rv_real - 100, rv_real + 100),
                                    rv_fixed=rv_fixed,
                                    params_0=params_0)

        res, state = tempfit.run_ml(state)
        res, state = tempfit.calculate_error_ml(state)
        res, state = tempfit.calculate_cov_ml(state)
        res, state = tempfit.finish_ml(state)
                
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
            dict(continuum_fit=True, continuum_per_arm=True, continuum_per_exp=True, use_priors=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(continuum_fit=True, continuum_per_arm=True, continuum_per_exp=False, use_priors=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        ]

        f, axs = plt.subplots(len(configs), 1, figsize=(6, 4 * len(configs)), squeeze=False)

        for ax, config in zip(axs[:, 0], configs):
            self.tempfit_fit_rv_test_helper(ax, **config)

        self.save_fig(f)

    def test_fit_rv_rv_fixed(self):
        configs = [
            dict(continuum_fit=True, continuum_per_arm=True, continuum_per_exp=True, use_priors=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(continuum_fit=True, continuum_per_arm=True, continuum_per_exp=False, use_priors=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        ]

        f, axs = plt.subplots(len(configs), 1, figsize=(6, 4 * len(configs)), squeeze=False)

        for ax, config in zip(axs[:, 0], configs):
            self.tempfit_fit_rv_test_helper(ax, rv_fixed=True, **config)

        self.save_fig(f)

    def tempfit_run_mcmc_test_helper(self,
                                   ax,
                                   continuum_fit, continuum_per_arm, continuum_per_exp,
                                   normalize,
                                   convolve_template,
                                   multiple_arms,
                                   multiple_exp, use_priors):
        
        (tempfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0) = \
            self.get_initialized_tempfit(
                continuum_fit=continuum_fit,
                normalize=normalize,
                convolve_template=convolve_template,
                multiple_arms=multiple_arms,
                multiple_exp=multiple_exp,
                use_priors=use_priors,

                # Override to allow working with Roman's current grid
                # TODO: remove this
                M_H=-1.5, T_eff=4000, log_g=2.5, a_M=-0.5
            )
        
        tempfit.correction_model.use_cont_norm = continuum_fit
        tempfit.correction_model.cont_per_arm = continuum_per_arm
        tempfit.correction_model.cont_per_exp = continuum_per_exp

        tempfit.init_correction_models(specs, rv_bounds=(-500, 500), force=True)

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
            dict(continuum_fit=True, continuum_per_arm=True, continuum_per_exp=True, use_priors=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(continuum_fit=True, continuum_per_arm=True, continuum_per_exp=False, use_priors=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        ]

        f, axs = plt.subplots(len(configs), 1, figsize=(6, 4 * len(configs)), squeeze=False)

        for ax, config in zip(axs[:, 0], configs):
            self.tempfit_run_mcmc_test_helper(ax, **config)

        self.save_fig(f)

    def test_calculate_fisher(self):

        configs = [
            dict(continuum_fit=True, use_priors=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(continuum_fit=True, use_priors=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        ]

        f, axs = plt.subplots(len(configs), 1, squeeze=False)

        for ax, config in zip(axs[:, 0], configs):
            tempfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = \
                self.get_initialized_tempfit(
                    # Override to allow working with Roman's current grid
                    # TODO: remove this
                    M_H=-1.5, T_eff=4000, log_g=2.5, a_M=-0.5,
                    **config)
            
            tempfit.correction_model.use_cont_norm = True
            tempfit.correction_model.cont_per_arm = True
            tempfit.correction_model.cont_per_exp = True

            
            state = tempfit.init_state(specs, rv_0=rv_real)
            res, state = tempfit.run_ml(state)
            res, state = tempfit.calculate_error_ml(state)
            res, state = tempfit.calculate_cov_ml(state)
            res, state = tempfit.finish_ml(state)
            
            F = {}
            C = {}
            err = {}
            for mode, method in zip([
                    # 'full',
                    'params_rv',
                    'rv',
                    # 'full',
                    # 'params_rv',
                    # 'rv',
                ], [
                    # 'hessian',
                    'hessian',
                    'hessian',
                    # 'emcee',
                    # 'emcee',
                    # 'sampling',
                ]):
                FF, CC = tempfit.calculate_F(specs, None, res.rv_fit, res.params_fit, mode=mode, method=method, step=0.01)
                F[f'{mode}_{method}'] = FF
                C[f'{mode}_{method}'] = CC
                err[f'{mode}_{method}'] = np.sqrt(CC[-1, -1])

            pass

        self.save_fig(f)
