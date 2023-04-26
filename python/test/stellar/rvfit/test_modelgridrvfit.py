import os
import numpy as np
import matplotlib.pyplot as plt

from test.pfs.ga.pfsspec.stellar.stellartestbase import StellarTestBase
from pfs.ga.pfsspec.core.physics import Physics
from pfs.ga.pfsspec.core.grid import ArrayGrid
from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid.bosz import Bosz
from pfs.ga.pfsspec.core import Filter
from pfs.ga.pfsspec.sim.obsmod import Detector
from pfs.ga.pfsspec.sim.obsmod.background import Sky, Moon
from pfs.ga.pfsspec.sim.obsmod.observations import PfsObservation
from pfs.ga.pfsspec.sim.obsmod.pipelines import StellarModelPipeline
from pfs.ga.pfsspec.sim.obsmod.calibration import FluxCalibrationBias
from pfs.ga.pfsspec.core.obsmod.psf import PcaPsf, GaussPsf
from pfs.ga.pfsspec.core.obsmod.resampling import FluxConservingResampler
from pfs.ga.pfsspec.stellar.rvfit import ModelGridRVFit, ModelGridRVFitTrace

from .rvfittestbase import RVFitTestBase

class ModelGridRVFitTraceTest(ModelGridRVFitTrace):
    def __init__(self):
        super().__init__()

        self.mcmc_x_0 = None
        self.mcmc_x = None
        self.mcmc_log_L = None

    def on_eval_F_mcmc(self, x_0, x, log_L):
        super().on_eval_F_mcmc(x_0, x, log_L)

        self.mcmc_x_0 = x_0
        self.mcmc_x = x
        self.mcmc_log_L = log_L

class TestModelGridRVFit(RVFitTestBase):
    
    def get_rvfit(self, flux_correction=False, use_priors=False, **kwargs):
        trace = ModelGridRVFitTraceTest()
        rvfit = ModelGridRVFit(trace=trace)
        rvfit.mcmc_burnin = 5       # Just a few MCMC steps to make it fast
        rvfit.mcmc_steps = 5
        rvfit.params_0 = kwargs
        rvfit.template_resampler = FluxConservingResampler()
        rvfit.template_grids = {
            'b': self.get_bosz_grid(),
            'mr': self.get_bosz_grid()
        }

        if flux_correction:
            rvfit.use_flux_corr = True
            rvfit.flux_corr_basis = RVFitTestBase.flux_correction_polys

        if use_priors:
            rvfit.rv_prior = lambda rv: -(rv - self.rv_real)**2 / 100**2
            rvfit.params_priors = {
                'T_eff': lambda T_eff: -(T_eff - 4500)**2 / 300**2
            }
        else:
            rvfit.rv_prior = None
            rvfit.params_priors = None

        rvfit.params_fixed = {
            'C_M': 0,
            'a_M': 0
        }

        return rvfit
    
    def test_get_normalization(self):
        rvfit = self.get_rvfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        # Provide the templates
        spec_norm, temp_norm = rvfit.get_normalization({'mr': spec}, {'mr': temp})

        # Provide the template parameters
        spec_norm, temp_norm = rvfit.get_normalization({'mr': spec}, M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)

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

                # Do not specify the templates here
                log_L, _, _, _ = rvfit.calculate_log_L(specs, None, rv)
                log_L, phi, chi, ndf = rvfit.calculate_log_L(specs, None, rv, params=params_0)
                log_L, _, _, _ = rvfit.calculate_log_L(specs, None, rv, a=a)
                log_L, phi, chi, ndf = rvfit.calculate_log_L(specs, None, rv, a=a, params=params_0)
        
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

    def rvfit_fit_rv_test_helper(self, ax, flux_correction, normalize, convolve_template, multiple_arms, multiple_exp, use_priors):
        rvfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = self.get_initialized_rvfit(flux_correction, normalize, convolve_template, multiple_arms, multiple_exp, use_priors)

        ax.axvline(rv_real, color='r', label='rv real')

        rv, rv_err, params, params_err, a, a_err = rvfit.fit_rv(specs,
                                  rv_0=rv_real + 10, rv_bounds=(rv_real - 100, rv_real + 100),
                                  params_0=params_0)
        
        ax.axvline(rv, color='b', label='rv fit')
        ax.axvline(rv - rv_err, color='b')
        ax.axvline(rv + rv_err, color='b')

        # TODO: create a log_L map instead?
        # # rvv = np.linspace(rv_real - 10 * rv_err, rv_real + 10 * rv_err, 101)
        # rvv = np.linspace(rv - 0.001, rv + 0.001, 101)
        # log_L, phi, chi, ndf = rvfit.calculate_log_L(specs, temps, rvv)
        # ax.plot(rvv, log_L, '.')
        # ax.set_xlim(rvv[0], rvv[-1])

        ax.set_title(f'RV={rv_real:.2f}, RF_fit={rv:.3f}+/-{rv_err:.3f}')
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

    def test_calculate_fisher(self):

        configs = [
            dict(flux_correction=False, use_priors=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(flux_correction=True, use_priors=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(flux_correction=True, use_priors=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        ]

        f, axs = plt.subplots(len(configs), 1, squeeze=False)

        for ax, config in zip(axs[:, 0], configs):
            rvfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = self.get_initialized_rvfit(**config)
            rv, rv_err, params, params_err, a, a_err = rvfit.fit_rv(specs, rv_0=rv_real)
            
            F = {}
            C = {}
            err = {}
            for mode, method in zip([
                    'full',
                    'params_rv',
                    'rv',
                    'full',
                    'params_rv',
                    'rv',
                ], [
                    'hessian',
                    'hessian',
                    'hessian',
                    'emcee',
                    'emcee',
                    'sampling',
                ]):
                FF, CC = rvfit.calculate_F(specs, rv, params, mode=mode, method=method)
                F[f'{mode}_{method}'] = FF
                C[f'{mode}_{method}'] = CC
                err[f'{mode}_{method}'] = np.sqrt(CC[-1, -1])

            pass

        self.save_fig(f)

    def test_eval_F_emcee(self):
        config = dict(flux_correction=False, use_priors=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        rvfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = self.get_initialized_rvfit(**config)
        rv, rv_err, params, params_err, a, a_err = rvfit.fit_rv(specs, rv_0=rv_real)

        rvfit.calculate_F(specs, rv, params, mode='params_rv', method='emcee')

        #

        grid = rvfit.template_grids['mr']
        params_free = [p for i, p, ax in grid.enumerate_axes() if p not in rvfit.params_fixed] + [ 'rv' ]

        n = rvfit.trace.mcmc_x.shape[-1]
        f, axs = plt.subplots(n, n, figsize=(2 * n, 2 * n), squeeze=False)

        for i in range(n):
            for j in range(i + 1):
                axs[i, j].plot(rvfit.trace.mcmc_x[:, i], rvfit.trace.mcmc_x[:, j], '.')
                axs[i, j].set_xlabel(params_free[i])
                axs[i, j].set_ylabel(params_free[j])

        self.save_fig(f)