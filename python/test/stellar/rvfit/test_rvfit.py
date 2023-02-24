import numpy as np
import matplotlib.pyplot as plt

from pfs.ga.pfsspec.core.obsmod.resampling import FluxConservingResampler
from pfs.ga.pfsspec.stellar.rvfit import RVFit, RVFitTrace

from .rvfittestbase import RVFitTestBase

class TestRVFit(RVFitTestBase):
    
    def get_rvfit(self, flux_correction=False):
        trace = RVFitTrace()
        rvfit = RVFit(trace=trace)
        rvfit.template_resampler = FluxConservingResampler()

        if flux_correction:
            def polys(wave):
                npoly = 5
                wmin = 3000
                wmax = 12000
                normwave = (wave - wmin) / (wmax - wmin) * 2 - 1
                polys = np.empty((wave.shape[0], npoly))

                coeffs = np.eye(npoly)
                for i in range(npoly):
                    polys[:, i] = np.polynomial.Chebyshev(coeffs[i])(normwave)

                return polys

            rvfit.use_flux_corr = True
            rvfit.flux_corr_basis = polys

        return rvfit
    
    def get_initialized_rvfit(self, flux_correction, normalize, convolve_template, multiple_arms, multiple_exp):
        rv_real = 100
        rvfit = self.get_rvfit(flux_correction=flux_correction)

        if flux_correction:
            phi_shape = (5,)
            chi_shape = (5, 5)
        else:
            phi_shape = ()
            chi_shape = ()

        if multiple_arms:
            arms = ['b', 'mr']
        else:
            arms = ['mr']

        if multiple_exp:
            specs = { k: [self.get_observation(arm=k, rv=rv_real, M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0) for _ in range(2)] for k in arms }
        else:
            specs = { k: self.get_observation(arm=k, rv=rv_real, M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0) for k in arms }
        temps = { k: self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0) for k in arms}
        psfs = { k: self.get_test_psf(k) for k in arms}

        if convolve_template:
            rvfit.psf = psfs
        else:
            rvfit.psf = None

        if normalize:
            rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization(specs, temps)

        return rvfit, rv_real, specs, temps, psfs, phi_shape, chi_shape

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

        sp = rvfit.process_spectrum(spec)

        rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization({'mr': spec}, {'mr': temp})
        sp = rvfit.process_spectrum(spec)

    def test_process_template(self):
        spec = self.get_observation(arm='mr')
        
        rvfit = self.get_rvfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        tm = rvfit.process_template(temp, spec, 100)

        psf = self.get_test_psf(arm='mr')
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        tm = rvfit.process_template(temp, spec, 100, psf=psf)

        rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization({'mr': spec}, {'mr': temp})
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        tm = rvfit.process_template(temp, spec, 100, psf=psf)

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

    def rvfit_test_helper(self, ax, flux_correction, normalize, convolve_template, multiple_arms, multiple_exp, calculate_log_L=False, fit_lorentz=False, guess_rv=False, fit_rv=False, calculate_error=False):
        rvfit, rv_real, specs, temps, psfs, phi_shape, chi_shape = self.get_initialized_rvfit(flux_correction, normalize, convolve_template, multiple_arms, multiple_exp)

        ax.axvline(rv_real, color='r', label='rv real')

        if calculate_log_L:
            def calculate_log_L_helper(rv, shape):
                log_L, phi, chi = rvfit.calculate_log_L(specs, temps, rv)
                a = rvfit.eval_a(phi, chi)

                self.assertEqual(shape, log_L.shape)
                self.assertEqual(shape + phi_shape, phi.shape)
                self.assertEqual(shape + chi_shape, chi.shape)

                ax.plot(rv, log_L, 'o')

                log_L, _, _ = rvfit.calculate_log_L(specs, temps, rv, a=a)
        
            # Test with scalar
            calculate_log_L_helper(100, ())
            
            # Test with vector
            rv = np.linspace(-300, 300, 31)
            calculate_log_L_helper(rv, rv.shape)

        if fit_lorentz or guess_rv or fit_rv or calculate_error:
            rv = np.linspace(-300, 300, 31)
            log_L, phi, chi = rvfit.calculate_log_L(specs, temps, rv)
            pp, _ = rvfit.fit_lorentz(rv, log_L)

            y1 = rvfit.lorentz(rv, *pp)

            ax.plot(rv, log_L, 'o')
            ax.plot(rv, y1, '-')
  
        if guess_rv or fit_rv or calculate_error:
            rv0 = rvfit.guess_rv(specs, temps)
            ax.axvline(rv0, color='k', label='rv guess')

    def test_eval_flux_corr_basis(self):
        rvfit, rv_real, specs, temps, psfs, phi_shape, chi_shape = self.get_initialized_rvfit(True, True, True, True, True)

        f, ax = plt.subplots(1, 1)
        
        basis, basis_size = rvfit.eval_flux_corr_basis(specs)
        for k in specs:
            for ei, ee in enumerate(specs[k] if isinstance(specs[k], list) else [specs[k]]):
                ax.plot(ee.wave, basis[k][ei])

        self.save_fig(f)

    def test_calculate_log_L(self):
        configs = [
            dict(flux_correction=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(flux_correction=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
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

    def rvfit_fit_rv_test_helper(self, ax, flux_correction, normalize, convolve_template, multiple_arms, multiple_exp):
        rvfit, rv_real, specs, temps, psfs, phi_shape, chi_shape = self.get_initialized_rvfit(flux_correction, normalize, convolve_template, multiple_arms, multiple_exp)

        ax.axvline(rv_real, color='r', label='rv real')

        rv, rv_err = rvfit.fit_rv(specs, temps)
        ax.axvline(rv, color='b', label='rv fit')
        ax.axvline(rv - rv_err, color='b')
        ax.axvline(rv + rv_err, color='b')

        # rvv = np.linspace(rv_real - 10 * rv_err, rv_real + 10 * rv_err, 101)
        rvv = np.linspace(rv - 0.001, rv + 0.001, 101)
        log_L, phi, chi = rvfit.calculate_log_L(specs, temps, rvv)
        ax.plot(rvv, log_L, '.')
        ax.set_xlim(rvv[0], rvv[-1])

        ax.set_title(f'RV={rv_real:.2f}, RF_fit={rv:.3f}+/-{rv_err:.3f}')
        # ax.set_xlim(rv_real - 50 * rv_err, rv_real + 50 * rv_err)

    def test_fit_rv(self):
        configs = [
            dict(flux_correction=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(flux_correction=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        ]

        f, axs = plt.subplots(len(configs), 1, figsize=(6, 4 * len(configs)), squeeze=False)

        for ax, config in zip(axs[:, 0], configs):
            self.rvfit_fit_rv_test_helper(ax, **config)

        self.save_fig(f)

    def test_calculate_fisher(self):
        configs = [
            dict(flux_correction=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
            dict(flux_correction=True, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True)
        ]

        f, axs = plt.subplots(len(configs), 1, squeeze=False)

        for ax, config in zip(axs[:, 0], configs):
            rvfit, rv_real, specs, temps, psfs, phi_shape, chi_shape = self.get_initialized_rvfit(**config)
            rv, rv_err = rvfit.fit_rv(specs, temps)
            F = {}
            C = {}
            err = {}
            for method in ['full_phi_chi', 'full_hessian', 'rv_hessian', 'full_alex', 'full_emcee', 'rv_sampling']:
                FF, CC = rvfit.calculate_F(specs, temps, rv, method=method)
                F[method] = FF
                C[method] = CC
                err[method] = np.sqrt(CC[-1, -1])

            pass

        self.save_fig(f)

    def test_calculate_rv_bouchy(self):
        configs = [
            dict(flux_correction=False, normalize=True, convolve_template=True, multiple_arms=True, multiple_exp=True),
        ]

        rvfit, rv_real, specs, temps, psfs, phi_shape, chi_shape = self.get_initialized_rvfit(**configs[0])
        rvfit.calculate_rv_bouchy(specs, temps, rv_real)

# Run when profiling

# $ python -m cProfile -o tmp/tmp.prof python/test/pfs/ga/pfsspec/stellar/rvfit/test_rvfit.py
# $ python -m snakeviz --server tmp/tmp.prof

# t = TestRVFit()
# t.setUpClass()
# t.setUp()
# t.test_fit_rv()