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
                npoly = 10
                normwave = (wave - wave[0]) / (wave[-1] - wave[0]) * 2 - 1
                polys = np.empty((wave.shape[0], npoly))

                coeffs = np.eye(npoly)
                for i in range(npoly):
                    polys[:, i] = np.polynomial.Chebyshev(coeffs[i])(normwave)

                return polys

            rvfit.flux_corr = True
            rvfit.flux_corr_basis = polys

        return rvfit

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
        spec = self.get_observation()
        
        rvfit = self.get_rvfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        tm = rvfit.process_template(temp, spec, 100)

        psf = self.get_test_psf()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        tm = rvfit.process_template(temp, spec, 100, psf=psf)

        rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization({'mr': spec}, {'mr': temp})
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        tm = rvfit.process_template(temp, spec, 100, psf=psf)

    def test_calculate_log_L_nocorrection(self):
        rvfit = self.get_rvfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)
        psf = self.get_test_psf()

        # Test with scalar
        rv = 100.0
        rvfit.psf = {'mr': psf}
        rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization({'mr': spec}, {'mr': temp})
        log_L, phi, chi = rvfit.calculate_log_L({'mr': spec}, {'mr': temp}, rv)
        self.assertEqual((), log_L.shape)
        self.assertEqual((), phi.shape)
        self.assertEqual((), chi.shape)

        plt.plot(rv, log_L, 'o')

        # Test with vector
        rv = np.linspace(-300, 300, 31)
        rvfit.psf = {'mr': psf}
        log_L, phi, chi = rvfit.calculate_log_L({'mr': spec}, {'mr': temp}, rv)
        self.assertEqual((31,), log_L.shape)
        self.assertEqual((31,), phi.shape)
        self.assertEqual((31,), chi.shape)

        # Test with multiple spectra
        rv = 100.0
        rvfit.psf = None
        log_L, phi, chi = rvfit.calculate_log_L({'b': spec, 'mr': spec}, {'b': temp, 'mr': temp}, rv)
        self.assertEqual((), log_L.shape)
        self.assertEqual((), phi.shape)
        self.assertEqual((), chi.shape)

        plt.plot(rv, log_L, 'o')

        rv = np.linspace(-300, 300, 31)
        rvfit.psf = {'b': psf, 'mr': psf}
        log_L, phi, chi = rvfit.calculate_log_L({'b': spec, 'mr': spec}, {'b': temp, 'mr': temp}, rv)
        self.assertEqual((31,), log_L.shape)
        self.assertEqual((31,), phi.shape)
        self.assertEqual((31,), chi.shape)
        plt.plot(rv, log_L, 'o')

        self.save_fig()

    def test_calculate_log_L_correction(self):
        rvfit = self.get_rvfit(flux_correction=True)
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)
        psf = self.get_test_psf()

        # Test with scalar
        rv = 100.0
        rvfit.psf = {'mr': psf}
        rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization({'mr': spec}, {'mr': temp})
        log_L, phi, chi = rvfit.calculate_log_L({'mr': spec}, {'mr': temp}, rv)

        # Test with vector
        rv = np.linspace(-300, 300, 31)
        rvfit.psf = {'mr': psf}
        rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization({'mr': spec}, {'mr': temp})
        log_L, phi, chi = rvfit.calculate_log_L({'mr': spec}, {'mr': temp}, rv)
        plt.plot(rv, log_L, 'o')

        # Test with multiple spectra
        rv = 100.0
        rvfit.psf = None
        rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization({'b': spec, 'mr': spec}, {'b': temp, 'mr': temp})
        log_L, phi, chi = rvfit.calculate_log_L({'b': spec, 'mr': spec}, {'b': temp, 'mr': temp}, rv)
        plt.plot(rv, log_L, 'o')

        rv = np.linspace(-300, 300, 31)
        rvfit.psf = {'b': psf, 'mr': psf}
        rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization({'b': spec, 'mr': spec}, {'b': temp, 'mr': temp})
        log_L, phi, chi = rvfit.calculate_log_L({'b': spec, 'mr': spec}, {'b': temp, 'mr': temp}, rv)
        plt.plot(rv, log_L, 'o')

        self.save_fig()

    def test_calculate_rv_error_nocorrection(self):
        rvfit = self.get_rvfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization({'mr': spec}, {'mr': temp})
        rv_error = rvfit.calculate_rv_error({'mr': spec}, {'mr': temp}, 125.1)
        rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization({'b': spec, 'mr': spec}, {'b': temp, 'mr': temp})
        rv_error = rvfit.calculate_rv_error({'b': spec, 'mr': spec}, {'b': temp, 'mr': temp}, 125.1)

    def test_calculate_rv_error_correction(self):
        rvfit = self.get_rvfit(flux_correction=True)
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)
        
        rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization({'mr': spec}, {'mr': temp})
        rv_error = rvfit.calculate_rv_error({'mr': spec}, {'mr': temp}, 125.1)
        rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization({'b': spec, 'mr': spec}, {'b': temp, 'mr': temp})
        rv_error = rvfit.calculate_rv_error({'b': spec, 'mr': spec}, {'b': temp, 'mr': temp}, 125.1)

    def test_calculate_fisher_nocorrection(self):
        rvfit = self.get_rvfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        F = rvfit.calculate_fisher({'mr': spec}, {'mr': temp}, 125.1)
        F = rvfit.calculate_fisher({'b': spec, 'mr': spec}, {'b': temp, 'mr': temp}, 125.1)

        rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization({'mr': spec}, {'mr': temp})
        F = rvfit.calculate_fisher({'mr': spec}, {'mr': temp}, 125.1)

        rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization({'b': spec, 'mr': spec}, {'b': temp, 'mr': temp})
        F = rvfit.calculate_fisher({'b': spec, 'mr': spec}, {'b': temp, 'mr': temp}, 125.1)

        pass

    def test_calculate_fisher_correction(self):
        rvfit = self.get_rvfit(flux_correction=True)
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        F = rvfit.calculate_fisher({'mr': spec}, {'mr': temp}, 125.1)
        F = rvfit.calculate_fisher({'b': spec, 'mr': spec}, {'b': temp, 'mr': temp}, 125.1)

        rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization({'mr': spec}, {'mr': temp})
        F = rvfit.calculate_fisher({'mr': spec}, {'mr': temp}, 125.1)

        rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization({'b': spec, 'mr': spec}, {'b': temp, 'mr': temp})
        F = rvfit.calculate_fisher({'b': spec, 'mr': spec}, {'b': temp, 'mr': temp}, 125.1)

    def test_fit_lorentz(self):
        rvfit = self.get_rvfit()

        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        rv = np.linspace(-300, 300, 31)
        log_L, phi, chi = rvfit.calculate_log_L({'mr': spec}, {'mr': temp}, rv)

        pp, _ = rvfit.fit_lorentz(rv, log_L)

        y1 = rvfit.lorentz(rv, *pp)

        plt.plot(rv, log_L, 'o')
        plt.plot(rv, y1, '-')
        self.save_fig()

    def test_guess_rv(self):
        rvfit = self.get_rvfit()

        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        rv0 = rvfit.guess_rv({'mr': spec}, {'mr': temp})

    def test_fit_rv_nocorrection(self):
        rv_real = 125.0

        rvfit = self.get_rvfit()

        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=rv_real)

        rv = np.linspace(-300, 300, 31)
        log_L, phi, chi = rvfit.calculate_log_L({'mr': spec}, {'mr': temp}, rv)
        plt.plot(rv, log_L, 'o')

        pp, _ = rvfit.fit_lorentz(rv, log_L)
        y1 = rvfit.lorentz(rv, *pp)
        plt.plot(rv, y1, '-')

        rv0 = rvfit.guess_rv({'mr': spec}, {'mr': temp})
        plt.axvline(rv0, color='k')
    
        rv, rv_err = rvfit.fit_rv({'mr': spec}, {'mr': temp})
        plt.axvline(rv, color='b')
        plt.axvline(rv - rv_err, color='b')
        plt.axvline(rv + rv_err, color='b')
        plt.axvline(rv_real, color='r')

        plt.title(f'RV={rv_real}, RF_fit={rv}+/-{rv_err}')
        plt.xlim(rv_real - 20 * rv_err, rv_real + 20 * rv_err)
        
        self.save_fig()

        F = rvfit.calculate_fisher({'mr': spec}, {'mr': temp}, rv)
        FF = rvfit.calculate_fisher_special({'mr': spec}, {'mr': temp}, rv)

        pass

    def test_fit_rv_correction(self):
        rv_real = 125.0

        rvfit = self.get_rvfit(flux_correction=True)

        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=rv_real)

        rv = np.linspace(-300, 300, 31)
        log_L, phi, chi = rvfit.calculate_log_L({'mr': spec}, {'mr': temp}, rv)
        plt.plot(rv, log_L, 'o')

        pp, _ = rvfit.fit_lorentz(rv, log_L)
        y1 = rvfit.lorentz(rv, *pp)
        plt.plot(rv, y1, '-')

        rv0 = rvfit.guess_rv({'mr': spec}, {'mr': temp})
        plt.axvline(rv0, color='k')
    
        rv, rv_err = rvfit.fit_rv({'mr': spec}, {'mr': temp})
        plt.axvline(rv, color='b')
        plt.axvline(rv - rv_err, color='b')
        plt.axvline(rv + rv_err, color='b')
        plt.axvline(rv_real, color='r')

        plt.title(f'RV={rv_real}, RF_fit={rv}+/-{rv_err}')
        plt.xlim(rv_real - 20 * rv_err, rv_real + 20 * rv_err)
        
        self.save_fig()

    def test_fit_rv_multiple(self):
        rvfit = self.get_rvfit()

        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        rv, rv_err = rvfit.fit_rv({'b': spec, 'mr': spec}, {'b': temp, 'mr': temp})

# Run when profiling

# $ python -m cProfile -o tmp/tmp.prof python/test/pfs/ga/pfsspec/stellar/rvfit/test_rvfit.py
# $ python -m snakeviz --server tmp/tmp.prof

# t = TestRVFit()
# t.setUpClass()
# t.setUp()
# t.test_fit_rv()