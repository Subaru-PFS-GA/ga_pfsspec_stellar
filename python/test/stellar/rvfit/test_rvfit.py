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
from pfs.ga.pfsspec.stellar.rvfit import RVFit, RVFitTrace

class TestRVFit(StellarTestBase):
    def get_test_spectrum(self, M_H=-2.0, T_eff=4500, log_g=1.5, C_M=0, a_M=0):
        grid = self.get_bosz_grid()
        spec = grid.get_nearest_model(M_H=M_H, T_eff=T_eff, log_g=log_g, C_M=C_M, a_M=a_M)
        return grid, spec

    def get_test_psf(self):
        # fn = os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/pfs/psf/import/mr.2/pca.h5')
        # psf = PcaPsf()
        # psf.load(fn, format='h5')

        fn = os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/pfs/psf/import/mr.2/gauss.h5')
        psf = GaussPsf()
        psf.load(fn, format='h5')

        return psf

    def get_template(self, M_H=-2.0, T_eff=4500, log_g=1.5, C_M=0, a_M=0):
        grid = self.get_bosz_grid()
        psf = self.get_test_psf()
        temp = grid.get_nearest_model(M_H=M_H, T_eff=T_eff, log_g=log_g, C_M=C_M, a_M=a_M)
        temp.convolve_psf(psf)
        return temp

    def get_observation(self, noise_level=1.0, rv=0.0, M_H=-2.0, T_eff=4500, log_g=1.5, C_M=0, a_M=0):
        grid, spec = self.get_test_spectrum(M_H=M_H, T_eff=T_eff, log_g=log_g, C_M=C_M, a_M=a_M)

        fn = os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/hsc/filters/HSC-g.txt')
        filter = Filter()
        filter.read(fn)

        fn = os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/pfs/noise/import/sky.see/mr/sky.h5')
        sky = Sky()
        sky.preload_arrays = True
        sky.load(fn, format='h5')

        fn = os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/pfs/noise/import/moon/r/moon.h5')
        moon = Moon()
        moon.preload_arrays = True
        moon.load(fn, format='h5')

        fn = os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/pfs/arms/mr.json')
        detector = Detector()
        detector.load_json(fn)
        detector.psf = self.get_test_psf()

        obs = PfsObservation()
        obs.detector = detector
        obs.sky = sky
        obs.moon = moon

        pp = StellarModelPipeline()
        pp.model_res = grid.resolution or 150000
        pp.mag_filter = filter
        pp.observation = obs
        pp.noise_level = noise_level
        pp.noise_freeze = True
        pp.calibration = FluxCalibrationBias()

        args = {
            'mag': 22,
            'seeing': 0.5,
            'exp_time': 15 * 60,
            'exp_count': 4 * 3,
            'target_zenith_angle': 0,
            'target_field_angle': 0.0,
            'moon_zenith_angle': 45,
            'moon_target_angle': 60,
            'moon_phase': 0.,
            'z': Physics.vel_to_z(rv)
        }

        pp.run(spec, **args)

        return spec

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