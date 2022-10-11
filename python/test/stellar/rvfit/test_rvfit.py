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
from pfs.ga.pfsspec.stellar.rvfit import RVFit

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
            'redshift': Physics.vel_to_z(rv)
        }

        pp.run(spec, **args)

        return spec

    def get_rvfit(self):
        rvfit = RVFit()
        rvfit.template_resampler = FluxConservingResampler()
        return rvfit

    def test_get_observation(self):
        spec = self.get_observation()
        
        spec.plot(xlim=(7000, 9000))
        self.save_fig()

    def test_process_template(self):
        spec = self.get_observation()
        
        rvfit = self.get_rvfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        temp = rvfit.process_template(temp, spec, 100)

        rvfit.template_psf = self.get_test_psf()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        temp = rvfit.process_template(temp, spec, 100)

    def test_get_log_L(self):
        rvfit = self.get_rvfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        # Test with scalar
        rv = 100.0
        y0, _, _ = rvfit.get_log_L(spec, temp, rv)
        plt.plot(rv, y0, 'o')

        # Test with vector
        rv = np.linspace(-300, 300, 31)
        y0, _, _ = rvfit.get_log_L(spec, temp, rv)

        # Test with multiple spectra
        rv = 100.0
        y0, _, _ = rvfit.get_log_L([spec, spec], [temp, temp], rv)
        plt.plot(rv, y0, 'o')

        plt.plot(rv, y0)
        self.save_fig()

    def test_get_fisher(self):
        rvfit = self.get_rvfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        F = rvfit.get_fisher(spec, temp, 125.1)
        F = rvfit.get_fisher([spec], [temp], 125.1)

        pass

    def test_fit_lorentz(self):
        rvfit = self.get_rvfit()

        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        rv = np.linspace(-300, 300, 31)
        y0, _, _ = rvfit.get_log_L(spec, temp, rv)

        pp, _ = rvfit.fit_lorentz(rv, y0)

        y1 = rvfit.lorentz(rv, *pp)

        plt.plot(rv, y0, 'o')
        plt.plot(rv, y1, '-')
        self.save_fig()

    def test_guess_rv(self):
        rvfit = self.get_rvfit()

        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        rv0 = rvfit.guess_rv(spec, temp)

    def test_fit_rv(self):
        rvfit = self.get_rvfit()

        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        rv = np.linspace(-300, 300, 31)
        y0, _, _ = rvfit.get_log_L(spec, temp, rv)
        plt.plot(rv, y0, 'o')

        pp, _ = rvfit.fit_lorentz(rv, y0)
        y1 = rvfit.lorentz(rv, *pp)
        plt.plot(rv, y1, '-')

        rv0 = rvfit.guess_rv(spec, temp)
        plt.axvline(rv0, color='k')
    
        rv, rv_err = rvfit.fit_rv(spec, temp)
        plt.axvline(rv, color='r')
        plt.axvline(rv - rv_err, color='r')
        plt.axvline(rv + rv_err, color='r')
        
        self.save_fig()

    def test_fit_rv_multiple(self):
        rvfit = self.get_rvfit()

        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        rv, rv_err = rvfit.fit_rv([spec, spec], [temp, temp])

# Run when profiling

# $ python -m cProfile -o tmp/tmp.prof python/test/pfs/ga/pfsspec/stellar/rvfit/test_rvfit.py
# $ python -m snakeviz --server tmp/tmp.prof

# t = TestRVFit()
# t.setUpClass()
# t.setUp()
# t.test_fit_rv()