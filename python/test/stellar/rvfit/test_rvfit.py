import os
import numpy as np
import matplotlib.pyplot as plt
from pfs.ga.pfsspec.core.physics import Physics

from test.pfs.ga.pfsspec.stellar.stellartestbase import StellarTestBase
from pfs.ga.pfsspec.core.grid import ArrayGrid
from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid.bosz import Bosz
from pfs.ga.pfsspec.core import Filter
from pfs.ga.pfsspec.sim.obsmod import Sky, Moon, Detector
from pfs.ga.pfsspec.sim.obsmod.observations import PfsObservation
from pfs.ga.pfsspec.sim.obsmod.pipelines import StellarModelPipeline
from pfs.ga.pfsspec.core.psf import PcaPsf, GaussPsf
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

        spec.flux = spec.flux + noise_level * spec.flux_err * np.random.normal(size=spec.flux.shape)

        return spec

    def get_rvfit(self):
        rvfit = RVFit()
        rvfit.grid = self.get_bosz_grid()
        rvfit.psf = self.get_test_psf()
        return rvfit

    def test_get_observation(self):
        spec = self.get_observation()
        
        spec.plot(xlim=(7000, 9000))
        self.save_fig()

    def test_get_template(self):
        rvfit = self.get_rvfit()

        temp = rvfit.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)

    def test_rebin_template(self):
        rvfit = self.get_rvfit()

        temp = rvfit.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation()

        temp = rvfit.rebin_template(temp, 100, spec)

    def test_get_log_L(self):
        rvfit = self.get_rvfit()
        temp = rvfit.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        # Test with scalar
        rv = 100.0
        y0, _, _ = rvfit.get_log_L(spec, temp, rv)
        plt.plot(rv, y0, 'o')

        # Test with vector
        rv = np.linspace(-300, 300, 31)
        y0, _, _ = rvfit.get_log_L(spec, temp, rv)

        plt.plot(rv, y0)
        self.save_fig()

    def test_get_fisher(self):
        rvfit = self.get_rvfit()
        temp = rvfit.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        F = rvfit.get_fisher(spec, temp, 125.1)

        pass

    def test_fit_lorentz(self):
        rvfit = self.get_rvfit()

        temp = rvfit.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
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

        temp = rvfit.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        rv0 = rvfit.guess_rv(spec, temp)

    def test_fit_rv(self):
        rvfit = self.get_rvfit()

        temp = rvfit.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        rv = np.linspace(-300, 300, 31)
        y0, _, _ = rvfit.get_log_L(spec, temp, rv)
        plt.plot(rv, y0, 'o')

        pp, _ = rvfit.fit_lorentz(rv, y0)
        y1 = rvfit.lorentz(rv, *pp)
        plt.plot(rv, y1, '-')

        rv0 = rvfit.guess_rv(spec, temp)
        plt.axvline(rv0, color='k')
    
        rv = rvfit.fit_rv(spec, temp)
        plt.axvline(rv, color='r')
        
        self.save_fig()

    def test_run(self):
        
        
        rvfit = RVFit()
        rvfit.grid = self.get_bosz_grid()
        rvfit.psf = self.get_test_psf()
        
        rvfit.run(spec)
