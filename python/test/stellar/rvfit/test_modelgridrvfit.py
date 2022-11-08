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

class TestModelGridRVFit(StellarTestBase):
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
        trace = ModelGridRVFitTrace()
        rvfit = ModelGridRVFit(trace=trace)
        rvfit.template_resampler = FluxConservingResampler()
        rvfit.template_grids = {
            'b': self.get_bosz_grid(),
            'mr': self.get_bosz_grid()
        }

        if flux_correction:
            def polys(wave):
                npoly = 10
                normwave = (wave - wave[0]) / (wave[-1] - wave[0]) * 2 - 1
                polys = np.empty((wave.shape[0], npoly))

                coeffs = np.eye(npoly)
                for i in range(npoly):
                    polys[:, i] = np.polynomial.Chebyshev(coeffs[i])(normwave)

                return polys

            rvfit.basis_functions = polys

        return rvfit

    def test_fit_rv(self):
        rvfit = self.get_rvfit(flux_correction=True)

        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        # rv = np.linspace(-300, 300, 31)
        # log_L, phi, chi = rvfit.calculate_log_L({'mr': spec}, {'mr': temp}, rv)
        # plt.plot(rv, log_L, 'o')

        # pp, _ = rvfit.fit_lorentz(rv, log_L)
        # y1 = rvfit.lorentz(rv, *pp)
        # plt.plot(rv, y1, '-')

        # rv0 = rvfit.guess_rv({'mr': spec}, {'mr': temp})
        # plt.axvline(rv0, color='k')

        params_0 = {
            'M_H': -2.0 * np.random.normal(1, 0.01),
            'T_eff': 4500 * np.random.normal(1, 0.01),
            'log_g': 1.5 * np.random.normal(1, 0.01)
        }

        params_fixed = {
            'C_M': 0,
            'a_M': 0
        }

        params_bounds = {
            'M_H': (-2.5, -1.0),
            'T_eff': (4000, 5000),
            'log_g': (1.0, 3.5)
        }
    
        rv, rv_err, params, params_err = rvfit.fit_rv({'mr': spec}, rv_0=100.0, rv_bounds=(0, 200), params_0=params_0, params_fixed=params_fixed, params_bounds=params_bounds)

        #plt.axvline(rv, color='r')
        #plt.axvline(rv - rv_err, color='r')
        #plt.axvline(rv + rv_err, color='r')

        temp = self.get_template(**params)
        temp = rvfit.process_template(temp, spec, rv)

        mask = (8450 <= spec.wave) & (spec.wave <= 8700)

        plt.plot(spec.wave[mask], spec.flux[mask] / np.median(spec.flux))
        plt.plot(temp.wave[mask], temp.flux[mask] / np.median(temp.flux))
        
        self.save_fig()