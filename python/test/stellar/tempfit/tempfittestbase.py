import os
import numpy as np
import numpy.testing as npt

from pfs.ga.pfsspec.core.physics import Physics
from pfs.ga.pfsspec.core.grid import ArrayGrid
from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid.bosz import Bosz
from pfs.ga.pfsspec.core import Filter
from pfs.ga.pfsspec.sim.obsmod import Detector
from pfs.ga.pfsspec.core.obsmod.psf import PcaPsf, GaussPsf
from pfs.ga.pfsspec.sim.obsmod.background import Sky, Moon
from pfs.ga.pfsspec.sim.obsmod.observations import PfsObservation
from pfs.ga.pfsspec.sim.obsmod.pipelines import StellarModelPipeline
from pfs.ga.pfsspec.sim.obsmod.calibration import FluxCalibrationBias

from test.pfs.ga.pfsspec.stellar.stellartestbase import StellarTestBase

class TempFitTestBase(StellarTestBase):
    def __init__(self, methodName):
        super().__init__(methodName)

        self.rv_real = 100

    def get_test_grid(self):
        return self.get_bosz_grid()

    def get_test_spectrum(self, M_H=-2.0, T_eff=4500, log_g=1.5, C_M=0, a_M=0):
        grid = self.get_test_grid()
        spec = grid.get_nearest_model(M_H=M_H, T_eff=T_eff, log_g=log_g, C_M=C_M, a_M=a_M)
        return grid, spec

    def get_test_psf(self, arm):
        # fn = os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/pfs/psf/import/mr.2/pca.h5')
        # psf = PcaPsf()
        # psf.load(fn, format='h5')

        fn = os.path.join(self.PFSSPEC_DATA_PATH, f'subaru/pfs/psf/import/{arm}.2/gauss.h5')
        psf = GaussPsf()
        psf.load(fn, format='h5')

        return psf

    def get_template(self, M_H=-2.0, T_eff=4500, log_g=1.5, C_M=0, a_M=0):
        grid = self.get_bosz_grid()
        temp = grid.get_nearest_model(M_H=M_H, T_eff=T_eff, log_g=log_g, C_M=C_M, a_M=a_M)
        return temp
    
    def get_normalized_template(self, M_H=-2.0, T_eff=4500, log_g=1.5, C_M=0, a_M=0):
        grid = self.get_bosz_grid()
        temp = grid.get_nearest_model(M_H=M_H, T_eff=T_eff, log_g=log_g, C_M=C_M, a_M=a_M)
        temp.flux /= temp.cont
        temp.cont = None
        return temp

    def get_observation(self, arm='mr', noise_level=1.0, rv=0.0, M_H=-2.0, T_eff=4500, log_g=1.5, C_M=0, a_M=0):
        grid, spec = self.get_test_spectrum(M_H=M_H, T_eff=T_eff, log_g=log_g, C_M=C_M, a_M=a_M)

        fn = os.path.join(self.PFSSPEC_DATA_PATH, f'subaru/hsc/filters/HSC-g.txt')
        filter = Filter()
        filter.read(fn)

        fn = os.path.join(self.PFSSPEC_DATA_PATH, f'subaru/pfs/noise/import/sky.see/{arm}/sky.h5')
        sky = Sky()
        sky.preload_arrays = True
        sky.load(fn, format='h5')

        fn = os.path.join(self.PFSSPEC_DATA_PATH, f'subaru/pfs/noise/import/moon/{arm}/moon.h5')
        moon = Moon()
        moon.preload_arrays = True
        moon.load(fn, format='h5')

        fn = os.path.join(self.PFSSPEC_DATA_PATH, f'subaru/pfs/arms/{arm}.json')
        detector = Detector()
        detector.load_json(fn)
        detector.psf = self.get_test_psf(arm)

        obs = PfsObservation()
        obs.detector = detector
        obs.sky = sky
        obs.moon = moon
        obs.sky_residual = 0.01

        pp = StellarModelPipeline()
        pp.model_res = grid.resolution or 150000
        pp.mag_filter = filter
        pp.observation = obs
        pp.noise_level = noise_level
        pp.noise_freeze = True
        pp.calibration = FluxCalibrationBias()
        pp.calibration.amplitude = 0.05

        args = {
            'mag': 22.0,
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
    
    def get_tempfit(self,
                    flux_correction=False,
                    use_priors=False,
                    **kwargs):
        raise NotImplementedError()
    
    def get_initialized_tempfit(self, /, 
                                flux_correction=False,
                                continuum_fit=False,
                                normalize=False,
                                convolve_template=True,
                                multiple_arms=True,
                                multiple_exp=True,
                                use_priors=True,
                                **kwargs):
        
        if kwargs is None or len(kwargs) == 0:
            params_0 = dict(M_H=-1.5, T_eff=4000, log_g=2.5, a_M=0, C_M=0)
        else:
            params_0 = kwargs
        
        tempfit = self.get_tempfit(flux_correction=flux_correction,
                                   continuum_fit=continuum_fit,
                                   use_priors=use_priors,
                                   **params_0)
       
        if flux_correction:
            phi_shape = (6,)
            chi_shape = (6, 6)
        else:
            phi_shape = ()
            chi_shape = ()

        if multiple_arms:
            arms = ['b', 'mr']
        else:
            arms = ['mr']

        if multiple_exp:
            specs = { k: [self.get_observation(arm=k, rv=self.rv_real, **params_0) for _ in range(2)] for k in arms }
        else:
            specs = { k: self.get_observation(arm=k, rv=self.rv_real, **params_0) for k in arms }

        # TODO: normalize templates
        if not continuum_fit:
            temps = { k: self.get_template(**params_0) for k in arms}
        else:
            temps = { k: self.get_normalized_template(**params_0) for k in arms}

        psfs = { k: self.get_test_psf(k) for k in arms}

        if convolve_template:
            tempfit.psf = psfs
        else:
            tempfit.psf = None

        if normalize:
            tempfit.spec_norm, tempfit.temp_norm = tempfit.get_normalization(specs, temps)

        return tempfit, self.rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0


