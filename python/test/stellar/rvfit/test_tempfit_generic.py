import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from pfs.ga.pfsspec.core.obsmod.resampling import FluxConservingResampler
from pfs.ga.pfsspec.stellar.rvfit import RVFit, TempFitTrace

from .tempfittestbase import TempFitTestBase

class TestTempFitFluxCorr(TempFitTestBase):

    def get_tempfit(self,
                    flux_correction=False,
                    use_priors=False,
                    **kwargs):
        
        trace = TempFitTrace()

        tempfit = RVFit(trace=trace, correction_model=None)
        tempfit.mcmc_burnin = 5       # Just a few MCMC steps to make it fast
        tempfit.mcmc_samples = 5
        tempfit.template_resampler = FluxConservingResampler()

        if use_priors:
            tempfit.rv_prior = lambda rv: -(rv - self.rv_real)**2 / 100**2
        else:
            tempfit.rv_prior = None

        return tempfit
    
    
    def test_get_observation(self):
        spec = self.get_observation()
        
        spec.plot(xlim=(7000, 9000))
        self.save_fig()

    def test_get_normalization(self):
        tempfit = self.get_tempfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        spec_norm, temp_norm = tempfit.get_normalization({'mr': spec}, {'mr': temp})

    def test_process_spectrum(self):
        tempfit = self.get_tempfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation()

        sp = tempfit.process_spectrum('', 0, spec)

        tempfit.spec_norm, tempfit.temp_norm = tempfit.get_normalization({'mr': spec}, {'mr': temp})
        sp = tempfit.process_spectrum('', 0, spec)

    def test_process_template(self):
        spec = self.get_observation(arm='mr')
        
        tempfit = self.get_tempfit()
        tempfit.determine_wlim({ 'mr': spec }, (-300, 300))
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        tm = tempfit.process_template('mr', temp, spec, 100)

        psf = self.get_test_psf(arm='mr')
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        tm = tempfit.process_template('mr', temp, spec, 100, psf=psf)

        tempfit.spec_norm, tempfit.temp_norm = tempfit.get_normalization({'mr': spec}, {'mr': temp})
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        tm = tempfit.process_template('mr', temp, spec, 100, psf=psf)

    def test_diff_template(self):
        tempfit = self.get_tempfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)

        wave, dfdl = tempfit.diff_template(temp)
        wave, dfdl = tempfit.diff_template(temp, np.linspace(3000, 9000, 6000))

    def test_log_diff_template(self):
        tempfit = self.get_tempfit()
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)

        wave, dfdl = tempfit.log_diff_template(temp)
        wave, dfdl = tempfit.log_diff_template(temp, np.linspace(3000, 9000, 6000))

    def test_determine_wlim(self):
        tempfit, rv_real, specs, temps, psfs, phi_shape, chi_shape, params_0 = \
            self.get_initialized_tempfit(
                flux_correction=True,
                normalize=True,
                convolve_template=True,
                multiple_arms=True,
                multiple_exp=True,
                use_priors=False)
        
        # Test different types of freedom
        wlim = tempfit.determine_wlim(specs, rv_bounds=(-500, 500), per_arm=False, per_exp=False)
        self.assertIsInstance(wlim, tuple)

        wlim = tempfit.determine_wlim(specs, rv_bounds=(-500, 500), per_arm=False, per_exp=True)
        self.assertIsInstance(wlim, list)
        self.assertIsInstance(wlim[0], tuple)

        wlim = tempfit.determine_wlim(specs, rv_bounds=(-500, 500), per_arm=True, per_exp=False)
        self.assertIsInstance(wlim, dict)
        self.assertIsInstance(wlim['b'], tuple)

        wlim = tempfit.determine_wlim(specs, rv_bounds=(-500, 500), per_arm=True, per_exp=True)
        self.assertIsInstance(wlim, dict)
        self.assertIsInstance(wlim['b'], list)
        self.assertIsInstance(wlim['b'][0], tuple)