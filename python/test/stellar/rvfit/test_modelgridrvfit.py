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

class TestModelGridRVFit(RVFitTestBase):
    
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

            rvfit.flux_corr = True
            rvfit.flux_corr_basis = polys

        rvfit.params_fixed = [ 'a_M', 'C_M']

        return rvfit

    def test_calculate_fisher_correction(self):
        rvfit = self.get_rvfit(flux_correction=True)
        temp = self.get_template(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)
        spec = self.get_observation(rv=125)

        rv_0 = 125.1
        params_0 = dict(M_H=-1.5, T_eff=4000, log_g=1, a_M=0, C_M=0)

        F = rvfit.calculate_fisher({'mr': spec}, rv_0, params_0)
        F = rvfit.calculate_fisher({'b': spec, 'mr': spec}, rv_0, params_0)

        rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization({'mr': spec}, {'mr': temp})
        F = rvfit.calculate_fisher({'mr': spec}, {'mr': temp}, rv_0, params_0)

        rvfit.spec_norm, rvfit.temp_norm = rvfit.get_normalization({'b': spec, 'mr': spec}, {'b': temp, 'mr': temp})
        F = rvfit.calculate_fisher({'b': spec, 'mr': spec}, {'b': temp, 'mr': temp}, 125.1)

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