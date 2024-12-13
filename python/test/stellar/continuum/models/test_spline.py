import os
import numpy as np

from ...stellartestbase import StellarTestBase

from pfs.ga.pfsspec.stellar.continuum.models import Spline
from pfs.ga.pfsspec.stellar.continuum import ContinuumModelTrace

class TestSpline(StellarTestBase):
    def get_test_grid(self, args):
        #grid = self.get_bosz_grid()
        grid = self.get_phoenix_grid()
        #grid = self.get_phoenix_pca_grid()
        grid.init_from_args(args)
        return grid
    
    def get_test_model(self, spec, included_ranges=None, excluded_ranges=None):
        trace = ContinuumModelTrace(
            figdir=os.path.expandvars('${PFSSPEC_TEST}'),
            logdir=os.path.expandvars('${PFSSPEC_TEST}'),
        )
        trace.plot_fit_start = True
        trace.plot_fit_end = True
        trace.plot_fit_iter = False

        model = Spline(control_points=100, trace=trace)
        model.included_ranges = included_ranges
        model.excluded_ranges = excluded_ranges
        model.init_wave(spec.wave)

        return model

    def test_fit(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=3800, log_g=1, C_M=0, a_M=0, wlim=[4000, 9000])

        model = self.get_test_model(spec)

        params = model.fit_spectrum(spec)
        self.assertIsNotNone(params['spline_t'])
        self.assertIsNotNone(params['spline_c'])

    def test_fit_mask(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=3800, log_g=1, C_M=0, a_M=0, wlim=[4000, 9000])

        mask = np.full(spec.wave.shape, True)

        model = self.get_test_model(spec)

        params = model.fit_spectrum(spec, mask=mask)

        self.assertIsNotNone(params['spline_t'])
        self.assertIsNotNone(params['spline_c'])

    def test_fit_mask_include_exclude(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=3800, log_g=1, C_M=0, a_M=0, wlim=[4000, 9000])

        mask = np.full(spec.wave.shape, True)

        model = self.get_test_model(spec, included_ranges=[[4500, 8000]], excluded_ranges=[[4800, 4900]])

        params = model.fit_spectrum(spec, mask=mask)
        
        self.assertIsNotNone(params['spline_t'])
        self.assertIsNotNone(params['spline_c'])

    def test_fit_wlim(self):
        args = {
            'wave_lim': [7500, 8200]
        }
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=3800, log_g=1, C_M=0, a_M=0)
        
        model = self.get_test_model(spec)

        params = model.fit_spectrum(spec)

        self.assertIsNotNone(params['spline_t'])
        self.assertIsNotNone(params['spline_c'])

    def test_eval(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=4500, log_g=4, C_M=0, a_M=0)
        model = self.get_test_model(spec)
        params = model.fit_spectrum(spec)
        cont = model.eval(params)

    def test_normalize(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=4500, log_g=4, C_M=0, a_M=0)
        model = self.get_test_model(spec)
        params = model.fit_spectrum(spec)
        model.normalize(spec, params)

    def test_denormalize(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=4500, log_g=4, C_M=0, a_M=0)
        model = self.get_test_model(spec)
        params = model.fit_spectrum(spec)
        model.normalize(spec, params)
        model.denormalize(spec, params)