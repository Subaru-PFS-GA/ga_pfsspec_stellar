import os
import numpy as np

from ...stellartestbase import StellarTestBase

from pfs.ga.pfsspec.stellar.continuum.models import PiecewiseChebyshev
from pfs.ga.pfsspec.stellar.continuum import ContinuumModelTrace

class TestChebyshev(StellarTestBase):
    def get_test_grid(self, args):
        #grid = self.get_bosz_grid()
        grid = self.get_phoenix_grid()
        #grid = self.get_phoenix_pca_grid()
        grid.init_from_args(args)
        return grid
    
    def get_test_model(self, spec):
        trace = ContinuumModelTrace(
            figdir=os.path.expandvars('${PFSSPEC_TEST}'),
            logdir=os.path.expandvars('${PFSSPEC_TEST}'),
        )
        trace.plot_fit_start = True
        trace.plot_fit_end = True
        trace.plot_fit_iter = False

        model = PiecewiseChebyshev(trace=trace)
        model.init_wave(spec.wave)

        return model

    def test_find_limits(self):
        # By default, the model uses Hydrogen ionization limits to define the ranges

        args = {}
        grid = self.get_test_grid(args)
        model = PiecewiseChebyshev()
        model.find_limits(grid.wave, model.wave_limits_dlambda)
        self.assertEqual(4, len(model.fit_ranges))
        self.assertEqual(4, len(model.fit_masks))
        self.assertEqual(4, len(model.eval_ranges))
        self.assertEqual(4, len(model.eval_masks))

    def test_fit(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=3800, log_g=1, C_M=0, a_M=0, wlim=[4000, 9000])

        model = self.get_test_model(spec)

        params = model.fit_spectrum(spec)
        self.assertEqual((28,), params['chebyshev'].shape)

    def test_fit_mask(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=3800, log_g=1, C_M=0, a_M=0, wlim=[4000, 9000])

        mask = np.full(spec.wave.shape, True)

        model = PiecewiseChebyshev()
        model.init_wave(spec.wave)
        params = model.fit_spectrum(spec, mask=mask)
        self.assertEqual((28,), params['chebyshev'].shape)

    def test_fit_wlim(self):
        args = {
            'wave_lim': [7500, 8200]
        }
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=3800, log_g=1, C_M=0, a_M=0)
        model = PiecewiseChebyshev()
        model.init_wave(spec.wave)
        params = model.fit_spectrum(spec)
        self.assertEqual((28,), params['chebyshev'].shape)

    def test_fit_nocont(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=3800, log_g=1, C_M=0, a_M=0)
        spec.cont = None
        model = PiecewiseChebyshev()
        model.init_wave(spec.wave)
        params = model.fit_spectrum(spec)
        self.assertEqual((28,), params['chebyshev'].shape)

    def test_eval(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=4500, log_g=4, C_M=0, a_M=0)
        model = PiecewiseChebyshev()
        model.init_wave(spec.wave)
        params = model.fit_spectrum(spec)
        model.init_wave(spec.wave)
        cont = model.eval(params)

    def test_normalize(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=4500, log_g=4, C_M=0, a_M=0)
        model = PiecewiseChebyshev()
        model.init_wave(spec.wave)
        params = model.fit_spectrum(spec)
        model.normalize(spec, params)

    def test_denormalize(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=4500, log_g=4, C_M=0, a_M=0)
        model = PiecewiseChebyshev()
        model.init_wave(spec.wave)
        params = model.fit_spectrum(spec)
        model.normalize(spec, params)
        model.denormalize(spec, params)