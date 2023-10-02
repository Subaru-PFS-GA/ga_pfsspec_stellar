import numpy as np

from ...stellartestbase import StellarTestBase

from pfs.ga.pfsspec.stellar.continuum.models.alex import Alex, AlexContinuumModelTrace

class TestAlexContinuumModel(StellarTestBase):
    def get_test_grid(self, args):
        #grid = self.get_bosz_grid()
        grid = self.get_phoenix_grid()
        #grid = self.get_phoenix_pca_grid()
        grid.init_from_args(args)
        return grid

    def test_get_nearest_model(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=4500, log_g=4, C_M=0, a_M=0)
        self.assertIsNotNone(spec)

    def test_find_limits(self):
        args = {}
        grid = self.get_test_grid(args)
        trace = AlexContinuumModelTrace()
        model = Alex(trace)
        model.find_limits(grid.wave)
        self.assertEqual(3, len(model.blended_fit_masks))

    def test_get_convex_hull(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=4500, log_g=4, C_M=0, a_M=0)

        trace = AlexContinuumModelTrace()
        model = Alex(trace)
        x, y = model.get_convex_hull(spec.wave, spec.flux)

        pass

    def test_fit(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=3800, log_g=1, C_M=0, a_M=0)
        trace = AlexContinuumModelTrace()
        model = Alex(trace)
        model.init_wave(spec.wave)
        params = model.fit(spec)
        self.assertEqual((21,), params['legendre'].shape)
        self.assertEqual((5,), params['blended_0'].shape)
        self.assertEqual((5,), params['blended_1'].shape)
        self.assertEqual((5,), params['blended_2'].shape)

    def test_fit_nocont(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=8000, log_g=1, C_M=0, a_M=0)
        spec.cont = None
        trace = AlexContinuumModelTrace()
        model = Alex(trace)
        model.init_wave(spec.wave)
        params = model.fit(spec)
        self.assertEqual((21,), params['legendre'].shape)
        self.assertEqual((5,), params['blended_0'].shape)
        self.assertEqual((5,), params['blended_1'].shape)
        self.assertEqual((5,), params['blended_2'].shape)

    def test_eval(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=4500, log_g=4, C_M=0, a_M=0)
        trace = AlexContinuumModelTrace()
        model = Alex(trace)
        model.init_wave(spec.wave)
        params = model.fit(spec)
        cont = model.eval(params)

    def test_normalize(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=4500, log_g=4, C_M=0, a_M=0)
        trace = AlexContinuumModelTrace()
        model = Alex(trace)
        model.init_wave(spec.wave)
        params = model.fit(spec)
        model.normalize(spec, params)

    def test_denormalize(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=4500, log_g=4, C_M=0, a_M=0)
        trace = AlexContinuumModelTrace()
        model = Alex(trace)
        model.init_wave(spec.wave)
        params = model.fit(spec)
        model.normalize(spec, params)
        model.denormalize(spec, params)
