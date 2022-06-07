from ..stellartestbase import StellarTestBase

class TestModelGrid(StellarTestBase):

    def get_test_grid(self, args):
        grid = self.get_bosz_grid()
        grid.init_from_args(args)
        return grid

    def test_init_from_args(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual(slice(None), grid.get_wave_slice())

        args = { 'T_eff': [3800, 5000] }
        grid = self.get_test_grid(args)
        self.assertEqual(3800, grid.grid.axes['T_eff'].min)
        self.assertEqual(5000, grid.grid.axes['T_eff'].max)
        self.assertEqual(slice(1, 7, None), grid.grid.slice[1])

        args = { 'T_eff': [3800, 4800] }
        grid = self.get_test_grid(args)
        self.assertEqual(slice(1, 6, None), grid.grid.slice[1])

        args = { 'T_eff': [5000] }
        grid = self.get_test_grid(args)
        self.assertEqual(6, grid.grid.slice[1])

    def test_enumerate_axes(self):
        args = {}
        grid = self.get_test_grid(args)
        axes = grid.enumerate_axes(squeeze=True)
        self.assertEqual(5, len(list(axes)))

        args = {}
        grid = self.get_test_grid(args)
        slice = grid.get_slice()
        axes = grid.enumerate_axes(s=slice, squeeze=True)
        self.assertEqual(5, len(list(axes)))

        args = { 'T_eff': [3800, 4800] }
        grid = self.get_test_grid(args)
        slice = grid.get_slice()
        axes = grid.enumerate_axes(s=slice, squeeze=True)
        self.assertEqual(5, len(list(axes)))

        args = { 'T_eff': [5000] }
        grid = self.get_test_grid(args)
        slice = grid.get_slice()
        axes = grid.enumerate_axes(s=slice, squeeze=False)
        self.assertEqual(5, len(list(axes)))

        args = { 'T_eff': [5000] }
        grid = self.get_test_grid(args)
        slice = grid.get_slice()
        axes = grid.enumerate_axes(s=slice, squeeze=True)
        self.assertEqual(4, len(list(axes)))

    def test_get_shape(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual((14, 66, 11, 6, 4), grid.get_shape())

        args = { 'T_eff': [3800, 5000] }
        grid = self.get_test_grid(args)
        slice = grid.get_slice()
        self.assertEqual((14, 6, 11, 6, 4), grid.get_shape(s=slice))

        args = { 'T_eff': [5000] }
        grid = self.get_test_grid(args)
        slice = grid.get_slice()
        self.assertEqual((14, 11, 6, 4), grid.get_shape(s=slice, squeeze=True))

    def test_get_model_count(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual(139721, grid.get_model_count())

        args = { 'T_eff': [3800, 5000] }
        grid = self.get_test_grid(args)
        self.assertEqual(22113, grid.get_model_count())

        args = { 'T_eff': [5000] }
        grid = self.get_test_grid(args)
        self.assertEqual(3696, grid.get_model_count())

    def test_get_flux_shape(self):
        args = {}
        grid = self.get_test_grid(args)
        self.assertEqual((14, 66, 11, 6, 4, 15404), grid.get_flux_shape())

        args = { 'T_eff': [3800, 5000] }
        grid = self.get_test_grid(args)
        self.assertEqual((14, 6, 11, 6, 4, 15404), grid.get_flux_shape())

        args = { 'T_eff': [5000] }
        grid = self.get_test_grid(args)
        self.assertEqual((14, 1, 11, 6, 4, 15404), grid.get_flux_shape())

        args = { 'T_eff': [3800, 5000], 'wave_lim': [4800, 5600] }
        grid = self.get_test_grid(args)
        self.assertEqual((14, 6, 11, 6, 4, 1542), grid.get_flux_shape())

    def test_get_nearest_model(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=4500, log_g=4, C_M=0, a_M=0)
        self.assertIsNotNone(spec)

    def test_interpolate_model_linear(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.interpolate_model_linear(M_H=-1.2, T_eff=4125, log_g=4.3, C_M=0, a_M=0)
        self.assertIsNotNone(spec)