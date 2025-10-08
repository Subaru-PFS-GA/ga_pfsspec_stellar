import os

from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.core.obsmod.psf import PcaPsf
from ..stellartestbase import StellarTestBase

class TestModelGrid(StellarTestBase):

    def get_test_grid(self, args):
        grid = self.get_bosz_grid()
        grid.init_from_args(args)
        return grid
    
    def get_test_psf(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'subaru/pfs/psf/import/mr.2/pca.h5')
        psf = PcaPsf()
        psf.load(filename)
        return psf

    def test_from_file_arraygrid(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/bosz/bosz_50000/spectra.h5')
        grid = ModelGrid.from_file(filename)
        self.assertIsNotNone(grid.array_grid)
        self.assertIsNone(grid.rbf_grid)
        self.assertIsNone(grid.pca_grid)

    def test_from_file_arraygrid_mmap(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/bosz/bosz_50000/spectra.h5')
        grid = ModelGrid.from_file(filename, mmap_arrays=True)
        self.assertIsNotNone(grid.array_grid)
        self.assertIsNone(grid.rbf_grid)
        self.assertIsNone(grid.pca_grid)

        spec = grid.get_nearest_model(M_H=0., T_eff=4500, log_g=4, C_M=0, a_M=0)
        self.assertIsNotNone(spec)

    def test_from_file_rbf(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/rbf/phoenix/phoenix_HiRes_GK/flux-rbf/spectra.h5')
        grid = ModelGrid.from_file(filename)
        self.assertIsNone(grid.array_grid)
        self.assertIsNotNone(grid.rbf_grid)
        self.assertIsNone(grid.pca_grid)

    def test_from_file_pca_rbf(self):
        filename = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/rbf/phoenix/phoenix_HiRes_GK/pca-rbf/spectra.h5')
        grid = ModelGrid.from_file(filename)
        self.assertIsNone(grid.array_grid)
        self.assertIsNotNone(grid.rbf_grid)
        self.assertIsNotNone(grid.pca_grid)

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

    def test_get_wave_vectors(self):
        args = {}
        grid = self.get_test_grid({})
        grid.wave_type = None
        wave, wave_edges = grid.get_wave_vectors()
        self.assertIs(wave, grid.wave_vacuum)

        grid.wave_type = 'air'
        wave, wave_edges = grid.get_wave_vectors()
        self.assertIs(wave, grid.wave_air)

        grid.wave_type = 'vacuum'
        wave, wave_edges = grid.get_wave_vectors()
        self.assertIs(wave, grid.wave_vacuum)

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

    def test_get_model(self):
        pass

    def test_get_model_at(self):
        args = {}
        psf = self.get_test_psf()
        grid = self.get_test_grid(args)
        idx = grid.grid.get_index(M_H=0., T_eff=4500, log_g=4, C_M=0, a_M=0)
        
        spec = grid.get_model_at(idx)
        self.assertEqual(spec.wave.shape, spec.flux.shape)

        spec = grid.get_model_at(idx, wlim=(6000, 7000))
        self.assertEqual(spec.wave.shape, spec.flux.shape)

        spec = grid.get_model_at(idx, psf=psf)
        self.assertEqual(spec.wave.shape, spec.flux.shape)

        spec = grid.get_model_at(idx, wlim=(6000, 7000), psf=psf)
        self.assertEqual(spec.wave.shape, spec.flux.shape)

    def test_get_nearest_model(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=4500, log_g=4, C_M=0, a_M=0)
        self.assertIsNotNone(spec)

    def test_interpolate_model(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.interpolate_model(M_H=-1.2, T_eff=4125, log_g=4.3, C_M=0, a_M=0)
        self.assertIsNotNone(spec)

    def test_interpolate_model_linear(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.interpolate_model_linear(M_H=-1.2, T_eff=4125, log_g=4.3, C_M=0, a_M=0)
        self.assertIsNotNone(spec)

    def test_interpolate_model_spline(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.interpolate_model_spline(free_param='T_eff', M_H=-1.2, T_eff=4125, log_g=4.3, C_M=0, a_M=0)
        self.assertIsNotNone(spec)