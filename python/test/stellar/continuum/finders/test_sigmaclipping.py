import os
import numpy as np

from ...stellartestbase import StellarTestBase

from pfs.ga.pfsspec.stellar.continuum.finders import SigmaClipping
from pfs.ga.pfsspec.stellar.continuum.models import PiecewiseChebyshev
from pfs.ga.pfsspec.stellar.continuum import ContinuumModelTrace

class TestSigmaClipping(StellarTestBase):
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
    
    def get_test_finder(self):
        return 
    
    def test_find(self):
        args = {}
        grid = self.get_test_grid(args)
        spec = grid.get_nearest_model(M_H=0., T_eff=4200, log_g=1, C_M=0, a_M=0, wlim=[4000, 9000])

        model = self.get_test_model(spec)
        finder = SigmaClipping()

        iter = 5
        needs_more_iter = True
        mask = np.full(spec.wave.shape, True)
        while iter > 0 and needs_more_iter and mask.sum() > 0:
            params = model.fit_spectrum(spec, mask=mask)
            _, cont, _ = model.eval(params)
            mask, needs_more_iter = finder.find(iter, spec.wave, spec.flux, mask=mask, cont=cont)
            
            iter -= 1