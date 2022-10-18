import numpy as np
from collections.abc import Iterable

from ..continuumfinder import ContinuumFinder

class SigmaClipping(ContinuumFinder):
    def __init__(self, sigma=None, max_iter=None, orig=None):
        super().__init__(max_iter=max_iter, orig=orig)

        if not isinstance(orig, SigmaClipping):
            self.sigma = sigma if sigma is not None else [1, 3]
        else:
            self.sigma = sigma if sigma is not None else orig.sigma

    def find(self, iter, x, y, w=None, mask=None, model=None):
        if model is None:
            raise ValueError("Model must not be None for sigma clipping continuum finder.")

        sigma = self.sigma if isinstance(self.sigma, Iterable) else [self.sigma, self.sigma]

        # Filter out values that are below sigma[0] or above sigma[1] from the model
        s = np.std(y[mask] - model[mask])
        m = (model - sigma[0] * s < y) & (y < model + sigma[1] * s)
        ms = (~m[mask]).sum()       # number of newly excluded points

        # if mask is not None:
        #     m &= mask

        return m, iter < self.max_iter and ms > 0