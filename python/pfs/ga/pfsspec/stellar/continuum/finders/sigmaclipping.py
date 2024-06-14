import numpy as np
from collections.abc import Iterable

from .continuumfinder import ContinuumFinder

class SigmaClipping(ContinuumFinder):
    """
    Finds control points for continuum fitting using the sigma clipping method.
    """

    def __init__(self, max_iter=None, sigma=None, orig=None):
        super().__init__(orig=orig)

        # TODO: move max iter to the continuum model fitting settings
        if not isinstance(orig, SigmaClipping):
            self.max_iter = max_iter if max_iter is not None else 5
            self.sigma = sigma if sigma is not None else [1, 3]
        else:
            self.max_iter = max_iter if max_iter is not None else orig.max_iter
            self.sigma = sigma if sigma is not None else orig.sigma

    def find(self, iter, wave, flux, /, w=None, mask=None, cont=None):
        if cont is None:
            raise ValueError("Continuum model must not be None for sigma clipping continuum finder.")

        sigma = self.sigma if isinstance(self.sigma, Iterable) else [self.sigma, self.sigma]
        mask = mask if mask is not None else ()

        # Filter out values that are below sigma[0] or above sigma[1] from the model
        s = np.std(flux[mask] - cont[mask])
        m = (cont - sigma[0] * s < flux) & (flux < cont + sigma[1] * s)
        ms = (~m[mask]).sum()       # number of newly excluded points

        return m, iter < self.max_iter and ms > 0