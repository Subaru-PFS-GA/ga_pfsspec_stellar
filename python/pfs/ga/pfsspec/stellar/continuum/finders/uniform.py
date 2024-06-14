import numpy as np
from collections.abc import Iterable

from .continuumfinder import ContinuumFinder

class Uniform(ContinuumFinder):
    """
    Finds control points for continuum fitting by allocating them more or less uniformly.
    """

    def __init__(self, npix=None, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, Uniform):
            self.npix = npix if npix is not None else 10
        else:
            self.npix = npix if npix is not None else orig.npix

    def find(self, iter, wave, flux, /, w=None, mask=None, cont=None):
        t = np.round(np.linspace(0, wave[mask].size, int(wave[mask].size / self.npix)))[1:-1].astype(int)
        m = np.full(wave.size, False)
        m[t] = True

        return m, False