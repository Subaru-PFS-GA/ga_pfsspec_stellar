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
        """
        Return a mask which is True where the continuum pixels are.
        
        This particular continuum finder returns every npix-th pixel as a continuum pixel,
        to be used in conjunction with the spline fitter.
        """
        
        self.init_wave(wave)

        t = np.round(np.linspace(0, wave[mask].size, int(wave[mask].size / self.npix)))[1:-1].astype(int)
        cont_mask = np.full(wave.size, False)
        cont_mask[t] = True

        return cont_mask, False