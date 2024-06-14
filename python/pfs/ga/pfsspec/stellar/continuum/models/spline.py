import numpy as np
from scipy.interpolate import splrep, splev

from .continuummodel import ContinuumModel

class Spline(ContinuumModel):
    def __init__(self, continuum_finder=None, trace=None, orig=None):    
        super().__init__(continuum_finder=continuum_finder,
                         trace=trace, orig=orig)
        
        if not isinstance(orig, Spline):
            self.deg = 3                # Degree of the spline.
            self.npix = 200             # Minimum separation of knots in spectra pixels.
        else:
            self.deg = orig.deg
            self.npix = orig.npix

        self.version = 1

    @property
    def name(self):
        return 'spline'
    
    def add_args(self, parser):
        super().add_args(parser)

        # TODO

    def init_from_args(self, args):
        super().init_from_args(args)

        # TODO
        
    def get_constants(self, wave=None):
        """
        Return the constants necessary to evaluate the continuum model
        """

        constants = super().get_constants(wave=wave)
        constants.update({
            f'{self.name}_deg': np.array(self.deg),
        })

        return constants
    
    def set_constants(self, constants, wave=None):
        """
        Load the constants necessary to evaluate the continuum model
        """

        super().set_constants(constants, wave=wave)

        if self.version == 1:
            self.deg = constants[f'{self.name}_deg']
        
    def fit_impl(self, flux, flux_err, mask):
        """
        Fit the function.
        """
        
        wave = self.wave
        
        mask = mask.copy() if mask is not None else np.full(self.wave.shape, True)
        if self.included_mask is not None:
            mask &= self.included_mask
        if self.excluded_mask is not None:
            mask &= ~self.excluded_mask

        # Find the knots for the weighted least squares spline fit
        # Given a wave vector, draw random points from it so that the separation of
        # two neighbors is at least 1/npix of the total range.
        # TODO: this probably needs update given the masks
        knots = np.round(np.linspace(0, wave[mask].size, int(wave[mask].size / self.npix)))[1:-1].astype(int)

        w = 1 / flux_err[mask] ** 2 if flux_err is not None else None
        t, c, k = splrep(wave[mask], flux[mask], w=w, t=wave[mask][knots], k=self.deg)

        return {
            f'{self.name}_t': t,
            f'{self.name}_c': c
        }

    def eval_impl(self, params):
        """
        Evaluate the function.
        """
        t = params[f'{self.name}_t']
        c = params[f'{self.name}_c']

        spline = (t, c, self.deg)
        model = splev(self.wave, spline) 
        return model
