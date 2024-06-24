import numpy as np

from pfs.ga.pfsspec.core.util.copy import safe_deep_copy
from .continuummodel import ContinuumModel

class PiecewiseTrace():
    pass

class Piecewise(ContinuumModel):
    """
    Fit the continuum model as a piecewise combination of functions within predefined wavelength ranges.
    """

    # TODO: allow using different function in each piece, with their own name in params?

    def __init__(self, continuum_finder=None, trace=None, orig=None):
        super().__init__(continuum_finder=continuum_finder,
                         trace=trace, orig=orig)

        if not isinstance(orig, Piecewise):
            self.wave_limits = self.get_hydrogen_limits()
            self.wave_limits_dlambda = 1.0

            self.fit_ranges = None
            self.fit_masks = None
            self.fit_overflow = None
            
            self.eval_ranges = None
            self.eval_masks = None
            self.eval_overflow = None
        else:
            self.wave_limits = orig.wave_limits
            self.wave_limits_dlambda = orig.wave_limits_dlambda

            self.fit_ranges = orig.fit_ranges
            self.fit_masks = orig.fit_masks
            self.fit_overflow = orig.fit_overflow
            
            self.eval_ranges = orig.eval_ranges
            self.eval_masks = orig.eval_masks
            self.eval_overflow = orig.eval_overflow

        self.version = 1

    @property
    def name(self):
        return "piecewise"
    
    def create_function(self, i):
        raise NotImplementedError()
            
    def get_constants(self, wave=None):
        """
        Return the constants necessary to evaluate the continuum model
        """

        constants = super().get_constants(wave=wave)

        wave = wave if wave is not None else self.wave

        self.find_limits(wave, self.wave_limits_dlambda)

        limits = []
        for i in range(len(self.fit_ranges)):
            limits.append(self.fit_ranges[i][0])
            limits.append(self.fit_ranges[i][1])

        constants.update({ 
            f'{self.name}_limits': np.array(limits),
            f'{self.name}_limits_dlambda': self.wave_limits_dlambda,
        })

        return constants

    def set_constants(self, constants, wave=None):
        """
        Load the constants necessary to evaluate the continuum model
        """

        super().set_constants(constants, wave=wave)

        wave = wave if wave is not None else self.wave

        if self.version == 1:
            self.wave_limits_dlambda = constants[f'{self.name}_limits_dlambda']
            self.find_limits(wave, self.wave_limits_dlambda)
            
            limits = list(constants[f'{self.name}_limits'])
            for i in range(len(self.fit_ranges)):
                self.fit_ranges[i][0] = limits.pop(0)
                self.fit_ranges[i][1] = limits.pop(0)
        else:
            raise NotImplementedError()

    def save_items(self):
        super().save_items()

    def load_items(self):
        super().load_items()

    def init_wave(self, wave, force=True, omit_overflow=False):
        """
        Initialize the wave vector cache and masks.
        """

        super().init_wave(wave, force=force, omit_overflow=omit_overflow)

        self.find_limits(wave, self.wave_limits_dlambda, force=force)

    def find_limits(self, wave=None, dlambda=None, force=False):

        wave = wave if wave is not None else self.wave
        dlambda = dlambda if dlambda is not None else self.wave_limits_dlambda

        if force or self.fit_masks is None:
            self.fit_masks, self.fit_ranges, self.fit_overflow = self.limits_to_masks(
                wave, self.wave_limits, dlambda=dlambda,
                strict=False, omit_overflow=False
            )
        
        if force or self.eval_masks is None:
            # Same as above but use a dlambda buffer of 0
            # plus extrapolate continuum to the edges

            limits = safe_deep_copy(self.wave_limits)
            limits[0], limits[-1] = None, None

            self.eval_masks, self.eval_ranges, self.eval_overflow = self.limits_to_masks(
                wave, limits, dlambda=0,
                strict=False, omit_overflow=False
            )

    def get_normalized_x(self, x, wave_min, wave_max):
        return (x - wave_min) / (wave_max - wave_min) - 0.5

    def fit_between_limits(self, flux, flux_err=None, wave=None, mask=None):
        """
        Fit a function on the flux between the predefined wavelength ranges.
        """

        wave = wave if wave is not None else self.wave

        pp = []
        for i in range(len(self.fit_masks)):      
            success = False

            # Prepare the function to be fitted
            func = self.create_function(i)

            # Determine the mask marking the range of the piece
            piece_mask = self.fit_masks[i]

            if piece_mask.sum() > 0:
                # Mask within the piece
                custom_mask = piece_mask.copy()
                if mask is not None:
                    custom_mask &= mask
                if self.included_mask is not None:
                    custom_mask &= self.included_mask
                if self.excluded_mask is not None:
                    custom_mask &= ~self.excluded_mask
                
                # Determine the range of normalization
                # this is always based on the predefined limits, regardless of
                # the actual wavelength coverage and the masks
                wave_min, wave_max = self.fit_ranges[i]

                # Prepare data that we're fitting
                x = self.get_normalized_x(self.wave[piece_mask], wave_min, wave_max)
                y = flux[piece_mask]
                w = 1 / flux_err[piece_mask] if flux_err is not None else None          # 1 / sigma
                m = custom_mask[piece_mask]
                
                if m.sum() > func.get_min_point_count():
                    # Find initial parameters
                    success, p0 = func.find_p0(x, y, w=w, mask=m)

                    # TODO: review this, as it's likely wrong            
                    # # To find the initial values, run a broad maximum filter and downsample
                    # # This works only when we fit templates, so the error vector is None
                    # if flux_err is None:
                    #     # TODO: make this a function
                    #     # TODO: make size a variable
                    #     size = 2 * (x.shape[0] // 50) + 1
                    #     shift = size // 2
                    #     idx = np.arange(shift, x.shape[0] - shift) + (np.arange(size) - shift)[:, np.newaxis]
                    #     if x.shape[0] - 2 * shift > func.get_min_point_count():
                    #         p0_found, p0 = func.find_p0(x[shift:-shift], np.max(y[idx], axis=0), w=w[shift:-shift] if w is not None else None)
                    #         if not p0_found:
                    #             p0 = None
                    
                
                    success, p = self.fit_function(
                        i, func, x, y, w=w, p0=p0, mask=m,
                        continuum_finder=self.continuum_finder
                    )

            if not success:
                p = np.full(func.get_param_count(), np.nan)

            pp.append(p)

        return { func.name: np.concatenate(pp)}

    def eval_between_limits(self, params):
        """
        Evaluate the model for each wavelength range, between the limits.
        """

        model = np.full(self.wave.shape, np.nan)
        
        for i in range(len(self.eval_masks)):
            func = self.create_function(i)
            pcount = func.get_param_count()

            piece_mask = self.eval_masks[i]
            wave_min, wave_max = self.fit_ranges[i]         # Must use fit limits!

            if piece_mask.sum() > 0 and wave_min is not None and wave_max is not None:
                x = self.get_normalized_x(self.wave[piece_mask], wave_min, wave_max)
                p = params[func.name][i * pcount: (i + 1) * pcount]
                model[piece_mask] = func.eval(x, p)

        return model

    def get_flux(self, spec):
        if self.use_spec_continuum:
            flux = spec.cont
            flux_err = None
        else:
            flux = spec.flux
            flux_err = spec.flux_err

        return flux, flux_err
            
    def fit_impl(self, flux, flux_err, mask):
                
        mask = self.get_full_mask(mask)
        params = self.fit_between_limits(flux, flux_err=flux_err, mask=mask)
        
        return params

    def eval_impl(self, params):
        model = self.eval_between_limits(params)
        return model
