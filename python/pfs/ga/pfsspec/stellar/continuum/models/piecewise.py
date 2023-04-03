import numpy as np

from pfs.ga.pfsspec.core.util.copy import safe_deep_copy
from pfs.ga.pfsspec.core import Physics
from pfs.ga.pfsspec.stellar.continuum import ContinuumModel

class PiecewiseTrace():
    pass

class Piecewise(ContinuumModel):
    """
    Piecewise continuum fit between Hydrogen photoionization lines.
    """

    # TODO: allow using different function in each piece, with their own name in params?

    def __init__(self, use_log=None, use_continuum=None, continuum_finder=None, trace=None, orig=None):
        super().__init__(continuum_finder=continuum_finder, trace=trace, orig=orig)

        if not isinstance(orig, Piecewise):
            self.version = 1

            self.use_log = use_log if use_log is not None else False                    # Fit log flux instead of flux
            self.use_continuum = use_continuum if use_continuum is not None else False  # Fit continuum instead of flux

            limits = [2530,] + Physics.HYDROGEN_LIMITS + [17500,]
            self.photo_limits = Physics.air_to_vac(np.array(limits))
            self.limits_dlambda = 1

            self.fit_masks = None
            self.fit_limits = None
            self.eval_masks = None
            self.eval_limits = None
        else:
            self.version = orig.version

            self.use_log = use_log if use_log is not None else orig.use_log
            self.use_continuum = use_continuum if use_continuum is not None else orig.use_continuum

            self.photo_limits = orig.photo_limits
            self.limits_dlambda = orig.limits_dlambda

            self.fit_masks = orig.fit_masks
            self.fit_limits = orig.fit_limits
            self.eval_masks = orig.eval_masks
            self.eval_limits = orig.eval_limits

    def get_constants(self, wave):
        """
        Return the constants necessary to evaluate the continuum model
        """

        self.find_limits(wave, self.limits_dlambda)

        limits = []
        for i in range(len(self.fit_limits)):
            limits.append(self.fit_limits[i][0])
            limits.append(self.fit_limits[i][1])

        return { 
            'piecewise_version': self.version,
            'piecewise_limits_dlambda': self.limits_dlambda,
            'piecewise_limits': np.array(limits)
        }

    def set_constants(self, wave, constants):
        """
        Load the constants necessary to evaluate the continuum model
        """

        self.version = int(constants['piecewise_version'])

        if self.version == 1:
            self.limits_dlambda = constants['piecewise_limits_dlambda']
            self.find_limits(wave, self.limits_dlambda)
            
            limits = list(constants['piecewise_limits'])
            for i in range(len(self.fit_limits)):
                self.fit_limits[i][0] = limits.pop(0)
                self.fit_limits[i][1] = limits.pop(0)
        else:
            raise NotImplementedError()

    def save_items(self):
        raise NotImplementedError()

    def load_items(self):
        raise NotImplementedError()

    def init_wave(self, wave, force=True):
        self.find_limits(wave, self.limits_dlambda, force=force)
        self.wave = wave

    def create_function(self):
        raise NotImplementedError()

    def find_masks_between_limits(self, wave, dlambda):
        masks = []
        limits = []

        for i in range(len(self.photo_limits) - 1):
            mask = (wave >= self.photo_limits[i] + dlambda) & (wave < self.photo_limits[i + 1] - dlambda)

            masks.append(mask)
            wm = wave[mask]
            
            # TODO: limits should be independt of the wave grid if we want to transfer the model
            #       from grid to grid

            limits.append([self.photo_limits[i] + dlambda, self.photo_limits[i + 1] - dlambda])

            # if wm.size > 0:
            #     limits.append([wave[mask].min(), wave[mask].max()])
            # else:
            #     limits.append([np.nan, np.nan])

        return masks, limits

    def find_limits(self, wave, dlambda, force=False):
        if force or self.fit_masks is None:
            self.fit_masks, self.fit_limits = self.find_masks_between_limits(wave, dlambda=dlambda)
        
        if force or self.eval_masks is None:
            self.eval_masks, self.eval_limits = self.find_masks_between_limits(wave, dlambda=0)
            
            # Extrapolate continuum to the edges
            # Equality must be allowed here because eval_limits are calculated by taking
            # wave[mask].min/max which are the actual wavelength grid values
            self.eval_masks[0] = (wave <= self.eval_limits[0][1])
            self.eval_masks[-1] = (wave >= self.eval_limits[-1][0])

    def get_normalized_x(self, x, wave_min, wave_max):
        return (x - wave_min) / (wave_max - wave_min) - 0.5

    def fit_between_limits(self, flux, flux_err=None):
        func = self.create_function()

        pp = []
        for i in range(len(self.fit_masks)):
            piece_mask = self.fit_masks[i]
            wave_min, wave_max = self.fit_limits[i]

            x = self.get_normalized_x(self.wave[piece_mask], wave_min, wave_max)
            y = flux[piece_mask]
            w = 1 / flux_err if flux_err is not None else None          # 1 / sigma
            p0 = None
            
            # To find the initial values, run a broad maximum filter and downsample
            # This works only when we fit templates, so the error vector is None
            if flux_err is None:
                # TODO: make this a function
                # TODO: make size a variable
                size = 2 * (x.shape[0] // 50) + 1
                shift = size // 2
                idx = np.arange(shift, x.shape[0] - shift) + (np.arange(size) - shift)[:, np.newaxis]
                if x.shape[0] - 2 * shift > func.get_min_point_count():
                    p0_found, p0 = func.find_p0(x[shift:-shift], np.max(y[idx], axis=0), w=w[shift:-shift] if w is not None else None)
                    if not p0_found:
                        p0 = None
            
            if x.shape[0] > func.get_min_point_count():
                success, p = self.fit_function(i, func, x, y, w=w, p0=p0,
                    continuum_finder=self.continuum_finder)
            else:
                success, p = False, np.full(func.get_param_count(), np.nan)

            pp.append(p)

        return { func.name: np.concatenate(pp)}

    def eval_between_limits(self, pp):
        model = np.full(self.wave.shape, np.nan)
        func = self.create_function()
        pcount = func.get_param_count()

        for i in range(len(self.eval_masks)):
            piece_mask = self.eval_masks[i]
            wave_min, wave_max = self.fit_limits[i]         # Must use fit limits!

            if wave_min is not None and wave_max is not None:
                x = self.get_normalized_x(self.wave[piece_mask], wave_min, wave_max)
                p = pp[func.name][i * pcount: (i + 1) * pcount]
                model[piece_mask] = func.eval(x, p)

        return model

    def get_flux(self, spec):
        if self.use_continuum:
            flux = spec.cont
            flux_err = None
        else:
            flux = spec.flux
            flux_err = spec.flux_err

        return flux, flux_err

    def transform_flux_forward(self, flux, flux_err=None):
        if self.use_log:
            flux_err = (flux_err / flux) if flux_err is not None else None
            flux = np.log(flux)

        return flux, flux_err

    def transform_flux_reverse(self, flux, flux_err=None):
        if self.use_log:
            flux = np.ext(flux)
            flux_err = flux_err * flux if flux_err is not None else None

        return flux, flux_err
            
    def fit(self, spec):
        flux, flux_err = self.get_flux(spec)
        flux, flux_err = self.transform_flux_forward(flux, flux_err)
        params = self.fit_between_limits(flux, flux_err=flux_err)
        return params

    def eval(self, params):
        flux = self.eval_between_limits(params)
        flux, _ = self.transform_flux_reverse(flux)
        return self.wave, flux

    def normalize(self, spec, params):
        def normalize_vector(data, model):
            return data / model if data is not None else None

        _, model = self.eval(params)

        spec.wave = self.wave
        spec.flux = normalize_vector(spec.flux, model)
        spec.flux_err = normalize_vector(spec.flux_err, model)
        spec.cont = normalize_vector(spec.cont, model)

        spec.append_history(f'Spectrum is normalized using model `{type(self).__name__}`.')

    def denormalize(self, spec, params, s=None):
        def denormalize_vector(data, model):
            return data * model if data is not None else None
        
        _, model = self.eval(params)
        model = model[s or ()]

        # TODO: pass these to the spectrum class instead, here we
        #       don't know what vectors to normalize
        spec.flux = denormalize_vector(spec.flux, model)
        spec.flux_err = denormalize_vector(spec.flux_err, model)
        spec.cont = denormalize_vector(spec.cont, model)

        spec.append_history(f'Spectrum is denormalized using model `{type(self).__name__}`.')