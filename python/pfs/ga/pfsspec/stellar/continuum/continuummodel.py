import numpy as np

from pfs.ga.pfsspec.core import PfsObject

class ContinuumModelTrace():
    def __init__(self):
        pass

    def on_fit_function_iter(self, piece_id, iter, x, y, w, model, mask):
        pass

class ContinuumModel(PfsObject):

    PREFIX_CONTINUUM_MODEL = 'continumm_model'

    def __init__(self, continuum_finder=None, trace=None, orig=None):
        super().__init__()

        if not isinstance(orig, ContinuumModel):
            self.trace = trace
            self.continuum_finder = continuum_finder

            self.wave = None
            self.wave_mask = None
        else:
            self.trace = trace or orig.trace
            self.continuum_finder = continuum_finder or orig.continuum_finder

            self.wave = orig.wave
            self.wave_mask = orig.wave_mask

    @property
    def name(self):
        raise NotImplementedError()

    def add_args(self, parser):
        pass

    def init_from_args(self, parser):
        pass

    def save_items(self):
        pass

    def load_items(self):
        pass

    def get_interpolated_params(self):
        return []

    def get_constants(self, wave):
        return {}

    def set_constants(self, wave, constants):
        pass

    def set_constant(self, name, constants, default):
        if name in constants and constants[name] is not None:
            return constants[name]
        else:
            return default

    def init_wave(self, wave):
        # Initialize the wave vector cache and masks, if necessary
        raise NotImplementedError()

    def init_constants(self, grid):
        # Initialize the constants in a grid necessary to store the fitted parameters
        pass

    def init_values(self, grid):
        # Initialize the values in a grid necessary to store the fitted parameters
        # These are the parameters that we store for each spectrum and not the grid-wide
        # constants that are the same for each spectrum.
        for p in self.get_interpolated_params():
            grid.init_value(p.name)

    def allocate_values(self, grid):
        # Allocate the values in a grid necessary to store the fitted parameters
        raise NotImplementedError()

    def fill_params(self, name, params):
        # Fill in the hole of the parameter grid
        raise NotImplementedError()

    def smooth_params(self, name, params):
        # Smooth the parameter grid
        raise NotImplementedError()

    def fit(self, spec):
        raise NotImplementedError()

    def eval(self, params):
        raise NotImplementedError()

    def normalize(self, spec, params):
        raise NotImplementedError()

    def denormalize(self, spec, params, s=None):
        raise NotImplementedError()

    def fill_params(self, name, params):
        raise NotImplementedError()

    def smooth_params(self, name, params):
        raise NotImplementedError()

    def fit_function(self, id, func, x, y, w=None, p0=None, mask=None, continuum_finder=None, **kwargs):
        """
        Fit a function to a set of points. If a continuum finder algorithm is set,
        use that to identify points to fit to.
        """

        param_count = func.get_param_count()
        min_point_count = func.get_min_point_count()

        continuum_finder = continuum_finder if continuum_finder is not None else self.continuum_finder

        if mask is None:
            mask = np.full_like(x, True, dtype=bool)
        else:
            mask = mask.copy()
            
        iter = 0
        params = None
        success = False
        while True:
            if iter == 0 and p0 is not None and self.continuum_finder is not None:
                # Use initial parameter estimates in the first iteration
                params = p0
            else:
                # Filter points based on the previous iteration of fitting
                # and run fitting again
                params = func.fit(x[mask], y[mask], 
                    w=w[mask] if w is not None else None,
                    p0=params, **kwargs)    
            
            if self.trace:
                self.trace.on_fit_function_iter(id, iter, x, y, w, func.eval(x, params), mask)
            
            # We're always successful if it's a linear model and func.fit has been
            # executed at least once.
            # TODO: handle non-linear fitting with unsuccessful convergence here
            #       this requires returning a success flag from func.fit
            success = True  

            # If we fit to all points withing the mask and there's no continuum finder
            # to be executed iteratively, we're done, otherwise just do the iterations
            if continuum_finder is None:
                break

            # Evaluate the model at the current parameters and run the continuum finder to
            # filter out points that are not part of the continuum
            model = func.eval(x, params)
            mask, need_more_iter = continuum_finder.find(iter, x, y, w=w, mask=mask, model=model)
            iter += 1

            if not need_more_iter:
                break

            # Only continue fitting if we have enough data points
            if mask.sum() < min_point_count:
                break

        if success:
            return True, params
        else:
            return False, np.full(param_count, np.nan)

    def eval_model(self, model, x, params):
        return model.eval(x, params)

#region Utility functions

    def safe_log(self, x):
        return np.log(np.where(x > 1e-7, x, np.nan))

    def safe_log_error(self, x, sigma):
        return np.where(x > 1e-7, sigma / x, np.nan)

    def safe_exp(self, x):
        return np.exp(np.where(x < 100, x, np.nan))

    def safe_exp_error(self, x, sigma):
        return np.where(x < 100, x * sigma, np.nan)

#endregion