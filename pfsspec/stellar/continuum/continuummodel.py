import numpy as np

from pfsspec.core import PfsObject

class ContinuumModel(PfsObject):

    PREFIX_CONTINUUM_MODEL = 'continumm_model'

    def __init__(self, orig=None):
        super().__init__()

        if isinstance(orig, ContinuumModel):
            self.wave = orig.wave
            self.wave_mask = orig.wave_mask
        else:
            self.wave = None
            self.wave_mask = None

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

    def get_model_parameters(self):
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
        for p in self.get_model_parameters():
            grid.init_value(p.name)

    def allocate_values(self, grid):
        # Allocate the values in a grid necessary to store the fitted parameters
        raise NotImplementedError()

    def fit(self, spec):
        raise NotImplementedError()

    def eval(self, params):
        raise NotImplementedError()

    def normalize(self, spec, params):
        raise NotImplementedError()

    def denormalize(self, spec, params):
        raise NotImplementedError()

    def fill_params(self, name, params):
        raise NotImplementedError()

    def smooth_params(self, name, params):
        raise NotImplementedError()

    def fit_model_simple(self, model, x, y, w=None, p0=None, max_deg=None):
        # Simple chi2 fit to x and y with optional weights
        params = model.fit(x, y, w=w, p0=p0, max_deg=max_deg)
        return params

    def fit_model_sigmaclip(self, model, x, y, w=None, p0=None, sigma_low=1, sigma_high=1):
        w = w if w is not None else np.full(x.shape, 1.0)
        m = np.full(x.shape, True)
        p = p0
        for i in range(5):
            p = model.fit(x[m], y[m], w=w[m], p0=p)
            f = model.eval(x, p)
            
            # Sigma clipping
            s = np.std(f - y)
            m = (y < f + sigma_high * s) & (f - sigma_low * s < y)
        return p

    def eval_model(self, model, x, params):
        return model.eval(x, params)