import numpy as np

class CorrectionModel():
    """
    Base class for flux correction or continuum fitting to be
    applied to observed spectra before template fitting.
    """

    def __init__(self, orig=None):
        if not isinstance(orig, CorrectionModel):
            self.trace = None
            self.tempfit = None
        else:
            self.trace = orig.trace
            self.tempfit = orig.tempfit

    def reset(self):
        pass

    def add_args(self, config, parser):
        pass

    def init_from_args(self, script, config, args):
        pass

    def init_models(self, spectra, rv_bounds=None, force=False):
        raise NotImplementedError()
        
    def calculate_coeffs(self, spectra, templates, a=None):
        raise NotImplementedError()

    def get_objective_function(self, spectra, templates, rv_prior, mode='full'):
        raise NotImplementedError()

    def get_param_packing_functions(self, mode='full'):
        """
        Return functions to pack and unpack parameters for optimization.

        Pack amplitudes and/or flux correction or continuum fit parameters.
        """

        if mode == 'full' or 'a' in mode.split('_'):
            def pack_params(a):
                return np.atleast_1d(a)

            def unpack_params(a):
                if a.ndim == 2:
                    a = np.squeeze(a)
                elif a.size == 1:
                    a = a.item()

                return a

            def pack_bounds(a_bounds):
                if a_bounds is None:
                    raise NotImplementedError()
                else:
                    bounds = a_bounds

                return bounds
        else:
            pack_params, unpack_params, pack_bounds = None, None, None
            
        return pack_params, unpack_params, pack_bounds
    
