import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.optimize import minimize_scalar

from .rvfit import RVFit, RVFitTrace

class ModelGridRVFitTrace(RVFitTrace):
    pass

class ModelGridRVFit(RVFit):
    """
    Performs radial velocity fitting by interpolating templates from
    a model spectrum grid. It can either maximize the likelihood function
    or the significance function of Kaiser (2004).
    """

    def __init__(self, trace=None, orig=None):
        super().__init__(trace=trace, orig=orig)

        if not isinstance(orig, ModelGridRVFit):
            self.template_grids = None
            
            self.params_0 = None             # Initial template parameters
            self.params_bounds = None        # Template parameter bounds
        else:
            self.template_grids = orig.template_grids

            self.params_0 = orig.params_0
            self.params_bounds = orig.params_bounds
            
    def fit_rv(self, spectra, rv_0=None, rv_bounds=(-500, 500), params_0=None, params_bounds=None, params_fixed=None, method="Nelder-Mead"):
        """
        
        
        :param params_0: Dictionary of initial values.
        :param params_bounds: Dictionary of tuples, parameters bounds.
        :param params_fixed: Dictionary of fixed parameter values.
        """

        rv_0 = rv_0 if rv_0 is not None else self.rv_0
        rv_bounds = rv_bounds if rv_bounds is not None else self.rv_bounds

        params_0 = params_0 if params_0 is not None else self.params_0
        params_bounds = params_bounds if params_bounds is not None else self.params_bounds
        params_fixed = params_fixed if params_fixed is not None else {}

        if rv_0 is None:
            # TODO
            raise NotImplementedError()

        if params_0 is None:
            # TODO
            raise NotImplementedError()

        # Enumerate grid parameters in order and filter out values that should be kept fixed
        # Order is important because arguments are passed by position to the cost function
        for k in spectra:
            grid = self.template_grids[k]
            params_free = [p for i, p, ax in grid.enumerate_axes() if p not in params_fixed]
            break

        # Cost function
        def llh(x):
            # Gather the various parameters
            rv = x[0]
            params = {}
            for i, p in enumerate(params_free):
                params[p] = x[1 + i]
            for p in params_fixed:
                params[p] = params_fixed[p]

            templates = {}
            missing = False
            for k in spectra:
                # TODO: allow higher order interpolation
                grid = self.template_grids[k]
                temp = grid.interpolate_model_linear(**params)

                # Interpolation failed
                if temp is None:
                    missing = True
                    break

                templates[k] = temp

            if missing:
                return np.inf
            else:
                log_L, phi, chi = self.calculate_log_L(spectra, templates, rv)
                return -log_L

        # Initial values and bounds for optimization
        x0 = [ rv_0 ] + [ params_0[k] for k in params_free ]
        bounds = [ rv_bounds ] + [ params_bounds[k] if k in params_bounds else None for k in params_free ]

        out = minimize(llh, x0=x0, bounds=bounds, method=method)

        if out.success:
            # TODO: Error from fisher matrix
            # TODO: Continuum params and errors

            rv = out.x[0]
            params = {}
            for i, p in enumerate(params_free):
                params[p] = out.x[1 + i]
            for p in params_fixed:
                params[p] = params_fixed[p]
            return rv, None, params, None
        else:
            raise Exception(f"Could not fit RV using `{method}`")

        