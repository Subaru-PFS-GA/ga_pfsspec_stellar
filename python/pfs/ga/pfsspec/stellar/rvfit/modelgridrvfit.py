import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.optimize import minimize_scalar
import numdifftools as nd
import emcee

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
            
            self.params_0 = None             # Dict of initial template parameters
            self.params_fixed = None         # List of names of template params not fitted
            self.params_bounds = None        # Template parameter bounds
        else:
            self.template_grids = orig.template_grids

            self.params_0 = orig.params_0
            self.params_bounds = orig.params_bounds

    def get_templates(self, spectra, params):
        # Return the templates corresponding to the parameters.
        # Different grids are used for the different arms

        templates = {}
        missing = False
        for k in spectra:
            # TODO: allow higher order interpolation
            grid = self.template_grids[k]
            if self.template_psf is not None:
                psf = self.template_psf[k]
            else:
                psf = None 
            temp = grid.interpolate_model(psf=psf, **params)

            # Interpolation failed
            if temp is None:
                missing = True
                break

            templates[k] = temp

        return templates, missing

    def get_normalization(self, spectra, templates=None, params_0=None, **kwargs):
        # Calculate a normalization factor for the spectra, as well as
        # the templates assuming an RV=0 and params_0 for the template.
        # This is just a factor to bring spectra to the same scale and avoid
        # very large numbers in the Fisher matrix

        if params_0 is not None:
            params = params_0
        elif kwargs is not None and len(kwargs) > 0:
            params = kwargs
        elif self.params_0 is not None:
            params = self.params_0

        if templates is not None:
            missing = False
        else:
            templates, missing = self.get_templates(spectra, params)
            
        if missing:
            raise Exception("Template parameters are outside the grid.")
        else:
            return super().get_normalization(spectra, templates)
        
    def process_template_impl(self, template, spectrum, rv, psf=None):
        # 1. Make a copy, not in-place update
        t = template.copy()

        # 2. Shift template to the desired RV
        t.set_rv(rv)

        # 3. Skip convolution because convolution is pushed down to the
        #    model interpolator to support caching

        # 4. Normalize
        if self.temp_norm is not None:
            t.multiply(1.0 / self.temp_norm)

        if self.trace is not None:
            self.trace.on_process_template(rv, template, t)
            
        return t
        
    def calculate_log_L(self, spectra, templates, rv, params=None, a=None):
        # Calculate log_L using a provided set of templates or templates
        # at the provided parameters

        params = params if params is not None else self.params_0
        if templates is None:
            templates, missing = self.get_templates(spectra, params)
        else:
            missing = False

        if missing:
            raise Exception("Template parameters are outside the grid.")
        else:
            return super().calculate_log_L(spectra, templates, rv, a=a)
        
    #region Fisher matrix evaluation

    def determine_grid_bounds(self, params_bounds, params_free):
        if params_bounds is not None:
            params_bounds = (
                { p: params_bounds[p][0] for p in params_free },
                { p: params_bounds[p][1] for p in params_free },
            )
        else:
            # Param bounds are taken from grid limits, which is not the best solution
            # with grids having jagged edges
            grid = self.template_grids[list(self.template_grids.keys())[0]]

            params_bounds = (
                { p: grid.get_axis(p).values.min() for p in params_free },
                { p: grid.get_axis(p).values.max() for p in params_free }
            )

        return params_bounds
    
    def determine_free_params(self, params_fixed):
        # Enumerate grid parameters in order and filter out values that should be kept fixed
        # Order is important because arguments are passed by position to the cost function

        for k, grid in self.template_grids.items():
            params_free = [p for i, p, ax in grid.enumerate_axes() if p not in params_fixed]
            return params_free  # return from loop after very first iter!

    def eval_F(self, spectra, rv_0, params_0, params_fixed=None, step=None, mode='full', method='hessian'):
        # Evaluate the Fisher matrix around the provided rv_0 and params_0
        # values. The corresponding a_0 best fit flux correction will be
        # evaluated at the optimum. The Hessian will be calculated wrt either
        # RV only, or rv and the template parameters or rv, the template parameters
        # and the flux correction coefficients. Alternatively, the covariance
        # matrix will be determined using MCMC.

        # Collect fixed template parameters
        params_fixed = params_fixed if params_fixed is not None else self.params_fixed
        params_free = self.determine_free_params(params_fixed)

        params_bounds = self.determine_grid_bounds(self.params_bounds, params_free)

        if self.rv_bounds is not None:
            rv_bounds = self.rv_bounds
        else:
            rv_bounds = np.array([-np.inf, np.inf])

        if self.params_bounds and self.rv_bounds:
            bounds = (
                pack_params(np.full_like(a_0, -np.inf), self.params_bounds[0], self.rv_bounds[0])
            )
            bounds = (
                np.concatenate()
            )

        if mode == 'full' or mode == 'a_params_rv':
            def pack_params(a, params, rv):
                return np.concatenate([
                    np.atleast_1d(a),
                    np.array([ params[k] for k in params_free ]),
                    np.atleast_1d(rv)])
        
            def unpack_params(a_params_rv):
                rv = np.asscalar(a_params_rv[-1])
                a = a_params_rv[:-len(params_free) - 1]

                params = {}
                for i, p in enumerate(params_free):
                    params[p] = a_params_rv[a.size + i]
                for p in params_fixed:
                    params[p] = params_fixed[p]

                return a, params, rv
            
            def log_L(a_params_rv):
                a, params, rv = unpack_params(a_params_rv)
                templates, missing = self.get_templates(spectra, params)

                if missing:
                    # Trying to extrapolate outside the grid
                    return -np.inf
                else:
                    log_L, phi, chi = self.calculate_log_L(spectra, templates, rv, a=a)
                    return np.asscalar(log_L)
                
            # Calculate a_0
            templates, missing = self.get_templates(spectra, params_0)
            phi_0, chi_0 = self.eval_phi_chi(spectra, templates, rv_0)
            a_0 = self.eval_a(phi_0, chi_0)
            x_0 = pack_params(a_0, params_0, rv_0)

            bounds = (
                pack_params(np.full_like(a_0, -np.inf), params_bounds[0], rv_bounds[0]),
                pack_params(np.full_like(a_0, np.inf), params_bounds[1], rv_bounds[1])
            )
        elif mode == 'params_rv':
            def pack_params(params, rv):
                return np.concatenate([
                    np.array([ params[k] for k in params_free ]),
                    np.atleast_1d(rv)])
            
            def unpack_params(params_rv):
                rv = np.asscalar(params_rv[-1])

                params = {}
                for i, p in enumerate(params_free):
                    params[p] = params_rv[i]
                for p in params_fixed:
                    params[p] = params_fixed[p]

                return params, rv
            
            def log_L(params_rv):
                params, rv = unpack_params(params_rv)
                templates, missing = self.get_templates(spectra, params)

                if missing:
                    # Trying to extrapolate outside the grid
                    return -np.inf
                else:
                    log_L, _, _ = self.calculate_log_L(spectra, templates, rv)
                    return np.asscalar(log_L)
                
            x_0 = pack_params(params_0, rv_0)

            bounds = (
                pack_params(params_bounds[0], rv_bounds[0]),
                pack_params(params_bounds[1], rv_bounds[1])
            )
        elif mode == 'rv':
            def pack_params(rv):
                return np.atleast_1d(rv)
            
            def unpack_params(params):
                rv = params[0]
                return rv
            
            def log_L(params):
                rv = unpack_params(params)
                templates, missing = self.get_templates(spectra, params_0)

                if missing:
                    # Trying to extrapolate outside the grid
                    return -np.inf
                else:
                    log_L, _, _ = self.calculate_log_L(spectra, templates, rv)
                    return np.asscalar(log_L)
                
            x_0 = np.atleast_1d(rv_0)
            bounds = rv_bounds
        else:
            raise NotImplementedError()

        return self.eval_F_dispatch(x_0, log_L, step, method, bounds)
    
    def calculate_F(self, spectra, rv_0, params_0, params_fixed=None, step=None, mode='full', method='hessian'):
        # Calculate the Fisher matrix using different methods

        return self.eval_F(spectra, rv_0, params_0, params_fixed=params_fixed, step=step, mode=mode, method=method)

    #endregion
            
    def fit_rv(self, spectra, rv_0=None, rv_bounds=(-500, 500), params_0=None, params_bounds=None, params_fixed=None, method="Nelder-Mead"):
        """
        :param params_0: Dictionary of initial values.
        :param params_bounds: Dictionary of tuples, parameters bounds.
        :param params_fixed: Dictionary of fixed parameter values.
        """

        rv_0 = rv_0 if rv_0 is not None else self.rv_0
        rv_bounds = rv_bounds if rv_bounds is not None else self.rv_bounds

        params_0 = params_0 if params_0 is not None else self.params_0
        params_fixed = params_fixed if params_fixed is not None else self.params_fixed
        params_free = self.determine_free_params(params_fixed)
        params_bounds = params_bounds if params_bounds is not None else self.params_bounds
        params_bounds = self.determine_grid_bounds(params_bounds, params_free)

        if rv_0 is None:
            # TODO
            raise NotImplementedError()

        if params_0 is None:
            # TODO
            raise NotImplementedError()

        def pack_params(rv, **params):
            x = np.array([ rv ] + [ params[k] for k in params_free ])
            return x
        
        def pack_bounds(rv_bounds, params_bounds):
            bounds = [ rv_bounds ] 
            if params_bounds is not None:
                bounds += [ (params_bounds[0][k], params_bounds[1][k]) if k in params_bounds[0] else None for k in params_free ]
            else:
                bounds += [ None for k in params_free ]
            
            return bounds

        def unpack_params(x):
            rv = x[0]
            params = {}
            for i, p in enumerate(params_free):
                params[p] = x[1 + i]
            for p in params_fixed:
                params[p] = params_fixed[p]

            return rv, params

        # Cost function
        def llh(x):
            # Gather the various parameters
            rv, params = unpack_params(x)
            templates, missing = self.get_templates(spectra, params)

            if missing:
                return np.inf
            else:
                log_L, phi, chi = self.calculate_log_L(spectra, templates, rv)
                return -log_L

        # Initial values and bounds for optimization
        x0 = pack_params(rv_0, **params_0)
        bounds = pack_bounds(rv_bounds, params_bounds)
        
        out = minimize(llh, x0=x0, bounds=bounds, method=method)

        if out.success:
            rv_fit, params_fit = unpack_params(out.x)
        else:
            raise Exception(f"Could not fit RV using `{method}`")
        
        # Calculate the flux correction coefficients at best fit values
        templates, missing = self.get_templates(spectra, params_fit)
        phi_fit, chi_fit = self.eval_phi_chi(spectra, templates, rv_fit)
        a_fit = self.eval_a(phi_fit, chi_fit)
        
        # Calculate the error from the Fisher matrix
        _, C = self.eval_F(spectra, rv_fit, params_fit, params_fixed=params_fixed, mode='params_rv', method='hessian')
        err = np.sqrt(np.diag(C)) # sigma
        params_err = {}
        for i, p in enumerate(params_free):
            params_err[p] = err[i]
        for i, p in enumerate(params_fixed):
            params_err[p] = 0.0
        rv_err = err[-1]

        return rv_fit, rv_err, params_fit, params_err, a_fit, None