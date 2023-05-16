import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.optimize import minimize_scalar
import numdifftools as nd

from pfs.ga.pfsspec.core import Physics
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
            self.template_grids = None       # Model grids for each spectrograph arm
            
            self.params_0 = None             # Dict of initial template parameters
            self.params_fixed = None         # List of names of template params not fitted
            self.params_bounds = None        # Template parameter bounds
            self.params_priors = None        # Dict of callables for each parameter
        else:
            self.template_grids = orig.template_grids

            self.params_0 = orig.params_0
            self.params_fixed = orig.params_fixed
            self.params_bounds = orig.params_bounds
            self.params_priors = orig.params_priors

    def get_templates(self, spectra, params):
        # Return the templates corresponding to the parameters.
        # Different grids are used for the different arms

        templates = {}
        missing = False
        for k in spectra:
            grid = self.template_grids[k]

            if self.template_psf is not None:
                psf = self.template_psf[k]
            else:
                psf = None 

            # TODO: allow higher order interpolation
            if self.template_wlim is not None:
                wlim = self.template_wlim[k]
            else:
                wlim = None
            temp = grid.interpolate_model(psf=psf, wlim=wlim, **params)

            # Interpolation failed
            if temp is None:
                missing = True
                break

            templates[k] = temp

        return templates, missing

    def get_normalization(self, spectra, templates=None, params_0=None, params_fixed=None, **kwargs):
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
        
    def process_template_impl(self, arm, template, spectrum, rv, psf=None, wlim=None):
        # Skip convolution because convolution is pushed down to the
        # model interpolator to support caching            
        return super().process_template_impl(arm, template, spectrum, rv, psf=None, wlim=wlim)
        
    def calculate_log_L(self, spectra, templates, rv, rv_prior=None, params=None, params_priors=None, a=None):
        # Calculate log_L using a provided set of templates or templates
        # at the provided parameters

        rv_prior = rv_prior if rv_prior is not None else self.rv_prior
        params = params if params is not None else self.params_0
        params_priors = params_priors if params_priors is not None else self.params_priors

        if templates is None:
            templates, missing = self.get_templates(spectra, params)
        else:
            missing = False

        if missing:
            raise Exception("Template parameters are outside the grid.")
        else:
            log_L, phi, chi, ndf = super().calculate_log_L(spectra, templates, rv, rv_prior=rv_prior, a=a)
            if params_priors is not None:
                for p in params_priors.keys():
                    if p in params and params_priors[p] is not None:
                        log_L += params_priors[p](params[p])
            return log_L, phi, chi, ndf
        
    #region Fisher matrix evaluation

    def determine_grid_bounds(self, params_bounds, params_free):
        # Param bounds are taken from grid limits, which is not the best solution
        # with grids having jagged edges

        # Silently assume that grids have the same axis for each arm
        grid = self.template_grids[list(self.template_grids.keys())[0]]

        bounds = {}
        for p in params_free:
            # Use grid bounds if params_bounds is None
            if params_bounds is None or p not in params_bounds or params_bounds[p] is None:
                # Use axis limits from the grid
                bounds[p] = (grid.get_axis(p).values.min(), grid.get_axis(p).values.max())
            else:
                bounds[p] = params_bounds[p]

        return bounds
    
    def determine_free_params(self, params_fixed):
        # Enumerate grid parameters in order and filter out values that should be kept fixed
        # Order is important because arguments are passed by position to the cost function

        for k, grid in self.template_grids.items():
            params_free = [p for i, p, ax in grid.enumerate_axes() if params_fixed is None or p not in params_fixed]
            return params_free  # return from loop after very first iter!
        
    def get_packing_functions(self, params_0, params_free, params_fixed=None, mode='full'):
        # Return the parameter packing/unpacking functions for optimizers
        # pack_params: convert individual arguments into a single 1d array
        # unpack_params: get individual arguments from 1d array
        # pack_bounds: pack parameters bounds into a list of tuples

        if mode == 'full' or mode == 'a_params_rv':
            def pack_params(a, params, rv):
                # Leading dim is number of parameters for each component, rest is batch shape
                rv = np.atleast_1d(rv)
                if rv.size > 1:
                    rv = np.reshape(rv, (-1,) + rv.shape)
                params = np.array([ params[k] for k in params_free ])
                a = np.reshape(a, (-1,) + rv.shape[1:])

                return np.concatenate([a, params, rv])
        
            def unpack_params(a_params_rv):
                # Leading dim is number of parameters for each component, rest is batch shape
                params = {}
                for i, p in enumerate(params_free):
                    params[p] = a_params_rv[-(len(params_free) + 1) + i]
                for p in params_fixed:
                    params[p] = params_fixed[p]

                a = a_params_rv[:-(len(params_free) + 1)]
                if a.ndim == 2:
                    a = np.squeeze(a)
                elif a.size == 1:
                    a = a.item()

                rv = a_params_rv[-1]
                if rv.size == 0:
                    rv = rv.item()

                return a, params, rv

            def pack_bounds(a_bounds, params_bounds, rv_bounds):
                if a_bounds is None:
                    raise NotImplementedError()
                else:
                    bounds = a_bounds

                if params_bounds is None:
                    bounds += [ params_bounds[k] if k in params_bounds else None for k in params_free ]
                else:
                    bounds += [ None for k in params_free ]
                    
                bounds += [ rv_bounds ]
                
                return bounds
        elif mode == 'params_rv':
            def pack_params(params, rv):
                # Leading dim is number of parameters, rest is batch shape
                rv = np.atleast_1d(rv)
                if rv.size > 1:
                    rv = np.reshape(rv, (-1,) + rv.shape)
                params = np.array([ params[k] for k in params_free ])

                return np.concatenate([params, rv])
            
            def unpack_params(params_rv):
                rv = params_rv[-1]
                if rv.size == 1:
                    rv = rv.item()

                params = {}
                for i, p in enumerate(params_free):
                    params[p] = params_rv[i]
                for p in params_fixed:
                    params[p] = params_fixed[p]

                return params, rv
            
            def pack_bounds(params_bounds, rv_bounds):
                if params_bounds is not None:
                    bounds = [ params_bounds[k] if k in params_bounds else None for k in params_free ]
                else:
                    bounds = [ None for k in params_free ]
                bounds += [ rv_bounds ] 
                
                return bounds
        elif mode == 'rv':
            def pack_params(rv):
                return np.atleast_1d(rv)
            
            def unpack_params(rv):
                rv = rv
                if rv.size == 1:
                    rv = rv.item()

                return params_0, rv
            
            def pack_bounds(rv_bounds):
                return [ rv_bounds ]
            
        else:
            raise NotImplementedError()  
        
        return pack_params, unpack_params, pack_bounds
        
    def get_objective_function(self, spectra, rv_prior, params_0, params_priors, params_free, params_fixed=None, mode='full'):
        # Return the objection functionfor optimizers
        # log_L: evaluate the log likelihood

        pack_params, unpack_params, pack_bounds = self.get_packing_functions(params_0, params_free, params_fixed=params_fixed, mode=mode)
        
        if mode == 'full' or mode == 'a_params_rv':            
            def log_L(a_params_rv):
                a, params, rv = unpack_params(a_params_rv)
                templates, missing = self.get_templates(spectra, params)

                if missing:
                    # Trying to extrapolate outside the grid
                    return -np.inf
                else:
                    log_L, phi, chi, ndf = self.calculate_log_L(spectra, templates, rv, rv_prior=rv_prior, params_priors=params_priors, a=a)
                    if log_L.ndim == 0:
                        log_L = log_L.item()

                    return log_L
        elif mode == 'params_rv':
            def log_L(params_rv):
                params, rv = unpack_params(params_rv)
                templates, missing = self.get_templates(spectra, params)

                if missing:
                    # Trying to extrapolate outside the grid
                    return -np.inf
                else:
                    log_L, _, _, _ = self.calculate_log_L(spectra, templates, rv, rv_prior=rv_prior, params_priors=params_priors)
                    if log_L.ndim == 0:
                        log_L = log_L.item()

                    return log_L
        elif mode == 'rv':
            def log_L(rv):
                params_0, rv = unpack_params(rv)
                templates, missing = self.get_templates(spectra, params_0)

                if missing:
                    # Trying to extrapolate outside the grid
                    return -np.inf
                else:
                    log_L, _, _, _ = self.calculate_log_L(spectra, templates, rv, rv_prior=rv_prior, params_priors=params_priors)
                    if log_L.ndim == 0:
                        log_L = log_L.item()

                    return log_L
        else:
            raise NotImplementedError()  
        
        return log_L, pack_params, unpack_params, pack_bounds

    def eval_F(self, spectra, rv_0, params_0, rv_bounds=None, rv_prior=None, params_bounds=None, params_priors=None, params_fixed=None, step=None, mode='full', method='hessian'):
        # Evaluate the Fisher matrix around the provided rv_0 and params_0
        # values. The corresponding a_0 best fit flux correction will be
        # evaluated at the optimum. The Hessian will be calculated wrt either
        # RV only, or rv and the template parameters or rv, the template parameters
        # and the flux correction coefficients. Alternatively, the covariance
        # matrix will be determined using MCMC.

        # Collect fixed and free template parameters
        rv_prior = rv_prior if rv_prior is not None else self.rv_prior
        rv_bounds = rv_bounds if rv_bounds is not None else self.rv_bounds

        params_priors = params_priors if params_priors is not None else self.params_priors
        
        params_fixed = params_fixed if params_fixed is not None else self.params_fixed
        params_free = self.determine_free_params(params_fixed)
        
        params_bounds = params_bounds if params_bounds is not None else self.params_bounds
        params_bounds = self.determine_grid_bounds(self.params_bounds, params_free)
        
        # Get objective function
        log_L, pack_params, unpack_params, pack_bounds = self.get_objective_function(
            spectra, rv_prior, params_0, params_priors, params_free, params_fixed=params_fixed, mode=mode)

        if mode == 'full' or mode == 'a_params_rv':               
            # Calculate a_0
            templates, missing = self.get_templates(spectra, params_0)
            phi_0, chi_0, ndf_0 = self.eval_phi_chi(spectra, templates, rv_0)
            a_0 = self.eval_a(phi_0, chi_0)
            x_0 = pack_params(a_0, params_0, rv_0)
            bounds = pack_bounds(a_0.size * [(-np.inf, np.inf)], params_bounds, rv_bounds)
        elif mode == 'params_rv':                
            x_0 = pack_params(params_0, rv_0)
            bounds = pack_bounds(params_bounds, rv_bounds)
        elif mode == 'rv':                
            x_0 = pack_params(rv_0)
            bounds = pack_bounds(rv_bounds)
        else:
            raise NotImplementedError()

        return self.eval_F_dispatch(x_0, log_L, step, method, bounds)

    def calculate_F(self, spectra, rv_0, params_0, rv_bounds=None, rv_prior=None, params_bounds=None, params_priors=None, params_fixed=None, step=None, mode='full', method='hessian'):
        # Calculate the Fisher matrix using different methods

        return self.eval_F(spectra, rv_0, params_0, rv_bounds=rv_bounds, rv_prior=rv_prior, params_bounds=params_bounds, params_priors=params_priors, params_fixed=params_fixed, step=step, mode=mode, method=method)

    #endregion

    def guess_rv(self, spectra, templates=None, /, rv_bounds=(-500, 500), rv_prior=None, params_0=None, params_fixed=None, rv_steps=31, method='lorentz'):
        """
        Guess an initial state to close best RV, either using the supplied set of templates or
        initial model parameters.
        """

        # TODO: Maybe extend this to do a grid search in model parameters

        params_0 = params_0 if params_0 is not None else self.params_0
        params_fixed = params_fixed if params_fixed is not None else self.params_fixed

        if templates is None:
            # Look up the templates from the grid and pass those to the the parent class
            params = params_0.copy()
            if params_fixed is not None:
                params.update(params_fixed)
            templates, missing = self.get_templates(spectra, params)
            if missing:
                raise Exception('Cannot find an initial matching template to start RVFit.')

        return super().guess_rv(spectra, templates, rv_bounds=rv_bounds, rv_prior=rv_prior, rv_steps=rv_steps, method=method)

    def fit_rv(self, spectra, /,
               rv_0=None, rv_bounds=(-500, 500), rv_prior=None,
               params_0=None, params_bounds=None, params_priors=None, params_fixed=None,
               method="Nelder-Mead", max_iter=None):
        """
        :param params_0: Dictionary of initial values.
        :param params_bounds: Dictionary of tuples, parameters bounds.
        :param params_fixed: Dictionary of fixed parameter values.
        """

        rv_0 = rv_0 if rv_0 is not None else self.rv_0
        rv_bounds = rv_bounds if rv_bounds is not None else self.rv_bounds
        rv_prior = rv_prior if rv_prior is not None else self.rv_prior

        params_0 = params_0 if params_0 is not None else self.params_0
        params_fixed = params_fixed if params_fixed is not None else self.params_fixed
        params_free = self.determine_free_params(params_fixed)
        params_bounds = params_bounds if params_bounds is not None else self.params_bounds
        params_bounds = self.determine_grid_bounds(params_bounds, params_free)
        params_priors = params_priors if params_priors is not None else self.params_priors

        max_iter = max_iter if max_iter is not None else self.max_iter

        if params_0 is None:
            # TODO
            raise NotImplementedError()
        
        if rv_0 is None:
            _, _, _, rv_0 = self.guess_rv(spectra, None, 
                                          rv_bounds=rv_bounds, rv_prior=rv_prior, 
                                          params_0=params_0, params_fixed=params_fixed)
        
        if self.template_wlim is None:
            self.determine_wlim(spectra, rv_bounds=rv_bounds)

        # Get objective function
        log_L, pack_params, unpack_params, pack_bounds = self.get_objective_function(
            spectra, rv_prior, params_0, params_priors, params_free, params_fixed=params_fixed, mode='params_rv')

        # Initial values and bounds for optimization
        x0 = pack_params(params_0, rv_0)
        bounds = pack_bounds(params_bounds, rv_bounds)

        # Cost function
        def llh(params_rv):
            return -log_L(params_rv)
        
        out = minimize(llh, x0=x0, bounds=bounds, method=method,
                       options=dict(maxiter=self.max_iter))

        if out.success:
            params_fit, rv_fit = unpack_params(out.x)
        else:
            raise Exception(f"Could not fit RV using `{method}`, reason: {out.message}")
        
        # Calculate the flux correction coefficients at best fit values
        templates, missing = self.get_templates(spectra, params_fit)
        phi_fit, chi_fit, ndf_fit = self.eval_phi_chi(spectra, templates, rv_fit)
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

    def run_mcmc(self, spectra, /,
               rv_0=None, rv_bounds=(-500, 500), rv_prior=None,
               params_0=None, params_bounds=None, params_priors=None, params_fixed=None):
        """
        Given a set of spectra and templates, sample from the posterior distribution of RV.

        If no initial guess is provided, an initial state is generated automatically.
        """

        rv_0 = rv_0 if rv_0 is not None else self.rv_0
        rv_bounds = rv_bounds if rv_bounds is not None else self.rv_bounds
        rv_prior = rv_prior if rv_prior is not None else self.rv_prior

        params_0 = params_0 if params_0 is not None else self.params_0
        params_fixed = params_fixed if params_fixed is not None else self.params_fixed
        params_free = self.determine_free_params(params_fixed)
        params_bounds = params_bounds if params_bounds is not None else self.params_bounds
        params_bounds = self.determine_grid_bounds(params_bounds, params_free)
        params_priors = params_priors if params_priors is not None else self.params_priors

        if params_0 is None:
            # TODO
            raise NotImplementedError()
        
        if rv_0 is None:
            _, _, _, rv_0 = self.guess_rv(spectra, None, 
                                          rv_bounds=rv_bounds, rv_prior=rv_prior, 
                                          params_0=params_0, params_fixed=params_fixed)
            
        if self.template_wlim is None:
            self.determine_wlim(spectra, rv_bounds=rv_bounds)

        # Get objective function
        log_L_fun, pack_params, unpack_params, pack_bounds = self.get_objective_function(
            spectra, rv_prior, params_0, params_priors, params_free, params_fixed=params_fixed, mode='params_rv')
        
        # Initial values and bounds for optimization
        x_0 = pack_params(params_0, rv_0)
        bounds = pack_bounds(params_bounds, rv_bounds)

        # TODO: merge with fit_rv
        # Everything is the same as fit_rv up until this points
        
        # Run sampling
        x, log_L = self.sample_log_L(log_L_fun, x_0=x_0, bounds=bounds)
        params, rv = unpack_params(x.T)

        # TODO: we could calculate the flux correction here but is it worth it?

        return rv, params, None