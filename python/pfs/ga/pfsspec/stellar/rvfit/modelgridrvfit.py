import logging

import numpy as np
from numpy.linalg import LinAlgError
from scipy.optimize import curve_fit, minimize
from scipy.optimize import minimize_scalar
import numdifftools as nd

from pfs.ga.pfsspec.core.sampling import MCMC
from pfs.ga.pfsspec.core import Physics
from pfs.ga.pfsspec.core.sampling import Parameter, Distribution
from .rvfit import RVFit
from .rvfitresults import RVFitResults
from .modelgridrvfittrace import ModelGridRVFitTrace

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
            self.params_steps = None          # Spec size for each parameter
        else:
            self.template_grids = orig.template_grids

            self.params_0 = orig.params_0
            self.params_fixed = orig.params_fixed
            self.params_bounds = orig.params_bounds
            self.params_priors = orig.params_priors
            self.params_steps = orig.params_steps

    def reset(self):
        super().reset()

    def add_args(self, config, parser):
        super().add_args(config, parser)
    
    def init_from_args(self, script, config, args):
        super().init_from_args(script, config, args)
    
        # Look up template parameters from the command-line. Figure out the ones that
        # are sampled randomly and ones that are fixed
        params = {}
        grid = self.template_grids[list(self.template_grids.keys())[0]]
        for i, p, ax in grid.enumerate_axes():
            params[p] = Parameter(p)
            params[p].init_from_args(args)

        self.params_0 = {}
        self.params_fixed = {}
        self.params_bounds = {}
        self.params_priors = {}
        self.params_steps = {}

        # Add parameters and priors
        for p in params:
            x_0, bounds, fixed = params[p].generate_initial_value()
            step = params[p].generate_step_size(step_size_factor=0.1)
            self.params_0[p] = x_0
            self.params_bounds[p] = bounds
            self.params_steps[p] = step
            if fixed:
                self.params_fixed[p] = x_0

            if params[p].has_dist():
                self.params_priors[p] = params[p].get_dist()

    def create_trace(self):
        return ModelGridRVFitTrace()

    def reset(self):
        super().reset()

    def create_trace(self):
        return ModelGridRVFitTrace()

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
        else:
            params = None

        if params_fixed is not None:
            pass
        elif self.params_fixed is not None:
            params_fixed = self.params_fixed

        if params is not None and params_fixed is not None:
            params = { **params, **params_fixed }

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
                    if p in params:
                        log_L += self.eval_prior(params_priors[p], params[p])

            if self.trace is not None:
                self.trace.on_calculate_log_L(spectra, templates, rv, params, a)

            return log_L, phi, chi, ndf
        
    def map_log_L(self, spectra, templates, size=10,
                  rv=None, rv_prior=None, rv_bounds=None,
                  params=None, params_fixed=None, params_priors=None, params_bounds=None,
                  squeeze=False):
        # Evaluate log L on a grid

        rv_prior = rv_prior if rv_prior is not None else self.rv_prior
        rv_bounds = rv_bounds if rv_bounds is not None else self.rv_bounds
        params_priors = params_priors if params_priors is not None else self.params_priors
        params_bounds = params_bounds if params_bounds is not None else self.params_bounds
        params_fixed = params_fixed if params_fixed is not None else self.params_fixed

        params_free = self.determine_free_params(params_fixed)

        def get_axis(name, value, bounds, prior):
            if value is None:
                if bounds is not None:
                    min, max = bounds
                elif prior is not None and prior.has_min_max():
                    min, max = prior.get_min_max()
                else:
                    raise Exception(f"Can't determine bounds for parameter `{name}` for map generation.")
                
                value = np.linspace(min, max, size + 1, endpoint=True)

            return np.atleast_1d(value)

        axes = []
        labels = []

        rvv = get_axis('rv', rv, rv_bounds, rv_prior)
        if not squeeze or rvv.size > 1:
            axes.append(rvv)
            labels.append('rv')
        shape = rvv.shape
        
        ppv = {}
        for p in params_free:
            v = get_axis(p, params[p] if params is not None else None, 
                                params_bounds[p] if params_bounds is not None else None,
                                params_priors[p] if params_priors is not None else None)
            ppv[p] = v
            if not squeeze or v.size > 1:
                axes.append(v)
                labels.append(p)
            shape += v.shape
            
        # Construct the data cube    
        log_L = np.empty(shape)
        
        # Iterate over the grid and evaluate the function
        for ix in np.ndindex(*shape):
            rv = rvv[ix[0]]
            params = {}
            for i, (p, v) in enumerate(ppv.items()):
                params[p] = v[ix[i + 1]]

            lp, _, _, _ = self.calculate_log_L(spectra, templates,
                                               rv=rv, rv_prior=rv_prior,
                                               params={ **params, **params_fixed }, params_priors=params_priors)
            log_L[ix] = lp
            
        if squeeze:
            log_L = log_L.squeeze()

        return log_L, axes, labels
        
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
                # Override bounds if outside the grid
                bounds[p] = (max(params_bounds[p][0], grid.get_axis(p).values.min()),
                             min(params_bounds[p][1], grid.get_axis(p).values.max()))

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

        if mode == 'a_params_rv':
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
        elif mode == 'full' or mode == 'params_rv':
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
        elif mode == 'params':
            def pack_params(params):
                return np.array([ params[k] for k in params_free ])
            
            def unpack_params(params):
                pp = {}
                for i, p in enumerate(params_free):
                    pp[p] = params[i]
                for p in params_fixed:
                    pp[p] = params_fixed[p]

                return pp

            def pack_bounds(params_bounds):
                if params_bounds is not None:
                    bounds = [ params_bounds[k] if k in params_bounds else None for k in params_free ]
                else:
                    bounds = [ None for k in params_free ]
                                
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
        
    def get_objective_function(self, spectra, rv_0, rv_prior, params_0, params_priors, params_free, params_fixed=None, mode='full'):
        # Return the objection functionfor optimizers
        # log_L: evaluate the log likelihood

        pack_params, unpack_params, pack_bounds = self.get_packing_functions(params_0, params_free, params_fixed=params_fixed, mode=mode)
        
        if mode == 'a_params_rv':            
            def log_L(a_params_rv):
                a, params, rv = unpack_params(a_params_rv)
                templates, missing = self.get_templates(spectra, params)

                if missing:
                    # Trying to extrapolate outside the grid
                    return -np.inf
                else:
                    log_L, phi, chi, ndf = self.calculate_log_L(spectra, templates, rv, rv_prior=rv_prior, params=params, params_priors=params_priors, a=a)
                    if log_L.ndim == 0:
                        log_L = log_L.item()

                    return log_L
        elif mode == 'full' or mode == 'params_rv':
            def log_L(params_rv):
                params, rv = unpack_params(params_rv)
                templates, missing = self.get_templates(spectra, params)

                if missing:
                    # Trying to extrapolate outside the grid
                    return -np.inf
                else:
                    log_L, _, _, _ = self.calculate_log_L(spectra, templates, rv, rv_prior=rv_prior, params=params, params_priors=params_priors)
                    if log_L.ndim == 0:
                        log_L = log_L.item()

                    return log_L
        elif mode == 'params':
            def log_L(params):
                params = unpack_params(params)
                templates, missing = self.get_templates(spectra, params)

                if missing:
                    # Trying to extrapolate outside the grid
                    return -np.inf
                else:
                    log_L, _, _, _ = self.calculate_log_L(spectra, templates, rv_0, rv_prior=rv_prior, params=params, params_priors=params_priors)
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
                    log_L, _, _, _ = self.calculate_log_L(spectra, templates, rv, rv_prior=rv_prior, params=params_0, params_priors=params_priors)
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
            spectra, rv_0, rv_prior, params_0, params_priors, params_free, params_fixed=params_fixed, mode=mode)
        
        if mode == 'full':
            return self.eval_F_full(spectra, 
                                    rv_0, params_0, rv_bounds=rv_bounds, rv_prior=rv_prior, 
                                    params_bounds=params_bounds, params_priors=params_priors, params_fixed=params_fixed,
                                    step=step)
        if mode == 'a_params_rv':               
            # Calculate a_0
            templates, missing = self.get_templates(spectra, params_0)
            phi_0, chi_0, ndf_0 = self.eval_phi_chi(spectra, templates, rv_0)
            a_0 = self.eval_a(phi_0, chi_0)
            x_0 = pack_params(a_0, params_0, rv_0)
            bounds = pack_bounds(a_0.size * [(-np.inf, np.inf)], params_bounds, rv_bounds)
        elif mode == 'params_rv':                
            x_0 = pack_params(params_0, rv_0)
            bounds = pack_bounds(params_bounds, rv_bounds)
        elif mode == 'params':
            x_0 = pack_params(params_0)
            bounds = pack_bounds(params_bounds)
        elif mode == 'rv':                
            x_0 = pack_params(rv_0)
            bounds = pack_bounds(rv_bounds)
        else:
            raise NotImplementedError()

        bounds = self.get_bounds_array(bounds)

        return self.eval_F_dispatch(x_0, log_L, step, method, bounds)
    
    def eval_F_full(self, spectra, rv_0, params_0, rv_bounds=None, rv_prior=None, params_bounds=None, params_priors=None, params_fixed=None, step=None):
        def matinv_safe(a):
            if isinstance(a, np.ndarray):
                return np.linalg.inv(a)
            else:
                return 1.0 / a
            
        def matmul_safe(*aa):
            m = None
            for a in aa:
                if m is None:
                    m = a
                elif not isinstance(m, np.ndarray) or not isinstance(a, np.ndarray):
                    m = m * a
                else:
                    m = np.matmul(m, a)
            return m
        
        def matshape_safe(a):
            if isinstance(a, np.ndarray):
                return a.shape[0]
            else:
                return 1
        
        # Evaluate the Fisher matrix using Eq. 28 from the paper

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
            spectra, rv_0, rv_prior, params_0, params_priors, params_free, params_fixed=params_fixed, mode='full')
        
        # Calculate derivatives by the model params and RV only, compose Fisher
        # matrix according to Eq. 28
        
        templates, missing = self.get_templates(spectra, params_0)
        phi_0, chi_0, ndf_0 = self.eval_phi_chi(spectra, templates, rv_0)
        nu_0 = np.sqrt(self.eval_nu2(phi_0, chi_0))
        chi_0_inv = matinv_safe(chi_0)

        x_0 = pack_params(params_0, rv_0)
        bounds = pack_bounds(params_bounds, rv_bounds)

        # Calculate Eq. 33

        # Evaluate the Hessian of nu
        def nu(params_rv):
            params, rv = unpack_params(params_rv)
            params, rv = unpack_params(params_rv)
            templates, missing = self.get_templates(spectra, params)
            phi, chi, _ = self.eval_phi_chi(spectra, templates, rv)
            nu2 = self.eval_nu2(phi, chi)
            return np.sqrt(nu2)
        
        dd_nu = nd.Hessian(nu, step=step)
        dd_nu_0 = dd_nu(x_0)

        # Evaluate the Jacobian of phi
        def phi(params_rv):
            params, rv = unpack_params(params_rv)
            templates, missing = self.get_templates(spectra, params)
            phi, _, _ = self.eval_phi_chi(spectra, templates, rv)
            return phi
        
        d_phi = nd.Jacobian(phi, step=step)
        d_phi_0 = d_phi(x_0)

        # Evaluate the Hessian of the priors
        # Here we assume independent priors so really just take the second derivatives
        if params_priors is not None:
            def pi(params_rv):
                params, rv = unpack_params(params_rv)
                
                pi_rv = np.exp(self.eval_prior(rv_prior, rv))

                pi_params = {}
                for i, p in enumerate(params_free):
                    if p in params_priors:
                        pi_params[p] = np.exp(self.eval_prior(params_priors[p], params[p]))
                    else:
                        pi_params[p] = 1.0

                return pack_params(pi_params, pi_rv)

            dd_pi = nd.Derivative(pi, order=2)
            dd_pi_0 = dd_pi(x_0)
        else:
            dd_pi_0 = np.zeros_like(d_phi_0)

        # Assemble the Fisher matrix
        
        # Number of coeff, number of params
        da = matshape_safe(chi_0)
        dp = dd_nu_0.shape[0]
        
        F = np.full((da + dp, da + dp), np.nan, dtype=chi_0.dtype)
        
        F[:da, :da] = chi_0
        F[:da, da:] = -d_phi_0
        F[da:, :da] = -d_phi_0.T
        F[da:, da:] = -nu_0 * dd_nu_0 + matmul_safe(d_phi_0.T, chi_0_inv, d_phi_0) + np.diag(dd_pi_0)

        return F, np.linalg.inv(F)

    def calculate_F(self, spectra, rv_0, params_0, rv_bounds=None, rv_prior=None, params_bounds=None, params_priors=None, params_fixed=None, step=None, mode='full', method='hessian'):
        # Calculate the Fisher matrix using different methods

        return self.eval_F(spectra, rv_0, params_0, rv_bounds=rv_bounds, rv_prior=rv_prior, params_bounds=params_bounds, params_priors=params_priors, params_fixed=params_fixed, step=step, mode=mode, method=method)

    #endregion

    def sample_params_prior(self, p, params_priors, params_fixed, bounds=None):
        # TODO: what if the prior is a callable and not a distribution?
        d = params_priors[p]
        v = d.sample()
        if bounds is not None and p in bounds:
            if bounds[p][0] is not None:
                v = max(v, bounds[p][0])
            if bounds[p][1] is not None:
                v = min(v, bounds[p][1])
        return v
    
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
    
    def prepare_fit(self, spectra, /,
              rv_0=None, rv_bounds=(-500, 500), rv_prior=None, rv_step=None,
              params_0=None, params_bounds=None, params_priors=None, params_steps=None,
              params_fixed=None):
        
        """
        :param params_0: Dictionary of initial values.
        :param params_bounds: Dictionary of tuples, parameters bounds.
        :param params_fixed: Dictionary of fixed parameter values.
        """
        
        rv_0 = rv_0 if rv_0 is not None else self.rv_0
        rv_bounds = rv_bounds if rv_bounds is not None else self.rv_bounds
        rv_prior = rv_prior if rv_prior is not None else self.rv_prior
        rv_step = rv_step if rv_step is not None else self.rv_step

        params_0 = params_0 if params_0 is not None else self.params_0
        params_fixed = params_fixed if params_fixed is not None else self.params_fixed
        params_free = self.determine_free_params(params_fixed)
        params_bounds = params_bounds if params_bounds is not None else self.params_bounds
        params_priors = params_priors if params_priors is not None else self.params_priors
        params_steps = params_steps if params_steps is not None else self.params_steps

        # Make sure all randomly generated parameters are within the grid
        # TODO: this doesn't account for any possible holes
        grid_bounds = self.determine_grid_bounds(params_bounds, params_free)

        if params_0 is None:
            # TODO
            raise NotImplementedError()
        
        if rv_0 is None and spectra is not None:
            _, _, _, rv_0 = self.guess_rv(spectra, None, 
                                          rv_bounds=rv_bounds, rv_prior=rv_prior, 
                                          params_0=params_0, params_fixed=params_fixed,
                                          method='max')
            
        if params_fixed is None:
            params_fixed = []
            
        if params_fixed is None:
            params_fixed = []
            
        if self.template_wlim is None:
            self.determine_wlim(spectra, rv_bounds=rv_bounds)

        # Get objective function
        log_L_fun, pack_params, unpack_params, pack_bounds = self.get_objective_function(
            spectra,
            rv_0, rv_prior,
            params_0, params_priors, params_free, params_fixed=params_fixed,
            mode='params_rv')
        
        # Initial values
        if params_0 is not None and rv_0 is not None:
            x_0 = pack_params(params_0, rv_0)
        else:
            x_0 = None

        # Step size for MCMC
        if params_steps is not None and rv_step is not None:
            steps = pack_params(params_steps, rv_step)
        else:
            steps = None

        # Parameter bounds for optimizers, bounds is a list of tuples, convert to an array
        bounds = grid_bounds
        bounds = pack_bounds(bounds, rv_bounds)
        bounds = self.get_bounds_array(bounds)

        return (rv_0, rv_bounds, rv_prior, rv_step,
                params_0, params_fixed, params_free, params_bounds, params_priors, params_steps,
                log_L_fun, pack_params, unpack_params, pack_bounds,
                x_0, bounds, steps)

    def fit_rv(self, spectra, *,
               rv_0=None, rv_bounds=(-500, 500), rv_prior=None,
               params_0=None, params_bounds=None, params_priors=None, params_fixed=None,
               method="Nelder-Mead", max_iter=None,
               calculate_error=True, calculate_cov=True):
        
        assert isinstance(spectra, dict)
                
        max_iter = max_iter if max_iter is not None else self.max_iter

        # Initialize flux correction model
        if self.flux_corr is None:
            self.init_flux_corr(spectra, rv_bounds=rv_bounds)
        
        (rv_0, rv_bounds, rv_prior, rv_step,
            params_0, params_fixed, params_free, params_bounds, params_priors, params_steps,
            log_L_fun, pack_params, unpack_params, pack_bounds,
            x_0, bounds, steps) = self.prepare_fit(spectra, 
                                            rv_0=rv_0, rv_bounds=rv_bounds, rv_prior=rv_prior,
                                            params_0=params_0, params_bounds=params_bounds,
                                            params_priors=params_priors, params_steps=None,
                                            params_fixed=params_fixed)
        
        # TODO: If no free parameters, fall back to superclass implementation
        if len(params_free) == 0:
            raise NotImplementedError()

        # TODO: If no free parameters, fall back to superclass implementation
        if len(params_free) == 0:
            raise NotImplementedError()

        # Cost function
        def llh(params_rv):
            return -log_L_fun(params_rv)
        
        # Verify that the starting points is valid
        log_L_0 = llh(x_0)
        if np.isinf(log_L_0) or np.isnan(log_L_0):
            raise Exception(f"Invalid starting point for RV fitting.")
        
        if self.trace is not None:
            self.trace.on_fit_rv_start(spectra, None,
                                       rv_0, rv_bounds, rv_prior, rv_step,
                                       params_0, params_bounds, params_priors, params_steps,
                                       log_L_fun)
        
        # Generate the initial simplex
        if steps is not None:
            xx_0 = np.empty((x_0.size + 1, x_0.size))
            xx_0[:x_0.size] = x_0 - np.eye(x_0.size) * steps * 5
            xx_0[-1] = x_0
        else:
            logging.warning('No step size is specified, cannot generate initial simplex.')
            xx_0 = None
        
        if self.trace is not None:
            def callback(x):
                params_fit, rv_fit = unpack_params(x)
                self.trace.on_fit_rv_iter(rv_fit, params_fit)
        else:
            callback = None
        
        # TODO: This is Nelder-Mead only!
        out = minimize(llh, x0=x_0, bounds=bounds, method=method, callback=callback,
                       options=dict(maxiter=self.max_iter, initial_simplex=xx_0))

        if out.success:
            params_fit, rv_fit = unpack_params(out.x)
        else:
            raise Exception(f"Could not fit RV using `{method}`, reason: {out.message}")
        
        # Calculate the flux correction coefficients at best fit values
        templates, missing = self.get_templates(spectra, params_fit)
        phi_fit, chi_fit, ndf_fit = self.eval_phi_chi(spectra, templates, rv_fit)
        a_fit = self.eval_a(phi_fit, chi_fit)
        
        if calculate_error:
            # Error of RV
            F, C = self.eval_F(spectra, rv_fit, params_fit, params_fixed=params_fixed,
                               rv_bounds=rv_bounds, rv_prior=rv_prior,
                               step=rv_step,
                               mode='rv', method='hessian')
            
            with np.errstate(invalid='warn'):
                rv_err = np.sqrt(C).item()

            # Error of the parameters one by one
            params_err = {}
            for i, p in enumerate(params_free):
                pp = { p: params_fit[p] }
                pf = { s: params_fit[s] for j, s in enumerate(params_free) if j != i }
                pf.update({ s: params_fixed[s] for j, s in enumerate(params_fixed) })

                F, C = self.eval_F(spectra, rv_fit, pp, params_fixed=pf,
                                   rv_bounds=rv_bounds, rv_prior=rv_prior,
                                   params_bounds=params_bounds, params_priors=params_priors,
                                   step=1e-3,
                                   mode='params', method='hessian')

                with np.errstate(invalid='warn'):
                    params_err[p] = np.sqrt(C).item()

            for i, p in enumerate(params_fixed):
                params_err[p] =  0.0
        else:
            rv_err = np.nan
            params_err = {}
            for i, p in enumerate(params_free):
                params_err[p] = np.nan
            for i, p in enumerate(params_fixed):
                params_err[p] = np.nan

        if calculate_cov:
            # Calculate the correlated errors from the Fisher matrix            
            F, C = self.eval_F(spectra, 
                               rv_0=rv_fit, rv_bounds=None, rv_prior=None, 
                               params_0=params_fit, params_fixed=params_fixed,
                               params_bounds=params_bounds, params_priors=params_priors,
                               step=1e-3,
                               mode='params_rv', method='hessian')

            # Check if the covariance matrix is positive definite
            with np.errstate(all='raise'):
                try:
                    np.linalg.cholesky(C)
                except LinAlgError as ex:
                    # Matrix is not positive definite
                    F = C = np.full_like(C, np.nan)
                    
            with np.errstate(invalid='warn'):
                err = np.sqrt(np.diag(C)) # sigma
        else:
            C = None

        # If tracing, evaluate the template at the best fit parameters for each exposure
        if self.trace is not None:
            # Evaluate the basis functions
            # TODO: move this outside of function and pass in as arguments
            bases, basis_size = self.get_flux_corr_basis(spectra)

            tt = { arm: [] for arm in spectra }
            for arm in spectra:
                for ei, spec in enumerate(spectra[arm] if isinstance(spectra[arm], list) else [ spectra[arm] ]):
                    temp = templates[arm]
                    
                    psf = self.template_psf[arm] if self.template_psf is not None else None
                    wlim = self.template_wlim[arm] if self.template_wlim is not None else None
                
                    # This is a generic call to preprocess the template which might or
                    # might not include a convolution, depending on the RVFit implementation.
                    # When template convolution is pushed down to the model grid to support
                    # caching, convolution is skipped by the derived classes such as
                    # ModelGridRVFit
                    t = self.process_template(arm, temp, spec, rv_fit, psf=psf, wlim=wlim)
                    self.apply_flux_corr(t, bases[arm][ei], a_fit, renorm=True)

                    tt[arm].append(t)

            # TODO: pass in continuum model for plotting
            #       pass in covariance matrix
            self.trace.on_fit_rv_finish(spectra, None, tt,
                                        rv_0, rv_fit, rv_err, rv_bounds, rv_prior, rv_step,
                                        params_0, params_fit, params_err, params_bounds, params_priors, params_steps,
                                        params_free, C,
                                        log_L_fun)

        return RVFitResults(rv_fit=rv_fit, rv_err=rv_err,
                            params_free=params_free, params_fit=params_fit, params_err=params_err,
                            a_fit=a_fit, a_err=np.full_like(a_fit, np.nan),
                            cov=C)
    
    def randomize_init_params(self, spectra, rv_0=None, rv_bounds=None, rv_prior=None, rv_step=None,
                  params_0=None, params_bounds=None, params_priors=None, params_steps=None,
                  params_fixed=None, cov=None,
                  randomize=False, random_size=()):
        
        params_fixed = params_fixed if params_fixed is not None else self.params_fixed
        params_bounds = params_bounds if params_bounds is not None else self.params_bounds
        params_priors = params_priors if params_priors is not None else self.params_priors
        params_steps = params_steps if params_steps is not None else self.params_steps

        rv_err = None if cov is None else cov[-1, -1]
        rv, rv_err = super().randomize_init_params(spectra,
                                                   rv_0=rv_0, rv_bounds=rv_bounds,
                                                   rv_prior=rv_prior, rv_step=rv_step,
                                                   rv_err=rv_err,
                                                   randomize=randomize, random_size=random_size)

        # Override bounds with grid bounds
        params_free = self.determine_free_params(params_fixed)
        params_bounds = self.determine_grid_bounds(params_bounds, params_free)

        # We need pack_params to construct the covariance matrix
        log_L_fun, pack_params, unpack_params, pack_bounds = self.get_objective_function(
            spectra,
            rv_0, rv_prior,
            params_0, params_priors, params_free, params_fixed=params_fixed,
            mode='params_rv')

        # Generate an initial state for MCMC by sampling the priors randomly

        # Make sure the dict of params is populated        
        if params_0 is None and params_priors is not None:
            params_0 = { p: None for p in params_priors }

        params = {}
        for p in params_0:
            if params_0[p] is None or np.isnan(params_0[p]):
                if params_priors is not None and p in params_priors and params_priors[p] is not None:
                    params[p] = self.sample_params_prior(p, params_priors, params_bounds)
                else:
                    if self.params_0 is not None and p in self.params_0 and self.params_0[p] is not None:
                        params[p] = self.params_0[p]
                    else:
                        raise NotImplementedError()
            else:
                params[p] = params_0[p]

            # Randomize values (0.5% error) as starting point
            if randomize:
                if params_steps is not None and p in params_steps and params_steps[p] is not None:
                    params[p] = params[p] + params_steps[p] * (np.random.rand(*random_size) - 0.5)
                else:
                    params[p] = params[p] * (1.0 + 0.05 * (np.random.rand(*random_size) - 0.5))

            if params_bounds is not None and p in params_bounds and params_bounds[p][0] is not None:
                params[p] = np.maximum(params[p], params_bounds[p][0])
            if params_bounds is not None and p in params_bounds and params_bounds[p][1] is not None:
                params[p] = np.minimum(params[p], params_bounds[p][1])

        if cov is None or np.any(np.isnan(cov)):
            params_err = {}
            for p in params:
                if params_steps is not None and p in params_steps:
                    params_err[p] = params_steps[p] ** 2
                else:
                    params_err[p] = (0.05 * np.mean(params[p])) ** 2 + 1.0

            cov = pack_params(params_err, rv_err)
            cov = np.diag(cov)

        return rv, params, cov

    def run_mcmc(self, spectra, *,
               rv_0=None, rv_bounds=None, rv_prior=None, rv_step=None,
               params_0=None, params_bounds=None, params_priors=None, params_steps=None,
               params_fixed=None, cov=None,
               walkers=None, burnin=None, samples=None, thin=None, gamma=None):
        
        """
        Given a set of spectra and templates, sample from the posterior distribution of RV.

        If no initial guess is provided, an initial state is generated automatically.
        """

        assert isinstance(spectra, dict)

        walkers = walkers if walkers is not None else self.mcmc_walkers
        burnin = burnin if burnin is not None else self.mcmc_burnin
        samples = samples if samples is not None else self.mcmc_samples
        thin = thin if thin is not None else self.mcmc_thin
        gamma = gamma if gamma is not None else self.mcmc_gamma

        (rv_0, rv_bounds, rv_prior, rv_step,
            params_0, params_fixed, params_free, params_bounds, params_priors, params_steps,
            log_L_fun, pack_params, unpack_params, pack_bounds,
            x_0, bounds, steps) = self.prepare_fit(spectra, 
                                            rv_0=rv_0, rv_bounds=rv_bounds, rv_prior=rv_prior, rv_step=rv_step,
                                            params_0=params_0, params_bounds=params_bounds,
                                            params_priors=params_priors, params_steps=params_steps,
                                            params_fixed=params_fixed)

        if bounds is not None and np.any((np.transpose(x_0) < bounds[..., 0]) | (bounds[..., 1] < np.transpose(x_0))):
            raise Exception("Initial state is outside bounds.")

        # Group atmospheric parameters and RV into separate Gibbs steps
        gibbs_blocks = [[ i for i in range(len(params_free))], [len(params_free)]]
        
        mcmc = MCMC(log_L_fun, step_size=steps, gibbs_blocks=gibbs_blocks,
                    walkers=walkers, gamma=0.99)

        res_x, res_lp, accept_rate = mcmc.sample(x_0, burnin, samples, gamma=gamma)
                           
        # Extract results into separate parameters + RV
        # Unpack_params expects that the leading dimension is: # of params + 1 (RV)
        # Result shape: (samples, walkers)
        params, rv = unpack_params(res_x)
        
        return RVFitResults(rv_mcmc=rv, params_mcmc=params, log_L_mcmc=res_lp, accept_rate=accept_rate)
    
