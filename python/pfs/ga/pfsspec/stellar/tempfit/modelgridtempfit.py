import logging
from deprecated import deprecated
from types import SimpleNamespace
import numpy as np
from numpy.linalg import LinAlgError
from scipy.optimize import curve_fit, minimize
from scipy.optimize import minimize_scalar
import numdifftools as nd

from pfs.ga.pfsspec.core.sampling import MCMC
from pfs.ga.pfsspec.core import Physics
from pfs.ga.pfsspec.core.sampling import Parameter, Distribution
from pfs.ga.pfsspec.core.util.timer import Timer
from .tempfit import TempFit
from .tempfitflag import TempFitFlag
from .modelgridtempfitresults import ModelGridTempFitResults
from .modelgridtempfittrace import ModelGridTempFitTrace

from .setup_logger import logger

class ModelGridTempFit(TempFit):
    """
    Performs radial velocity fitting by interpolating templates from
    a model spectrum grid. It can either maximize the likelihood function
    or the significance function of Kaiser (2004).

    Variables
    ---------
    template_grids : dict
        Dictionary of model grids for each spectrograph arm
    params_0 : dict
        Initial template parameters
    params_fixed : dict
        Template parameters that are fixed
    params_bounds : dict
        Bounds for the template parameters
    params_priors : dict
        Prior distributions for the template parameters
    params_steps : dict
        Step sizes for MCMC and numerical differentiation of the template parameters
    """

    def __init__(self, trace=None, correction_model=None, orig=None):
        """
        Initialize the template fitting problem.
        """
        
        super().__init__(trace=trace, correction_model=correction_model, orig=orig)

        if not isinstance(orig, ModelGridTempFit):
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
        return ModelGridTempFitTrace()

    def reset(self):
        super().reset()

    def create_trace(self):
        return ModelGridTempFitTrace()

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

    def get_normalization(self, spectra, templates=None, rv_0=None, params_0=None, params_fixed=None, **kwargs):
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
            raise Exception(f"Template parameters {params} are outside the grid.")
        else:
            return super().get_normalization(spectra, templates, rv_0=rv_0)
        
    def process_template_impl(self, arm, template, spectrum, rv, psf=None, wlim=None):
        # Skip convolution because convolution is pushed down to the
        # model interpolator to support caching            
        return super().process_template_impl(arm, template, spectrum, rv, psf=None, wlim=wlim)
        
    def calculate_log_L(self, spectra, templates,
                        rv, rv_prior=None,
                        params=None, params_fixed=None, params_priors=None,
                        a=None,
                        pp_spec=None,
                        ignore_missing_template=False):
        
        """
        Calculate the logarithm of the likelihood at the given values of RV.
        
        The function calls into the underlying correction model to calculate log L
        and adds the contribution of the priors. While RV can be a vector, the
        model parameters are assumed to be fixed.
        """

        rv_prior = rv_prior if rv_prior is not None else self.rv_prior
        params = params if params is not None else self.params_0
        params_fixed = params_fixed if params_fixed is not None else self.params_fixed
        params_priors = params_priors if params_priors is not None else self.params_priors

        if params is not None and params_fixed is not None:
            params = { **params, **params_fixed }

        if templates is None:
            templates, missing = self.get_templates(spectra, params)
        else:
            missing = False

        if missing:
            msg = f"Template parameters {params} are outside the grid."
            if ignore_missing_template:
                log_L = np.nan
                logger.debug(msg)
            else:
                raise Exception(msg)
        else:
            log_L = super().calculate_log_L(spectra, templates,
                                            rv, rv_prior=rv_prior, a=a,
                                            pp_spec=pp_spec)
            
            if params_priors is not None:
                for p in params_priors.keys():
                    if p in params:
                        log_L += self.eval_prior(params_priors[p], params[p])

        return log_L
        
    def map_log_L(self, spectra, templates=None, size=10,
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

        # Initialize flux correction or continuum models for each arm and exposure
        self.init_correction_models(spectra, rv_bounds=rv_bounds)

        # Generate the map axes
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
            v = get_axis(p, params[p] if params is not None and p in params else None, 
                         params_bounds[p] if params_bounds is not None and p in params_bounds else None,
                         params_priors[p] if params_priors is not None and p in params_priors else None)
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
            pp = {}
            for i, (p, v) in enumerate(ppv.items()):
                pp[p] = v[ix[i + 1]]

            lp = self.calculate_log_L(spectra, templates,
                                      rv=rv,
                                      rv_prior=rv_prior,
                                      params={ **pp, **params_fixed },
                                      params_priors=params_priors,
                                      ignore_missing_template=True)
            log_L[ix] = lp
            
        if squeeze:
            log_L = log_L.squeeze()

        if self.trace is not None:
            params = { l: v for l, v in zip(labels, axes) }
            self.trace.on_map_log_L(spectra, templates, params, log_L)

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
            
    def get_param_packing_functions(self, params_free, mode='full'):
        # Correction model packing functions
        pp_a, up_a, pb_a = self._get_param_packing_functions_a(mode=mode)

        # Other parameter packing functions
        pp_params, up_params, pb_params = self._get_param_packing_functions_params(params_free, mode=mode)
        pp_rv, up_rv, pb_rv = self._get_param_packing_functions_rv(mode=mode)

        if pp_params is None:
            # If we don't optimize for the model parameters
            return super().get_param_packing_functions(mode=mode)
        else:
            if mode == 'full' or (pp_a is not None and pp_rv is not None):
                def pack_params(a, params, rv):
                    return np.concatenate([ pp_a(a), pp_params(params), pp_rv(rv) ], axis=-1)
                
                def unpack_params(a_params_rv):
                    rv = up_rv(a_params_rv[..., -1])
                    params = up_params(a_params_rv[..., :-1])
                    a = up_a(a_params_rv[..., :-1 - len(params)])
                    
                    return a, params, rv

                def pack_bounds(a_bounds, params_bounds, rv_bounds):
                    return pb_a(a_bounds) + pb_params(params_bounds) + pb_rv(rv_bounds)
            elif pp_a is None and pp_rv is not None:
                # params + RV only
                def pack_params(params, rv):
                    return np.concatenate([ pp_params(params), pp_rv(rv) ], axis=-1)
                
                def unpack_params(params_rv):
                    rv = up_rv(params_rv[..., -1])
                    params = up_params(params_rv[..., :-1])
                    
                    return params, rv

                def pack_bounds(params_bounds, rv_bounds):
                    return pb_params(params_bounds) + pb_rv(rv_bounds)
            elif pp_a is not None and pp_rv is None:
                # coefficients + params only
                def pack_params(a, params):
                    return np.concatenate([ pp_a(a), pp_params(params) ], axis=-1)
                
                def unpack_params(a_params):
                    params = up_params(a_params)
                    a = up_a(a_params[:-len(params)])
                    
                    return a, params

                def pack_bounds(a_bounds, params_bounds):
                    return pb_a(a_bounds) + pb_params(params_bounds)
            elif pp_a is None and pp_rv is None:
                # params only
                pack_params, unpack_params, pack_bounds = pp_params, up_params, pb_params
            else:
                pack_params, unpack_params, pack_bounds = None, None, None

        return pack_params, unpack_params, pack_bounds
        
    def _get_param_packing_functions_params(self, params_free, mode='full'):
        """
        Return functions that pack and unpack the parameters subject to optimization
        into a single 1d or 2d array and back.
        """

        modes = mode.split('_')

        if mode == 'full' or 'params' in modes:
            def pack_params(params):
                pp = np.array([ params[k] for k in params_free ])
                if np.ndim(pp) == 2:
                    pp = pp.T
                elif np.ndim(pp) < 2:
                    pp = pp[None, :]
                return pp
            
            def unpack_params(params):
                pp = {}
                for i, p in enumerate(params_free):
                    pp[p] = params[..., -len(params_free) + i]
                    if pp[p].size == 1:
                        pp[p] = pp[p].item()
                return pp

            def pack_bounds(params_bounds):
                if params_bounds is not None:
                    bounds = [ params_bounds[k] if k in params_bounds else None for k in params_free ]
                else:
                    bounds = [ None for k in params_free ]
                                
                return bounds
        elif 'params' not in modes:
            pack_params, unpack_params, pack_bounds = None, None, None
        else:
            raise NotImplementedError()
        
        return pack_params, unpack_params, pack_bounds
        
    def get_objective_function(self, spectra,
                               rv_0, rv_fixed, rv_prior,
                               params_0, params_priors, params_free, params_fixed=None,
                               mode='full',
                               pp_spec=None):
    
        """
        Return the objective function and parameter packing/unpacking functions for optimizers

        Parameters
        ----------
        spectra : dict or dict of list
            Dictionary of spectra for each arm and exposure
        """
        
        # Return the objection functionfor optimizers
        # log_L: evaluate the log likelihood

        pack_params, unpack_params, pack_bounds = self.get_param_packing_functions(params_free, mode=mode)
        
        if mode == 'full' or mode == 'a_params_rv':
            def log_L(a_params_rv):
                a, params, rv = unpack_params(a_params_rv)
                params = { **params, **params_fixed }
                templates, missing = self.get_templates(spectra, params)

                if missing:
                    # Trying to extrapolate outside the grid
                    return -np.inf
                else:
                    log_L = self.calculate_log_L(spectra, templates,
                                                 rv, rv_prior=rv_prior,
                                                 params=params, params_priors=params_priors,
                                                 a=a,
                                                 pp_spec=pp_spec)
                    return log_L
        elif mode == 'params_rv':
            def log_L(params_rv):
                params, rv = unpack_params(params_rv)
                params = { **params, **params_fixed }
                templates, missing = self.get_templates(spectra, params)

                if missing:
                    # Trying to extrapolate outside the grid
                    return -np.inf
                else:
                    log_L = self.calculate_log_L(spectra, templates,
                                                 rv, rv_prior=rv_prior,
                                                 params=params, params_priors=params_priors,
                                                 pp_spec=pp_spec)
                    return log_L
        elif mode == 'params':
            def log_L(params):
                params = unpack_params(params)
                params = { **params, **params_fixed }
                templates, missing = self.get_templates(spectra, params)

                if missing:
                    # Trying to extrapolate outside the grid
                    return -np.inf
                else:
                    log_L = self.calculate_log_L(spectra, templates,
                                                 rv_0, rv_prior=rv_prior,
                                                 params=params, params_priors=params_priors,
                                                 pp_spec=pp_spec)
                    return log_L
        elif mode == 'rv':
            def log_L(rv):
                rv = unpack_params(rv)
                params = { **params_0, **params_fixed }
                templates, missing = self.get_templates(spectra, params)

                if missing:
                    # Trying to extrapolate outside the grid
                    return -np.inf
                else:
                    log_L = self.calculate_log_L(spectra, templates,
                                                 rv, rv_prior=rv_prior,
                                                 params=params_0, params_priors=params_priors,
                                                 pp_spec=pp_spec)
                    return log_L
        else:
            raise NotImplementedError()
        
        return log_L, pack_params, unpack_params, pack_bounds

    def check_bounds_edge(self, state, eps=1e-3):
        """
        Check if the fitted parameters are at the edge of the bounds.
        """

        super().check_bounds_edge(state, eps=eps)

        for p in state.params_free:
            if state.params_fit is not None and p in state.params_fit:
                if state.params_bounds is not None and p in state.params_bounds and state.params_bounds[p] is not None:
                    if state.params_bounds[p][0] is not None and np.abs(state.params_fit[p] - state.params_bounds[p][0]) < eps:
                        logger.warning(f"Parameter {p} {state.params_fit[p]} is at the lower bound {state.params_bounds[p][0]}.")
                        state.params_flags[p] |= TempFitFlag.PARAMEDGE
                    if state.params_bounds[p][1] is not None and np.abs(state.params_fit[p] - state.params_bounds[p][1]) < eps:
                        logger.warning(f"Parameter {p} {state.params_fit[p]} is at the upper bound {state.params_bounds[p][1]}.")
                        state.params_flags[p] |= TempFitFlag.PARAMEDGE

            if state.params_flags[p] != 0:
                state.flags |= TempFitFlag.BADCONVERGE

    def check_prior_unlikely(self, state, lp_limit=-5):
        """
        Check if the priors are unlikely. This is a warning only, not an error.
        """

        super().check_prior_unlikely(state, lp_limit=lp_limit)
        
        if state.params_fit is not None and state.params_priors is not None:
            for p in state.params_fit:
                if state.params_fit[p] is not None and p in state.params_priors:
                    lp = self.eval_prior(state.params_priors[p], state.params_fit[p])
                    if lp is not None and lp / np.log(10) < lp_limit:
                        logger.warning(f"Prior for parameter {p} {state.params_fit[p]} is very unlikely with lp {lp}.")
                        state.params_flags[p] |= TempFitFlag.UNLIKELYPRIOR

    def calculate_F(self, spectra,
                    rv_0, params_0,
                    rv_bounds=None, rv_prior=None, rv_fixed=None,
                    params_bounds=None, params_priors=None, params_fixed=None,
                    step=None, mode='full', method='hessian',
                    pp_spec=None):
        
        """
        Evaluate the Fisher matrix around the provided rv_0 and params_0
        values. The corresponding a_0 best fit flux correction will be
        evaluated at the optimum. The Hessian will be calculated wrt either
        RV only, or rv and the template parameters or rv, the template parameters
        and the flux correction coefficients. Alternatively, the covariance
        matrix will be determined using MCMC.
        """

        # Collect fixed and free template parameters
        rv_fixed = rv_fixed if rv_fixed is not None else self.rv_fixed
        rv_prior = rv_prior if rv_prior is not None else self.rv_prior
        rv_bounds = rv_bounds if rv_bounds is not None else self.rv_bounds

        params_priors = params_priors if params_priors is not None else self.params_priors
        
        params_fixed = params_fixed if params_fixed is not None else self.params_fixed
        params_free = self.determine_free_params(params_fixed)
        
        params_bounds = params_bounds if params_bounds is not None else self.params_bounds
        params_bounds = self.determine_grid_bounds(self.params_bounds, params_free)

        params_count = (0 if rv_fixed else 1) + len(params_free)
        logger.info(f"Calculating the fisher matrix for {params_count} parameters with mode `{mode}` using method `{method}`.")
        
        # Get objective function
        log_L, pack_params, unpack_params, pack_bounds = self.get_objective_function(
            spectra,
            rv_0, rv_fixed, rv_prior,
            params_0, params_priors, params_free, params_fixed=params_fixed,
            mode=mode,
            pp_spec=pp_spec)
        
        if mode == 'full' or mode == 'a_params_rv':
            # Calculate a_0
            templates, missing = self.get_templates(spectra, params_0)
            a_0, _, _ = self.calculate_coeffs(spectra, templates, rv_0, pp_spec=pp_spec)
            x_0 = pack_params(a_0, params_0, rv_0)[0]
            bounds = pack_bounds(np.size(a_0) * [(-np.inf, np.inf)], params_bounds, rv_bounds)
        elif mode == 'params_rv':
            x_0 = pack_params(params_0, rv_0)[0]
            bounds = pack_bounds(params_bounds, rv_bounds)
        elif mode == 'params':
            x_0 = pack_params(params_0)[0]
            bounds = pack_bounds(params_bounds)
        elif mode == 'rv':
            x_0 = pack_params(rv_0)[0]
            bounds = pack_bounds(rv_bounds)
        else:
            raise NotImplementedError()

        bounds = self.get_bounds_array(bounds)

        # TODO: count function evaluations

        F, C = self.eval_F_dispatch(x_0, log_L, step, method, bounds)

        return F, C
    
    def calculate_F_full(self, spectra,
                    rv_0, params_0,
                    rv_bounds=None, rv_prior=None, rv_fixed=None,
                    params_bounds=None, params_priors=None, params_fixed=None,
                    step=None,
                    pp_spec=None):
        """
        Evaluate the Fisher matrix that includes the RV, the template parameters, as well
        as the flux correction coefficients.
        """

        # This is specific to the flux correction model
        # Consider moving it elsewhere
        raise NotImplementedError()
        
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
        rv_fixed = rv_fixed if rv_fixed is not None else self.rv_fixed
        rv_prior = rv_prior if rv_prior is not None else self.rv_prior
        rv_bounds = rv_bounds if rv_bounds is not None else self.rv_bounds

        params_priors = params_priors if params_priors is not None else self.params_priors
        
        params_fixed = params_fixed if params_fixed is not None else self.params_fixed
        params_free = self.determine_free_params(params_fixed)
        
        params_bounds = params_bounds if params_bounds is not None else self.params_bounds
        params_bounds = self.determine_grid_bounds(self.params_bounds, params_free)
        
        # Get objective function
        log_L, pack_params, unpack_params, pack_bounds = self.get_objective_function(
            spectra,
            rv_0, rv_fixed, rv_prior,
            params_0, params_priors, params_free, params_fixed=params_fixed,
            mode='full',
            pp_spec=pp_spec)
        
        # Calculate derivatives by the model params and RV only, compose Fisher
        # matrix according to Eq. 28
        
        templates, missing = self.get_templates(spectra, params_0)
        phi_0, chi_0, ndf_0 = self.eval_phi_chi(spectra, templates, rv_0)
        nu_0 = np.sqrt(self.eval_nu2(phi_0, chi_0))
        chi_0_inv = matinv_safe(chi_0)

        x_0 = pack_params(params_0, rv_0)[0]
        bounds = pack_bounds(params_bounds, rv_bounds)

        # Calculate Eq. 33

        # Evaluate the Hessian of nu
        def nu(params_rv):
            params, rv = unpack_params(params_rv)
            params = { **params, **params_fixed }
            templates, missing = self.get_templates(spectra, params)
            phi, chi, _ = self.eval_phi_chi(spectra, templates, rv)
            nu2 = self.eval_nu2(phi, chi)
            return np.sqrt(nu2)
        
        dd_nu = nd.Hessian(nu, step=step)
        dd_nu_0 = dd_nu(x_0)

        # Evaluate the Jacobian of phi
        def phi(params_rv):
            params, rv = unpack_params(params_rv)
            params = { **params, **params_fixed }
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
    
    def guess_rv(self, spectra, templates=None, /,
                 rv_bounds=None, rv_prior=None, rv_step=None,
                 params_0=None, params_fixed=None,
                 steps=None,
                 method='lorentz',
                 pp_spec=None):
        
        """
        Guess an initial state to close best RV, either using the supplied set of templates or
        initial model parameters.
        """

        # TODO: Maybe extend this to do a grid search in model parameters

        assert isinstance(spectra, dict)

        params_0 = params_0 if params_0 is not None else self.params_0
        params_fixed = params_fixed if params_fixed is not None else self.params_fixed

        self.init_correction_models(spectra, rv_bounds)

        if templates is None:
            # Look up the templates from the grid and pass those to the the parent class
            params = params_0.copy()
            if params_fixed is not None:
                params.update(params_fixed)
            templates, missing = self.get_templates(spectra, params)
            if missing:
                raise Exception('Cannot find an initial matching template to start TempFit.')
            
            logger.info(f"Using template with parameters {params} to guess RV.")

        return super().guess_rv(spectra, templates,
                                     rv_bounds=rv_bounds, rv_prior=rv_prior, rv_step=rv_step,
                                     steps=steps,
                                     method=method,
                                     pp_spec=pp_spec)
    
    def prepare_fit(self, spectra, /,
                    rv_0=None, rv_bounds=None, rv_prior=None, rv_step=None, rv_fixed=None,
                    params_0=None, params_bounds=None, params_priors=None, params_steps=None, params_fixed=None,
                    pp_spec=None):
        
        """
        Prepare the template fitting problem. This function initializes the
        template fitting problem and returns a state object that contains
        all the information needed to run the fitting.

        Parameters
        ----------
        spectra : dict or dict of list
            Dictionary of spectra for each arm and exposure
        rv_0 : float
            Initial guess for the RV
        rv_bounds : tuple
            Tuple of bounds for the RV
        rv_prior : callable
            Callable that returns the prior for the RV
        rv_step : float
            Step size for the RV
        rv_fixed : bool
            If True, the RV is fixed to the initial guess
        params_0 : dict
            Dictionary of initial values for the template parameters
        params_bounds : dict
            Dictionary of tuples, parameters bounds
        params_priors : dict
            Dictionary of priors for the template parameters
        params_steps : dict
            Dictionary of step sizes for the template parameters
        params_fixed : dict
            Dictionary of fixed parameter values
        pp_spec : dict of list
            Preprocessed spectra

        Returns
        -------
        state : SimpleNamespace
            State object that contains all the information needed to run the fitting
        """

        state = SimpleNamespace()

        state.rv_0 = rv_0 if rv_0 is not None else self.rv_0
        state.rv_fixed = rv_fixed if rv_fixed is not None else self.rv_fixed
        state.rv_bounds = rv_bounds if rv_bounds is not None else self.rv_bounds
        state.rv_prior = rv_prior if rv_prior is not None else self.rv_prior
        state.rv_step = rv_step if rv_step is not None else self.rv_step
        
        state.params_0 = params_0 if params_0 is not None else self.params_0
        state.params_fixed = params_fixed if params_fixed is not None else self.params_fixed
        state.params_bounds = params_bounds if params_bounds is not None else self.params_bounds
        state.params_priors = params_priors if params_priors is not None else self.params_priors
        state.params_steps = params_steps if params_steps is not None else self.params_steps

        # Determine the list of free parameters
        state.params_free = self.determine_free_params(state.params_fixed)

        # Determine the (buffered) wavelength limit in which the templates will be convolved
        # with the PSF. This should be slightly larger than the observed wavelength range.
        # TODO: this has side-effect, add template_wlim to the state instead?
        if self.template_wlim is None:
            # Use different template wlim for each arm but same for each exposure
            self.template_wlim = {}
            wlim = self.determine_wlim(spectra, per_arm=True, per_exp=False, rv_bounds=rv_bounds)
            for mi, arm in enumerate(spectra):
                self.template_wlim[arm] = wlim[mi]

        if state.params_0 is None:
            # TODO
            raise NotImplementedError()
                    
        if state.params_fixed is None:
            state.params_fixed = []

        # Calculate the pre-normalization constants. This is only here to make sure that the
        # matrix elements during calculations stay around unity
        # TODO: this has side effect, should add spec_norm and temp_norm to the state instead?
        if self.spec_norm is None or self.temp_norm is None:
            self.spec_norm, self.temp_norm = self.get_normalization(
                spectra,
                rv_0=state.rv_0,
                params_0=state.params_0,
                params_fixed=state.params_fixed)

        # Preprocess the spectra
        if pp_spec is not None:
            state.pp_spec = pp_spec
        else:
            state.pp_spec = self.preprocess_spectra(spectra)
        
        # Get objective function
        state.log_L_fun, state.pack_params, state.unpack_params, state.pack_bounds = self.get_objective_function(
            spectra,
            state.rv_0, state.rv_fixed, state.rv_prior,
            state.params_0, state.params_priors, state.params_free, params_fixed=state.params_fixed,
            mode='params_rv',
            pp_spec=state.pp_spec)
        
        # Initial values
        if state.params_0 is not None and state.rv_0 is not None:
            state.x_0 = state.pack_params(state.params_0, state.rv_0)[0]
        else:
            state.x_0 = None

        if state.x_0 is not None:
            state.log_L_0 = state.log_L_fun(state.x_0)
        else:
            state.log_L_0 = None

        # Step size for MCMC
        if state.params_steps is not None and state.rv_step is not None:
            state.steps = state.pack_params(state.params_steps, state.rv_step)[0]
        else:
            state.steps = None

        # Parameter bounds for optimizers, bounds is a list of tuples, convert to an array
        
        # Make sure all randomly generated parameters are within the grid
        # TODO: this doesn't account for any possible holes
        bounds = self.determine_grid_bounds(state.params_bounds, state.params_free)
        bounds = state.pack_bounds(bounds, state.rv_bounds)
        state.bounds = self.get_bounds_array(bounds)

        # Set default values of the flags
        state.flags = TempFitFlag.OK
        state.rv_flags = TempFitFlag.OK
        state.params_flags = { p: TempFitFlag.OK for p in state.params_0 }

        return state

    @deprecated("Use `run_ml` instead.")
    def fit_rv(self, spectra, **kwargs):
        return self.run_ml(spectra, **kwargs)

    def run_ml(self, spectra, /,
               rv_0=None, rv_bounds=None, rv_prior=None, rv_fixed=None,
               params_0=None, params_bounds=None, params_priors=None, params_fixed=None,
               method="Nelder-Mead", max_iter=None,
               calculate_error=True, calculate_cov=True):
        """
        Given a set of spectra and template grid, find the best fit RV, as well as template
        parameters by maximizing the likelihood function. Spectra are assumed to be of the same
        object in different wavelength ranges, with multiple exposures.

        Parameters
        ----------
        spectra : dict of Spectrum or dict of list of Spectrum
            Observed spectra
        rv_0 : float
            Initial guess for the RV
        rv_bounds : tuple
            RV bounds to limit the search for initial RV if not provided as well as
            limits to the fit.
        rv_prior : Distribution or callable
            Prior distribution for the RV
        rv_fixed : bool
            If True, the RV is fixed and no optimization is performed.
        params_0 : dict
            Initial guess for the template parameters
        params_bounds : dict
            Bounds for the template parameters
        params_priors : dict
            Prior distributions for the template parameters
        params_fixed : dict
            Fixed template parameters with values.
        method : str
            Optimization method to use: 'bounded' or 'grid'
        max_iter: int
            Maximum number of iterations for the optimization
        calculate_error : bool
            If True, calculate the error of the RV from the Fisher matrix.

        If no initial guess is provided, `rv_0` is determined automatically. If `rv_fixed` is
        `True`, the radial velocity is not fitted. If a template parameter is specified in
        `params_fixed`, no optimization is performed for that parameter.
        """
        
        assert isinstance(spectra, dict)
                
        max_iter = max_iter if max_iter is not None else self.max_iter

        # Initialize flux correction or continuum models for each arm and exposure
        self.init_correction_models(spectra, rv_bounds=rv_bounds)
        
        state = self.prepare_fit(
                spectra,
                rv_0=rv_0, rv_fixed=rv_fixed,
                rv_bounds=rv_bounds, rv_prior=rv_prior,
                params_0=params_0, params_bounds=params_bounds,
                params_priors=params_priors, params_steps=None,
                params_fixed=params_fixed)
                
        # TODO: If no free parameters, fall back to superclass implementation
        if len(state.params_free) == 0:
            raise NotImplementedError()
        
        # If rv_0 is not provided, guess it
        if state.rv_0 is None and spectra is not None:
            _, _, state.rv_0, state.log_L_0 = self.guess_rv(
                spectra,
                None, 
                rv_bounds=state.rv_bounds, rv_prior=state.rv_prior, rv_step=state.rv_step,
                params_0=state.params_0, params_fixed=state.params_fixed,
                method='max',
                pp_spec=state.pp_spec)
            
            # Update initial values
            if state.params_0 is not None and state.rv_0 is not None:
                state.x_0 = state.pack_params(state.params_0, state.rv_0)[0]

            if state.rv_fixed:
                logger.warning("No value of RV is provided, yet not fitting RV. The guessed value will be used.")

        if self.trace is not None:
            # When tracing, create a full list of included and excluded wavelengths, independent
            # of any masks defines by the spectra. This is for visualization purposes only.
            wave_include, wave_exclude = self.get_wave_include_exclude()
            self.trace.on_fit_rv_start(spectra, None,
                                       state.rv_0, state.rv_bounds, state.rv_prior, state.rv_step,
                                       state.params_0, state.params_bounds, state.params_priors, state.params_steps,
                                       state.log_L_0, state.log_L_fun,
                                       wave_include=wave_include, wave_exclude=wave_exclude)

        # Cost function - here we don't have to distinguish between the two cases of `rv_fixed`
        # because `prepare_fit` already returns the right function.
        def llh(params_rv):
            return -state.log_L_fun(params_rv)
        
        # Verify that the starting point is valid
        log_L_0 = llh(state.x_0)
        if np.isinf(log_L_0) or np.isnan(log_L_0):
            all_params = {**state.params_0, **state.params_fixed}
            raise Exception(f"Invalid starting point for template fitting. Are the parameters {all_params} outside the grid?")
        
        # Call the optimizer
        if method == 'Nelder-Mead':
            if self.trace is not None:
                # When tracing, evaluate the log likelihood at each iteration of
                # the optimization to verify the convergence
                def callback(x):
                    params, rv = state.unpack_params(x)
                    log_L = state.log_L_fun(x)
                    self.trace.on_fit_rv_iter(rv, params, log_L, state.log_L_fun)
            else:
                callback = None
        
            x_fit, fl = self.optimize_nelder_mead(state.x_0, state.steps, state.bounds,
                                                  llh,
                                                  method, max_iter,
                                                  callback=callback)
            state.flags |= fl
        else:
            raise NotImplementedError()
        
        # Unpack the optimized parameters
        state.params_fit, state.rv_fit = state.unpack_params(x_fit)
        state.params_fit = { **state.params_fit, **state.params_fixed }

        # Check if the fit is not at the edge of the bounds or priors and
        # that the results are not very unlikely by the priors
        # Set flags accordingly
        self.check_bounds_edge(state)
        self.check_prior_unlikely(state)

        # TODO: check if log_L decreased
                
        # Calculate the flux correction or continuum fit coefficients at best fit values
        templates, missing = self.get_templates(spectra, state.params_fit)
        state.a_fit, _, _ = self.calculate_coeffs(spectra, templates,
                                                  state.rv_fit,
                                                  pp_spec=state.pp_spec)
        
        # TODO: flags for error and covariance

        if calculate_error:
            state.rv_err, state.params_err = self.ml_calculate_error(spectra, state)
        else:
            # TODO: add flag that marks if error is not calculated
            state.rv_err = np.nan
            state.params_err = {}
            for i, p in enumerate(state.params_free):
                state.params_err[p] = np.nan
            for i, p in enumerate(state.params_fixed):
                state.params_err[p] = np.nan

        if calculate_cov:
            # TODO: pass in state
            state.C = self.ml_calculate_cov(spectra, state)
        else:
            state.C = None

        if self.trace is not None:
            # If tracing, evaluate the template at the best fit parameters for each exposure
            ss, tt = self.get_spectra_templates_for_trace(spectra, templates, state.rv_fit, state.a_fit)

            # Wrap log_L_fun to expect a single parameter only and use the best fit model parameters
            # This is for plptting purposes only
            def log_L_fun(rv):
                return self.calculate_log_L(spectra, templates,
                                            rv, rv_prior=state.rv_prior, params=state.params_fit,
                                            params_priors=state.params_priors,
                                            pp_spec=state.pp_spec)

            self.trace.on_fit_rv_finish(spectra, None, tt,
                                        state.rv_0, state.rv_fit, state.rv_err, state.rv_bounds, state.rv_prior, state.rv_step, state.rv_fixed,
                                        state.params_0, state.params_fit, state.params_err, state.params_bounds, state.params_priors, state.params_steps, state.params_free,
                                        state.C,
                                        state.log_L_0, log_L_fun(state.rv_fit), log_L_fun)

        return ModelGridTempFitResults(rv_fit=state.rv_fit, rv_err=state.rv_err, rv_flags=state.rv_flags,
                                       params_free=state.params_free, params_fit=state.params_fit, params_err=state.params_err, params_flags=state.params_flags,
                                       a_fit=state.a_fit, a_err=np.full_like(state.a_fit, np.nan),
                                       cov=state.C,
                                       flags=state.flags)
    
    def optimize_nelder_mead(self, x_0, steps, bounds,
                             llh,
                             method,
                             max_iter,
                             callback=None):
        
        """
        Run the Nelder-Mead optimizer to fit the RV and template parameters.
        """

        flags = 0

        # Generate the initial simplex for the Nelder-Mead method
        if steps is not None:
            xx_0 = np.empty((x_0.size + 1, x_0.size))
            xx_0[:x_0.size] = x_0 - np.eye(x_0.size) * steps * 5
            xx_0[-1] = x_0
        else:
            logger.warning('No step size is specified, cannot generate initial simplex.')
            xx_0 = None
            flags |= TempFitFlag.BADINIT

        with Timer("Starting Nelder-Mead optimization", logger=logger, level=logging.INFO):
            out = minimize(llh,
                           x0=x_0, bounds=bounds, method=method, callback=callback,
                           options=dict(maxiter=max_iter, initial_simplex=xx_0))

        if out.success:
            # Calculate the volume of the initial and final simplex
            def vol(xx):
                if xx is not None:
                    return np.linalg.det(xx[:-1] - xx[-1]) / np.math.factorial(xx.shape[1])
                else:
                    return np.nan
            
            vol_0 = vol(xx_0)
            vol_f = vol(out.final_simplex[0])

            logger.info(f"Nelder-Mead message: {out.message}")
            logger.info(f"Nelder-Mead number of iterations: {out.nit}, "
                         f"number of function evaluations: {out.nfev}")
            logger.info(f"Nelder-Mead initial simplex volume: {vol_0:0.3f}, final simplex volume: {vol_f:0.3f}")

            if max_iter == out.nit:
                logger.warning("Nelder-Mead optimization reached the maximum number of iterations.")
                flags |= TempFitFlag.MAXITER
                
            return out.x, flags
        else:
            raise Exception(f"Could not fit RV using `{method}`, reason: {out.message}")

    def ml_calculate_error(self, spectra, state):
        
        """
        Given the best fit parameters, calculate the errors of the RV and
        template parameters using the Fisher matrix.
        """
        
        if state.rv_fixed:
            rv_err = np.nan
        else:
            # Error of RV only!
            # TODO: pass in state
            F, C = self.calculate_F(spectra, state.rv_fit, state.params_fit,
                                    rv_bounds=state.rv_bounds, rv_prior=state.rv_prior, rv_fixed=state.rv_fixed,
                                    params_fixed=state.params_fixed,
                                    step=state.rv_step,
                                    mode='rv', method='hessian',
                                    pp_spec=state.pp_spec)
            
            with np.errstate(invalid='warn'):
                rv_err = np.sqrt(C).item()

        # Error of the parameters one by one
        params_err = {}
        for i, p in enumerate(state.params_free):
            pp = { p: state.params_fit[p] }
            pf = { s: state.params_fit[s] for j, s in enumerate(state.params_free) if j != i }
            pf.update({ s: state.params_fixed[s] for j, s in enumerate(state.params_fixed) })

            # TODO: pass in state
            F, C = self.calculate_F(spectra, state.rv_fit, pp,
                                    rv_bounds=state.rv_bounds, rv_prior=state.rv_prior, rv_fixed=state.rv_fixed,
                                    params_bounds=state.params_bounds, params_priors=state.params_priors, params_fixed=pf,
                                    step=1e-3,
                                    mode='params', method='hessian',
                                    pp_spec=state.pp_spec)

            with np.errstate(invalid='warn'):
                params_err[p] = np.sqrt(C).item()

        for i, p in enumerate(state.params_fixed):
            params_err[p] =  0.0

        return rv_err, params_err

    def ml_calculate_cov(self, spectra, state):
        """
        Given the best fit parameters, calculate the covariance matrix of the RV and
        template parameters using the Fisher matrix.
        """
        
        # Calculate the correlated errors from the Fisher matrix
        F, C = self.calculate_F(spectra, 
                                rv_0=state.rv_fit, rv_bounds=None, rv_prior=None, rv_fixed=state.rv_fixed,
                                params_0=state.params_fit, params_fixed=state.params_fixed,
                                params_bounds=state.params_bounds, params_priors=state.params_priors,
                                step=1e-3,
                                mode='params_rv', method='hessian',
                                pp_spec=state.pp_spec)

        # Check if the covariance matrix is positive definite
        with np.errstate(all='raise'):
            try:
                np.linalg.cholesky(C)
            except LinAlgError as ex:
                # Matrix is not positive definite
                F = C = np.full_like(C, np.nan)
                
        # with np.errstate(invalid='warn'):
        #     err = np.sqrt(np.diag(C)) # sigma

        return C
    
    def randomize_init_params(self, spectra, rv_0=None, rv_bounds=None, rv_prior=None, rv_step=None, rv_fixed=None,
                              params_0=None, params_bounds=None, params_priors=None, params_steps=None, params_fixed=None,
                              cov=None, randomize=False, random_size=(),
                              pp_spec=None):
        
        """
        Randomize the initial parameters for MCMC.

        If no initial guess is provided, an initial state is generated automatically.

        Parameters
        ----------
        spectra : dict or dict of list
            Dictionary of spectra for each arm and exposure
        rv_0 : float
            Initial guess for the RV
        rv_bounds : tuple
            RV limits for the fitting algorithm.
        rv_prior : Distribution or callable
            Prior distribution for the RV
        rv_step : float
            Step size for MCMC or numerical differentiation
        rv_fixed : bool
            Whether the RV is fixed
        params_0 : dict
            Initial guess for the template parameters
        params_bounds : dict
            Bounds for the template parameters
        params_priors : dict
            Prior distributions for the template parameters
        params_steps : dict
            Step sizes for MCMC or numerical differentiation
        params_fixed : dict
            Fixed template parameters with values
        cov : array
            Covariance matrix of the parameters
        randomize : bool
            Randomize the initial parameters, if False this function is just a pass through unless
            some of the parameters are None because then an initial value is randomly generated.
        random_size : tuple
            Size of the random array to generate

        Returns
        -------
        rv : float
            Randomized RV
        params : dict
            Randomized template parameters
        cov : array
            Initial covariance for adaptive parameter sampling
        """
        
        params_fixed = params_fixed if params_fixed is not None else self.params_fixed
        params_bounds = params_bounds if params_bounds is not None else self.params_bounds
        params_priors = params_priors if params_priors is not None else self.params_priors
        params_steps = params_steps if params_steps is not None else self.params_steps

        rv_err = None if cov is None else cov[-1, -1]
        rv, rv_err = super().randomize_init_params(spectra,
                                                   rv_0=rv_0, rv_bounds=rv_bounds,
                                                   rv_prior=rv_prior, rv_step=rv_step, rv_fixed=rv_fixed,
                                                   rv_err=rv_err,
                                                   randomize=randomize, random_size=random_size)

        # Override bounds with grid bounds
        params_free = self.determine_free_params(params_fixed)
        params_bounds = self.determine_grid_bounds(params_bounds, params_free)

        # We need pack_params to construct the covariance matrix
        log_L_fun, pack_params, unpack_params, pack_bounds = self.get_objective_function(
            spectra,
            rv_0, rv_fixed, rv_prior,
            params_0, params_priors, params_free, params_fixed=params_fixed,
            mode='params_rv',
            pp_spec=pp_spec)

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

            cov = pack_params(params_err, rv_err)[0]
            cov = np.diag(cov)

        return rv, params, cov

    def run_mcmc(self, spectra, *,
                 rv_0=None, rv_bounds=None, rv_prior=None, rv_step=None, rv_fixed=None,
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

        # Initialize flux correction or continuum models for each arm and exposure
        self.init_correction_models(spectra, rv_bounds=rv_bounds)

        state = self.prepare_fit(
                spectra,
                rv_0=rv_0, rv_bounds=rv_bounds, rv_prior=rv_prior, rv_step=rv_step, rv_fixed=rv_fixed,
                params_0=params_0, params_bounds=params_bounds,
                params_priors=params_priors, params_steps=params_steps,
                params_fixed=params_fixed)

        if state.bounds is not None and \
            np.any((np.transpose(state.x_0) < state.bounds[..., 0]) |
                   (state.bounds[..., 1] < np.transpose(state.x_0))):
            
            raise Exception("Initial state is outside bounds.")

        # Group atmospheric parameters and RV into separate Gibbs steps
        gibbs_blocks = [[ i for i in range(len(state.params_free))], [len(state.params_free)]]
        
        mcmc = MCMC(state.log_L_fun, step_size=state.steps,
                    gibbs_blocks=gibbs_blocks,
                    walkers=walkers, gamma=0.99)

        res_x, res_lp, accept_rate = mcmc.sample(state.x_0,
                                                 burnin, samples, gamma=gamma)
                           
        # Extract results into separate parameters + RV
        # Unpack_params expects that the leading dimension is: # of params + 1 (RV)
        # Result shape: (variables, samples, walkers)
        # Rotate dimensions so that variables are last
        params, rv = state.unpack_params(res_x.transpose(tuple(range(res_x.ndim)[1:]) + (0,)))
        
        # Append fixed params to dictionary with the same shape as RV
        for k in state.params_fixed:
            params[k] = np.full_like(rv, state.params_fixed[k])

        # TODO: assign flags, calculate moments, etc.
        
        return ModelGridTempFitResults(rv_mcmc=rv, rv_flags=state.rv_flags,
                                       params_mcmc=params, params_flags=state.params_flags,
                                       log_L_mcmc=res_lp,
                                       accept_rate=accept_rate,
                                       flags=state.flags)
    
