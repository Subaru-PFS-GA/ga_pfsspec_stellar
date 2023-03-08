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
            temp = grid.interpolate_model_linear(**params)

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

    def eval_F(self, spectra, rv_0, params_0, params_fixed=None, step=None, mode='full', method='hessian'):
        # Evaluate the Fisher matrix around the provided rv_0 and params_0
        # values. The corresponding a_0 best fit flux correction will be
        # evaluated at the optimum. The Hessian will be calculated wrt either
        # RV only, or rv and the template parameters or rv, the template parameters
        # and the flux correction coefficients. Alternatively, the covariance
        # matrix will be determined using MCMC.

        # Collect fixed template parameters
        params_fixed = params_fixed if params_fixed is not None else self.params_fixed

        # Enumerate grid parameters in order and filter out values that should be kept fixed
        # Order is important because arguments are passed by position to the cost function
        for k in spectra:
            grid = self.template_grids[k]
            params_free = [p for i, p, ax in grid.enumerate_axes() if p not in params_fixed]
            break

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
                    return np.inf
                else:
                    log_L, phi, chi = self.calculate_log_L(spectra, templates, rv, a=a)
                    return np.asscalar(log_L)
                
            # Calculate a_0
            templates, missing = self.get_templates(spectra, params_0)
            phi_0, chi_0 = self.eval_phi_chi(spectra, templates, rv_0)
            a_0 = self.eval_a(phi_0, chi_0)
            x_0 = pack_params(a_0, params_0, rv_0)
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
                    return np.inf
                else:
                    log_L, _, _ = self.calculate_log_L(spectra, templates, rv)
                    return np.asscalar(log_L)
                
            x_0 = pack_params(params_0, rv_0)
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
                    return np.inf
                else:
                    log_L, _, _ = self.calculate_log_L(spectra, templates, rv)
                    return np.asscalar(log_L)
                
            x_0 = np.atleast_1d(rv_0)
        else:
            raise NotImplementedError()

        return self.eval_F_dispatch(x_0, log_L, step, method)

    def calculate_F(self, spectra, rv_0, params_0, params_fixed=None, step=None, mode='full', method='hessian'):
        # Calculate the Fisher matrix using different methods

        return self.eval_F(spectra, rv_0, params_0, params_fixed=params_fixed, step=step, mode=mode, method=method)

        # TODO: DELETE
        # if method == 'full_hessian':
        #     return self.eval_F_full_hessian(spectra, rv_0, params_0, params_fixed=params_fixed, step=step)
        # elif method == 'rv_params_hessian':
        #     return self.eval_F_rv_params_hessian(spectra, rv_0, params_0, params_fixed=params_fixed, step=step)
        # elif method == 'rv_hessian':
        #     return self.eval_F_rv_hessian(spectra, rv_0, params_0, params_fixed=params_fixed, step=step)
        # elif method == 'full_emcee':
        #     return self.eval_F_full_emcee(spectra, rv_0, params_0, params_fixed=params_fixed, step=step)
        # elif method == 'rv_sampling':
        #     return self.eval_F_rv_sampling(spectra, rv_0, params_0, params_fixed=params_fixed, step=step)
        # else:
        #     raise NotImplementedError()

    # TODO: merge with base-class
    # def calculate_fisher(self, spectra, rv_0, params_0, rv_step=1.0, params_step=None):
    #     """
    #     Calculate the full Fisher matrix using numerical differentiation around `rv_0`
    #     and `params_0`
    #     """

    #     # We need to pack and unpack phi and chi because numdifftools don't
    #     # properly handle multivariate functions of higher dimensions.

    #     # We'll have to pass the parameters to the functions in the same order,
    #     # keep track of them now here.
    #     param_keys = list(params_0.keys())

    #     if not self.use_flux_corr:
    #         def pack_phi_chi(phi, chi):
    #             return np.array([phi, chi])

    #         def unpack_phi_chi(phi_chi, size):
    #             return phi_chi[0], phi_chi[1]
    #     else:
    #         def pack_phi_chi(phi, chi):
    #             return np.concatenate([phi.flatten(), chi.flatten()])

    #         def unpack_phi_chi(phi_chi, size):
    #             var_shape = phi_chi.shape[1:]   # Variables
    #             return phi_chi[:size], phi_chi[size:].reshape((size, size) + var_shape)

    #     def pack_params_rv(params, rv):
    #         params_rv = np.array([params[k] for i, k in enumerate(param_keys) if k not in self.params_fixed] + [rv])
    #         params_fixed = {k: params[k] for i, k in enumerate(param_keys) if k in self.params_fixed}
    #         return params_rv, params_fixed

    #     def unpack_params_rv(params_rv, **params_fixed):
    #         params = {k: params_rv[i] for i, k in enumerate(param_keys) if k not in self.params_fixed}
    #         params.update(params_fixed)
    #         rv = params_rv[-1]
    #         return params, rv

    #     # To work with numdifftools, we need to pack template parameters and rv
    #     # into a numpy array
    #     def phi_chi(params_rv, **params_fixed):
    #         params, rv = unpack_params_rv(params_rv, **params_fixed)
    #         templates, _ = self.get_templates(spectra, **params)
    #         phi, chi = self.eval_phi_chi(spectra, templates, rv)
    #         return pack_phi_chi(phi, chi)

    #     # Calculate a_0
    #     temp_0, _ = self.get_templates(spectra, **params_0)
    #     phi_0, chi_0 = self.eval_phi_chi(spectra, temp_0, rv_0)
    #     a_0 = self.eval_a(phi_0, chi_0)

    #     # First and second derivatives of the matrix elements by the
    #     # template parameters and RV

    #     params_rv_0, params_fixed_0 = pack_params_rv(params_0, rv_0)

    #     ## TODO: add step size (or take 1% or similar if not defined)
    #     step = 0.01 * params_rv_0

    #     d_phi_chi = nd.Jacobian(phi_chi, step=step)
    #     dd_phi_chi = nd.Hessian(phi_chi, step=step)

    #     d_phi_0, d_chi_0 = unpack_phi_chi(d_phi_chi(params_rv_0, **params_fixed_0), phi_0.size)
    #     dd_phi_0, dd_chi_0 = unpack_phi_chi(dd_phi_chi(params_rv_0, **params_fixed_0), phi_0.size)

    #     # Compose the matrix for the Hessian of L
    #     # Size is # of flux correction components + # of free parames including RV
    #     if not self.use_flux_corr:
    #         s1, s2 = 1, params_rv_0.shape[0]
    #         F = np.empty((s1 + s2, s1 + s2), dtype=phi_0.dtype)

    #         F[:s1, :s1] = -chi_0
    #         F[:s1, s1:] = d_phi_0 - a_0 * d_chi_0
    #         F[s1:, :s1] = F[:s1, s1:].T
    #         F[s1:, s1:] = a_0 * dd_phi_0 - 0.5 * a_0**2 * dd_chi_0
    #     else:
    #         s1, s2 = chi_0.shape[0], params_rv_0.shape[0]
    #         F = np.empty((s1 + s2, s1 + s2), dtype=phi_0.dtype)

    #         F[:s1, :s1] = -chi_0
    #         F[:s1, s1:] = d_phi_0 - np.einsum('ijk,j->ik', d_chi_0, a_0)
    #         F[s1:, :s1] = F[:s1, s1:].T
    #         F[s1:, s1:] = np.einsum('i,ikl->kl', a_0, dd_phi_0) - 0.5 * np.einsum('i,ijkl,j->kl', a_0, dd_chi_0, a_0)

    #     return F

    # TODO: merge with base-class
    # def calculate_F(self, spectra, rv_0, rv_bounds, rv_step, params_0, params_bounds, params_step, params_free, params_fixed):
    #     """
    #     Calculate the Fisher matrix around (rv_0, params_0), assumed to be the optimum
    #     """

    #     templates, _ = self.get_templates(spectra, **params_0)
    #     phi_0, chi_0 = self.eval_phi_chi(spectra, templates, rv_0)

    #     # Calculate the gradient of phi in (rv_0, params_0)
    #     def phichi(x):
    #         rv, params = self.pack_params(x, params_free, params_fixed)
    #         templates, missing = self.get_templates(spectra, **params)
    #         if missing:
    #             return None
    #         else:
    #             phi, chi = self.eval_phi_chi(spectra, templates, rv_0)
    #             return np.vstack([phi, chi]) #.flatten()
        
    #     # TODO: figure out step automatically
    #     dphichi = Jacobian(phichi, step=[1, 0.01, 10, 0.01])      # RV, [M/H], T_eff, log_g
    #     dpc = dphichi(self.unpack_params(rv_0, params_free, **params_0))
    #     dpc = dpc.reshape((chi_0.shape[0] + 1, chi_0.shape[1], -1))
    #     dphi = dpc[0]
    #     dchi = dpc[1:]

    #     ddphichi = Hessian(phichi, step=[1, 0.01, 10, 0.01])      # RV, [M/H], T_eff, log_g
    #     ddpc = ddphichi(self.unpack_params(rv_0, params_free, **params_0))
    #     ddpc = ddpc.reshape((chi_0.shape[0] + 1, chi_0.shape[1], -1))

    #     # Calculate the Hessian of chi around rv_0 and params_0
    #     return

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
        params_bounds = params_bounds if params_bounds is not None else self.params_bounds
        params_fixed = params_fixed if params_fixed is not None else self.params_fixed

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

        def pack_params(rv, **params):
            x = np.array([ rv ] + [ params[k] for k in params_free ])
            return x
        
        def pack_bounds(rv_bounds, params_bounds):
            bounds = [ rv_bounds ] 
            if params_bounds is not None:
                bounds += [ params_bounds[k] if k in params_bounds else None for k in params_free ]
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
        
        # Calculate the error from the Fisher matrix
        _, C = self.eval_F(spectra, rv_fit, params_fit, params_fixed=params_fixed, mode='params_rv', method='hessian')
        err = np.sqrt(np.diag(C)) # sigma
        params_err = {}
        for i, p in enumerate(params_free):
            params_err[p] = err[i]
        rv_err = err[-1]

        return rv_fit, rv_err, params_fit, params_err