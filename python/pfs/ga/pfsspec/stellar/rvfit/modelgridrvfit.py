import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.optimize import minimize_scalar
import numdifftools as nd

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
            self.params_fixed = None         # Params not fitted
            self.params_bounds = None        # Template parameter bounds
        else:
            self.template_grids = orig.template_grids

            self.params_0 = orig.params_0
            self.params_bounds = orig.params_bounds

    def get_templates(self, spectra, **params):
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

    def get_normalization(self, spectra, templates=None, params_0=None):
        # Calculate a normalization factor for the spectra, as well as
        # the templates assuming an RV=0 and params_0 for the template.
        # This is just a factor to bring spectra to the same scale and avoid
        # very large numbers in the Fisher matrix

        params_0 = params_0 if params_0 is not None else self.params_0
        templates = templates if templates is not None else self.get_templates(spectra, **params_0)

        return super().get_normalization(spectra, templates)

    def calculate_rv_error(self, spectra, rv_0, params_0, rv_step, params_step):
        # What is the correct formula here?
        raise NotImplementedError()

    def calculate_fisher(self, spectra, rv_0, params_0, rv_step=1.0, params_step=None):
        """
        Calculate the full Fisher matrix using numerical differentiation around `rv_0`
        and `params_0`
        """

        # We need to pack and unpack phi and chi because numdifftools don't
        # properly handle multivariate functions of higher dimensions.

        # We'll have to pass the parameters to the functions in the same order,
        # keep track of them now here.
        param_keys = list(params_0.keys())

        if not self.flux_corr:
            def pack_phi_chi(phi, chi):
                return np.array([phi, chi])

            def unpack_phi_chi(phi_chi, size):
                return phi_chi[0], phi_chi[1]
        else:
            def pack_phi_chi(phi, chi):
                return np.concatenate([phi.flatten(), chi.flatten()])

            def unpack_phi_chi(phi_chi, size):
                var_shape = phi_chi.shape[1:]   # Variables
                return phi_chi[:size], phi_chi[size:].reshape((size, size) + var_shape)

        def pack_params_rv(params, rv):
            params_rv = np.array([params[k] for i, k in enumerate(param_keys) if k not in self.params_fixed] + [rv])
            params_fixed = {k: params[k] for i, k in enumerate(param_keys) if k in self.params_fixed}
            return params_rv, params_fixed

        def unpack_params_rv(params_rv, **params_fixed):
            params = {k: params_rv[i] for i, k in enumerate(param_keys) if k not in self.params_fixed}
            params.update(params_fixed)
            rv = params_rv[-1]
            return params, rv

        # To work with numdifftools, we need to pack template parameters and rv
        # into a numpy array
        def phi_chi(params_rv, **params_fixed):
            params, rv = unpack_params_rv(params_rv, **params_fixed)
            templates, _ = self.get_templates(spectra, **params)
            phi, chi = self.eval_phi_chi(spectra, templates, rv)
            return pack_phi_chi(phi, chi)

        # Calculate a_0
        temp_0, _ = self.get_templates(spectra, **params_0)
        phi_0, chi_0 = self.eval_phi_chi(spectra, temp_0, rv_0)
        a_0 = self.eval_a(phi_0, chi_0)

        # First and second derivatives of the matrix elements by the
        # template parameters and RV

        
        params_rv_0, params_fixed_0 = pack_params_rv(params_0, rv_0)

        ## TODO: add step size (or take 1% or similar if not defined)
        step = 0.01 * params_rv_0

        d_phi_chi = nd.Jacobian(phi_chi, step=step)
        dd_phi_chi = nd.Hessian(phi_chi, step=step)

        d_phi_0, d_chi_0 = unpack_phi_chi(d_phi_chi(params_rv_0, **params_fixed_0), phi_0.size)

        ## TODO: a hesse-mátrix, úgy néz ki, csak skalárokra megy most
        ##       ez bazi sok számolás lenne

        dd_phi_0, dd_chi_0 = unpack_phi_chi(dd_phi_chi(params_rv_0, **params_fixed_0), phi_0.size)

        pass

    def calculate_F(self, spectra, rv_0, rv_bounds, rv_step, params_0, params_bounds, params_step, params_free, params_fixed):
        """
        Calculate the Fisher matrix around (rv_0, params_0), assumed to be the optimum
        """

        templates, _ = self.get_templates(spectra, **params_0)
        phi_0, chi_0 = self.eval_phi_chi(spectra, templates, rv_0)

        # Calculate the gradient of phi in (rv_0, params_0)
        def phichi(x):
            rv, params = self.array_to_params(x, params_free, params_fixed)
            templates, missing = self.get_templates(spectra, **params)
            if missing:
                return None
            else:
                phi, chi = self.eval_phi_chi(spectra, templates, rv_0)
                return np.vstack([phi, chi]) #.flatten()
        
        # TODO: figure out step automatically
        dphichi = Jacobian(phichi, step=[1, 0.01, 10, 0.01])      # RV, [M/H], T_eff, log_g
        dpc = dphichi(self.params_to_array(rv_0, params_free, **params_0))
        dpc = dpc.reshape((chi_0.shape[0] + 1, chi_0.shape[1], -1))
        dphi = dpc[0]
        dchi = dpc[1:]

        ddphichi = Hessian(phichi, step=[1, 0.01, 10, 0.01])      # RV, [M/H], T_eff, log_g
        ddpc = ddphichi(self.params_to_array(rv_0, params_free, **params_0))
        ddpc = ddpc.reshape((chi_0.shape[0] + 1, chi_0.shape[1], -1))

        # Calculate the Hessian of chi around rv_0 and params_0
        return



    def array_to_params(self, x, params_free, params_fixed):
        rv = x[0]
        params = {}
        for i, p in enumerate(params_free):
            params[p] = x[1 + i]
        for p in params_fixed:
            params[p] = params_fixed[p]

        return rv, params

    def params_to_array(self, rv, params_free, **params):
        x = np.array([ rv ] + [ params[k] for k in params_free ])
        return x
            
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
            rv, params = self.array_to_params(x, params_free, params_fixed)
            templates, missing = self.get_templates(spectra, **params)

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
            rv = out.x[0]
            params = {}
            for i, p in enumerate(params_free):
                params[p] = out.x[1 + i]
            for p in params_fixed:
                params[p] = params_fixed[p]

            # TODO: Error from fisher matrix
            # TODO: Continuum params and errors

            self.calculate_F(spectra, rv, rv_bounds, None, params, params_bounds, None, params_free, params_fixed)

            return rv, None, params, None
        else:
            raise Exception(f"Could not fit RV using `{method}`")

        