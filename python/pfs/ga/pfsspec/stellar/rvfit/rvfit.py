import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.optimize import minimize_scalar
from scipy.ndimage import binary_dilation
from scipy.interpolate import interp1d
import numdifftools as nd
from collections.abc import Iterable
import emcee

from pfs.ga.pfsspec.core import Physics
from pfs.ga.pfsspec.core.obsmod.resampling import FluxConservingResampler

class RVFitTrace():
    """
    Implements call-back function to profile and debug RV fitting. Allows for
    generating plots of intermediate steps.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.process_spectrum_count = 0
        self.process_template_count = 0
        self.resample_template_count = 0
        self.template_cache_hit_count = 0
        self.template_cache_miss_count = 0
        self.eval_phi_chi_count = 0
        self.eval_log_L_count = 0
        self.eval_log_L_a_count = 0

    def on_process_spectrum(self, spectrum, processed_spectrum):
        self.process_spectrum_count += 1

    def on_process_template(self, rv, template, processed_template):
        self.process_template_count += 1

    def on_resample_template(self, rv, spectrum, template, processed_template):
        self.resample_template_count += 1

    def on_template_cache_hit(self, template, rv_q, rv):
        self.template_cache_hit_count += 1
    
    def on_template_cache_miss(self, template, rv_q, rv):
        self.template_cache_miss_count += 1

    def on_eval_flux_corr_basis(self, spectra, basis):
        pass

    def on_eval_phi_chi(self, rv, spectra, templates, bases, sigma2, weights, masks, log_L, phi, chi):
        self.eval_phi_chi_count += 1

    def on_eval_log_L(self, phi, chi, log_L):
        self.eval_log_L_count += 1

    def on_eval_log_L_a(self, phi, chi, a, log_L):
        self.eval_log_L_a_count += 1

    def on_guess_rv(self, rv, log_L, fit, function, pp, pcov):
        pass

    def on_fit_rv(self, rv, spectra, templates):
        pass

class RVFit():
    """
    A basic RV estimation based on template fitting using a non-linear maximum likelihood or
    maximum significance method. Templates are optionally convolved with the instrumental LSF.
    The class supports various methods to determine the uncertainty of estimated parameters.
    """

    def __init__(self, trace=None, orig=None):
        
        if not isinstance(orig, RVFit):
            self.trace = trace              # Collect debug info

            self.template_psf = None        # Dict of psf to downgrade templates with
            self.template_resampler = FluxConservingResampler()  # Resample template to instrument pixels
            self.template_cache_resolution = 50  # Cache templates with this resolution in RV
            self.cache_templates = True    # Cache PSF-convolved templates

            self.rv_0 = None                # RV initial guess
            self.rv_bounds = None           # Find RV between these bounds

            self.use_flux_corr = False      # Use flux correction. Scalar if no basis is provided, otherwise linear combination of basis functions
            self.flux_corr_basis = None     # Callable that provides the basis functions for flux correction / continuum fitting.

            self.spec_norm = None           # Spectrum (observation) normalization factor
            self.temp_norm = None           # Template normalization factor

            self.use_mask = True            # Use mask from spectrum, if available
            self.use_error = True           # Use flux error from spectrum, if available
            self.use_weight = True          # Use weight from template, if available
        else:
            self.trace = orig.trace

            self.template_psf = orig.template_psf
            self.template_resampler = orig.template_resampler
            self.template_cache_resolution = orig.template_cache_resolution
            self.cache_templates = orig.cache_templates

            self.rv_0 = orig.rv_0
            self.rv_bounds = orig.rv_bounds

            self.use_flux_corr = orig.use_flux_corr
            self.flux_corr_basis = orig.flux_corr_basis

            self.spec_norm = orig.spec_norm
            self.temp_norm = orig.temp_norm

            self.use_mask = orig.use_mask
            self.use_weight = orig.use_weight

        self.reset()

    def reset(self):
        self.template_cache = {}
        self.flux_corr_basis_cache = None
        self.flux_corr_basis_size = None

    def get_normalization(self, spectra, templates, rv_0=None):
        # Calculate a normalization factor for the spectra, as well as
        # the templates assuming an RV=0 from the median flux. This is
        # just a factor to bring spectra to the same scale and avoid
        # very large numbers in the Fisher matrix

        rv_0 = rv_0 if rv_0 is not None else self.rv_0

        s = []
        t = []
        for k in spectra:
            for spec in spectra[k] if isinstance(spectra[k], list) else [spectra[k]]:
                spec = self.process_spectrum(spec)
                s.append(spec.flux)
            
            if self.template_psf is not None:
                psf = self.template_psf[k]
            else:
                psf = None
            
            if rv_0 is not None:
                rv = rv_0
            else:
                rv = 0.0

            temp = self.process_template(templates[k], spec, rv, psf)
            t.append(temp.flux)

        spec_flux = np.concatenate(s)
        temp_flux = np.concatenate(t)

        return np.median(spec_flux), np.median(temp_flux)

    def process_spectrum(self, spectrum):
        # Preprocess the spectrum to calculate the likelihood. This step
        # currently consist only of normalizing the flux with a factor.

        spec = spectrum.copy()
        if self.spec_norm is not None:
            spec.multiply(1.0 / self.spec_norm)

        if self.trace is not None:
            self.trace.on_process_spectrum(spectrum, spec)

        return spec

    def process_template(self, template, spectrum, rv, psf=None, diff=False):
        # Preprocess the template to match the observation. This step
        # involves shifting the high resolution template the a given RV,
        # convolving it with the PSF, still at high resolution, and normalizing
        # by a factor.
        # To improve performance, pre-processed templates are cached at quantized 
        # values of RV and reused when the requested RV is close to one in the
        # cache. This is fine as long as the PFS changes slowly with wavelength
        # which is most often the case. Still, templates sometimes get shifted
        # significantly when the target's RV is large so a single precomputed
        # template is not enough.
        # The template is always kept at high resolution during transformations
        # until it's resampled to the spectrum pixels using a flux-conserving
        # resampler.

        def process_template_impl(rv):
            # 1. Make a copy, not in-place update
            t = template.copy()

            # 2. Shift template to the desired RV
            t.set_rv(rv)

            # 3. If a PSF if provided, convolve down template to the instruments
            #    resolution. Otherwise assume the template is already convolved.
            if psf is not None:
                t.convolve_psf(psf)

            # 4. Normalize
            if self.temp_norm is not None:
                t.multiply(1.0 / self.temp_norm)

            if self.trace is not None:
                self.trace.on_process_template(rv, template, t)
                
            return t

        temp = None

        # If `cache_templates` is True, we can look up a template that's already been
        # convolved down with the PSF. Otherwise we just shift the template and do the
        # convolution on the fly.
        if self.cache_templates:
            # Quantize RV to the requested cache resolution
            rv_q = np.floor(rv / self.template_cache_resolution) * self.template_cache_resolution
            key = (template, rv_q)

            # Look up template in cache and shift to from quantized RV to actual RV
            if key in self.template_cache:
                temp = self.template_cache[key]
                if self.trace is not None:
                    self.trace.on_template_cache_hit(template, rv_q, rv)
            else:
                temp = process_template_impl(rv_q)
                if self.trace is not None:
                    self.trace.on_template_cache_miss(template, rv_q, rv)

            if key not in self.template_cache:
                self.template_cache[key] = temp

            # Shift from quantized RV to requested RV
            temp = temp.copy()
            temp.set_restframe()
            temp.set_redshift(Physics.vel_to_z(rv))
        else:
            # Compute convolved template from scratch
            temp = process_template_impl(rv)

        # Resample template to the binning of the observation
        temp.apply_resampler(self.template_resampler, spectrum.wave, spectrum.wave_edges)

        if self.trace is not None:
            self.trace.on_resample_template(rv, spectrum, template, temp)

        return temp
    
    def diff_template(self, template, wave=None):
        # Calculate the numerical derivative of the template and optionally resample
        # to a new wavelength grid using linear interpolation

        dfdl = np.empty_like(template.wave)
        dfdl[1:-1] = (template.flux[2:] - template.flux[:-2]) / (template.wave[2:] - template.wave[:-2])
        dfdl[0] = dfdl[1]
        dfdl[-1] = dfdl[-2]

        if wave is not None:
            ip = interp1d(template.wave, dfdl, bounds_error=False, fill_value=np.nan)
            dfdl = ip(wave)
        else:
            wave = template.wave

        return wave, dfdl
    
    def log_diff_template(self, template, wave=None):
        # Calculate the numerical log-derivative of the template and optionally resample
        # to a new wavelength grid using linear interpolation

        dfdlogl = np.empty_like(template.wave)
        dfdlogl[1:-1] = template.wave[1:-1] * (template.flux[2:] - template.flux[:-2]) / (template.wave[2:] - template.wave[:-2])
        dfdlogl[0] = dfdlogl[1]
        dfdlogl[-1] = dfdlogl[-2]

        if wave is not None:
            ip = interp1d(template.wave, dfdlogl, bounds_error=False, fill_value=np.nan)
            dfdlogl = ip(wave)
        else:
            wave = template.wave

        return wave, dfdlogl
    
    def eval_flux_corr_basis(self, spectra):
        # Evaluate the basis functions on the wavelength grid of the spectra

        basis = {}
        basis_size = None
        for k in spectra:
            basis[k] = []
            for spec in spectra[k] if isinstance(spectra[k], list) else [spectra[k]]:
                basis[k].append(self.flux_corr_basis(spec.wave))
                if basis_size is None:
                    basis_size = basis[k][-1].shape[-1]
                elif basis_size != basis[k][-1].shape[-1]:
                    raise Exception('Inconsistent basis size')

        if self.trace is not None:        
            self.trace.on_eval_flux_corr_basis(spectra, basis)
                
        return basis, basis_size

    def eval_phi_chi(self, spectra, templates, rv):
        """
        Calculate the log-likelihood of an observed spectrum for a template with RV.

        It assumes that the template is already convolved down to the instrumental
        resolution but not resampled to the instrumental bins yet.
        """

        if not isinstance(rv, np.ndarray):
            rvv = np.array([rv])
        else:
            rvv = rv.flatten()

        if not self.use_flux_corr:
            bases = None
            phi = np.zeros(rvv.shape)
            chi = np.zeros(rvv.shape)
        else:
            # Evaluate the basis for each spectrum
            if self.flux_corr_basis_cache is None:
                self.flux_corr_basis_cache, self.flux_corr_basis_size = self.eval_flux_corr_basis(spectra)
            bases, basis_size = self.flux_corr_basis_cache, self.flux_corr_basis_size            
            phi = np.zeros(rvv.shape + (basis_size,))
            chi = np.zeros(rvv.shape + (basis_size, basis_size))
                
        # For each value of rv0, sum up log_L contributions from spectrum - template pairs
        for i in range(rvv.size):
            trace_temp = {}         # Collect preprocessed templates for tracing
            trace_bases = {}         # Collect continuum basis functions for tracing
            trace_mask = {}
            trace_sigma2 = {}
            trace_weight = {}

            for k in spectra:
                trace_temp[k] = []
                trace_bases[k] = []
                trace_mask[k] = []
                trace_sigma2[k] = []
                trace_weight[k] = []

                if not isinstance(spectra[k], list):
                    specs = [spectra[k]]
                else:
                    specs = spectra[k]

                psf = self.template_psf[k] if self.template_psf is not None else None
                
                for ei in range(len(specs)):
                    spec = self.process_spectrum(specs[ei])
                    temp = self.process_template(templates[k], spec, rvv[i], psf=psf, diff=True)
                    trace_temp[k].append(temp)

                    # Determine mask
                    if self.use_mask:
                        mask = ~spec.mask if spec.mask is not None else np.s_[:]
                    else:
                        mask = np.s_[:]
                    trace_mask[k].append(mask)

                    # Flux error
                    sigma2 = spec.flux_err ** 2
                    trace_sigma2[k].append(sigma2)

                    # Weight (optional)
                    if self.use_weight and temp.weight is not None:
                        weight = temp.weight / temp.weight.sum() * temp.weight.size
                    else:
                        weight = np.ones_like(spec.flux)
                    trace_weight[k].append(weight)
                    
                    if bases is None:
                        phi[i] += np.sum(weight[mask] * spec.flux[mask] * temp.flux[mask] / sigma2[mask])
                        chi[i] += np.sum(weight[mask] * temp.flux[mask] ** 2 / sigma2[mask])
                    else:
                        basis = bases[k][ei]
                        try:
                            phi[i] += np.sum(weight[mask, None] * spec.flux[mask, None] * temp.flux[mask, None] * basis[mask, :] / sigma2[mask, None], axis=0)
                            chi[i] += np.sum(weight[mask, None, None] * (temp.flux[mask, None, None] ** 2 * np.matmul(basis[mask, :, None], basis[mask, None, :])) / sigma2[mask, None, None], axis=0)
                        except Exception as ex:
                            raise ex
                        trace_bases[k].append(basis)

            # Only trace when all arms are fitted but for each rv
            if self.trace is not None:
                # First dimension must index rv items
                log_L = self.eval_log_L(phi[np.newaxis, i], chi[np.newaxis, i])
                self.trace.on_eval_phi_chi(rvv[i], spectra, trace_temp, trace_bases, trace_sigma2, trace_weight, trace_mask, log_L[0], phi[i], chi[i])

        if not isinstance(rv, np.ndarray):
            phi = phi[0]
            chi = chi[0]
        else:
            if bases is None:
                phi = phi.reshape(rv.shape)
                chi = chi.reshape(rv.shape)
            else:
                phi = phi.reshape(rv.shape + (basis_size,))
                chi = chi.reshape(rv.shape + (basis_size, basis_size))

        return phi, chi
    
    def eval_a(self, phi, chi):
        if not self.use_flux_corr:
            a = phi / chi
        else:
            a = np.linalg.solve(chi, phi)
        return a

    def eval_nu2(self, phi, chi):
        # Importance at the optimum in the flux amplitude / correction parameters.
        if not self.use_flux_corr:
            nu2 = phi ** 2 / chi
        else:
            nu2 = np.empty(phi.shape[:-1])
            for i in np.ndindex(nu2.shape):
                nu2[i] = np.dot(phi[i], np.linalg.solve(chi[i], phi[i]))        
        return nu2

    def eval_log_L(self, phi, chi):
        # Likelihood at the optimum in the flux amplitude / correction parameters.
        log_L = 0.5 * self.eval_nu2(phi, chi)

        if self.trace is not None:
            self.trace.on_eval_log_L(phi, chi, log_L)

        return log_L        
    
    def eval_log_L_a(self, phi, chi, a):
        # Likelihood as a function of flux correction coefficients
        if not self.use_flux_corr:
            log_L = a * phi - 0.5 * a * a * chi
        else:
            #log_L = np.empty(phi.shape[:-1])
            #for i in np.ndindex(log_L.shape):
            log_L = np.squeeze(
                np.matmul(a[..., None, :], phi[..., :, None]) - 
                0.5 * np.matmul(a[..., None, :], np.matmul(chi, a[..., :, None])))

        if self.trace is not None:
            self.trace.on_eval_log_L_a(phi, chi, a, log_L)
        
        return log_L
    
    def calculate_log_L(self, spectra, templates, rv, a=None):
        # Depending on whether the flux correction coefficients `a`
        # are supplied, calculate the log likelihood at the optimum of
        # flux correction or at the specified flux correction values.
        phi, chi = self.eval_phi_chi(spectra, templates, rv)
        if a is None:
            log_L = self.eval_log_L(phi, chi)
            return log_L, phi, chi
        else:
            log_L = self.eval_log_L_a(phi, chi, a)
            return log_L, phi, chi
        
    #region Fisher matrix evaluation

    # There are multiple ways of evaluating the Fisher matrix. Functions
    # with _full_ return the full Fisher matrix including matrix elements for
    # the flux correction coefficients and rv. Functions with _rv_ return
    # the (single) matrix element corresponding to the RV only. Functions with
    # _hessian depend on the numerical evaluation of the Hessian with respect
    # to the parameters of the fitted model.

    def eval_F(self, spectra, templates, rv_0, step=None, mode='full', method='hessian'):
        # Evaluate the Fisher matrix around the provided rv_0. The corresponding
        # a_0 best fit flux correction will be evaluated at the optimum.
        # The Hessian will be calculated wrt either RV only, or rv and the
        # flux correction coefficients. Alternatively, the covariance
        # matrix will be determined using MCMC.

        if mode == 'full' or mode == 'a_rv':
            def pack_params(a, rv):
                a_rv = np.concatenate([np.atleast_1d(a), np.atleast_1d(rv)])
                return a_rv

            def unpack_params(a_rv):
                a = a_rv[:-1]
                rv = a_rv[-1]
                return a, rv

            def log_L(a_rv):
                a, rv = unpack_params(a_rv)
                log_L, _, _ = self.calculate_log_L(spectra, templates, rv, a=a)
                return np.asscalar(log_L)
            
            # Calculate a_0
            phi_0, chi_0 = self.eval_phi_chi(spectra, templates, rv_0)
            a_0 = self.eval_a(phi_0, chi_0)
            x_0 = pack_params(a_0, rv_0)
        elif mode == 'rv':
            def pack_params(rv):
                return np.atleast_1d(rv)

            def unpack_params(params):
                rv = params[0]
                return rv

            def log_L(params):
                rv = unpack_params(params)
                log_L, _, _ = self.calculate_log_L(spectra, templates, rv)
                return log_L
            
            x_0 = pack_params(rv_0)
        else:
            raise NotImplementedError()
        
        return self.eval_F_dispatch(x_0, log_L, step, method)

    def eval_F_dispatch(self, x_0, log_L, step, method):
        if method == 'hessian':
            return self.eval_F_hessian(x_0, log_L, step)
        elif method == 'emcee':
            return self.eval_F_emcee(x_0, log_L, step)
        elif method == 'sampling':
            return self.eval_F_sampling(x_0, log_L, step)
        else:
            raise NotImplementedError()

    def eval_F_hessian(self, x_0, log_L, step):
        # Evaluate the Fisher matrix by calculating the Hessian numerically

        # Default step size is 1% of optimum values
        if step is None:
            step = 0.01 * x_0

        dd_log_L = nd.Hessian(log_L, step=step)
        dd_log_L_0 = dd_log_L(x_0)

        return dd_log_L_0, np.linalg.inv(-dd_log_L_0)

    def eval_F_emcee(self, x_0, log_L, step):
        # Evaluate the Fisher matrix by MC sampling around the optimum
        ndim = x_0.size
        nwalkers = 2 * x_0.size + 1
        burnin = 100
        samples = 100
        p0 = (1 + np.random.uniform(size=(nwalkers, ndim)) * 0.1) * x_0
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_L)
        state = sampler.run_mcmc(p0, burnin, skip_initial_state_check=True)
        sampler.reset()
        sampler.run_mcmc(state, samples, skip_initial_state_check=True)
        x = sampler.get_chain(flat=True)

        # This is already the covanriance, we need its inverse here
        C = np.cov(x.T)
        return -np.linalg.inv(C), C
    
    def eval_F_sampling(self, x_0, log_L, step):
        # Sample a bunch of RVs around the optimum and fit a parabola
        # to obtain the error of RV            

        # TODO: this only works for a single parameter only, can we
        #       generalize to multiple parameters?

        if x_0.size != 1:
            # Currently fitting 1d parabola only
            raise NotImplementedError()

        # Default step size is 1% of optimum values
        if step is None:
            step = 0.01 * x_0

        rvv = np.random.uniform(x_0 - step, x_0 + step, size=1000)
        ll = np.empty_like(rvv)
        for ix, rv in np.ndenumerate(rvv):
            ll[ix] = log_L(np.atleast_1d(rv))
        p = np.polyfit(rvv, ll, 2)
        return np.array([[2.0 * p[0]]]), np.array([[-0.5 / p[0]]])

    def eval_F_full_phi_chi(self, spectra, templates, rv_0, step=None):
        # Evaluate the Fisher matrix from the first and second derivatives of
        # phi and chi around rv_0, based on the flux correction formulate

        # We need to pack and unpack phi and chi because numdifftools don't
        # properly handle multivariate functions of higher dimensions.

        if not self.use_flux_corr:
            def pack_phi_chi(phi, chi):
                return np.array([phi, chi])

            def unpack_phi_chi(phi_chi, size):
                return phi_chi[0], phi_chi[1]
        else:
            def pack_phi_chi(phi, chi):
                return np.concatenate([phi.flatten(), chi.flatten()])

            def unpack_phi_chi(phi_chi, size):
                return phi_chi[:size], phi_chi[size:].reshape((size, size))

        def phi_chi(rv):
            phi, chi = self.eval_phi_chi(spectra, templates, rv)
            return pack_phi_chi(phi, chi)

        # Calculate a_0
        phi_0, chi_0 = self.eval_phi_chi(spectra, templates, rv_0)
        a_0 = self.eval_a(phi_0, chi_0)

        # First and second derivatives of the matrix elements by RV
        d_phi_chi = nd.Derivative(phi_chi, step=step)
        dd_phi_chi = nd.Derivative(phi_chi, step=step, n=2)

        d_phi_0, d_chi_0 = unpack_phi_chi(d_phi_chi(rv_0), phi_0.size)
        dd_phi_0, dd_chi_0 = unpack_phi_chi(dd_phi_chi(rv_0), phi_0.size)

        if not self.use_flux_corr:
            # TODO: use special calculations from Alex
            F = np.empty((2, 2), dtype=phi_0.dtype)
            F[0, 0] = chi_0
            F[1, 0] = F[0, 1] = d_phi_0   # TODO: is this correct here?
            F[1, 1] = -a_0 * dd_phi_0 + 0.5 * a_0**2 * dd_chi_0
        else:
            # Assemble the Fisher matrix
            ndf = phi_0.size
            F = np.empty((ndf + 1, ndf + 1), dtype=phi_0.dtype)

            F[:ndf, :ndf] = chi_0
            F[-1, :-1] = F[:-1, -1] = -d_phi_0 + np.matmul(d_chi_0, a_0)
            F[-1, -1] = - np.dot(a_0, dd_phi_0) + 0.5 * np.dot(a_0, np.matmul(dd_chi_0, a_0))

        return F, np.linalg.inv(F)
    
    def eval_F_full_alex(self, spectra, templates, rv, step=None):
        # Calculate the Fisher matrix numerically from a local finite difference
        # around `rv` in steps of `step`.

        # TODO: what if we are fitting the continuum as well?

        if not isinstance(spectra, Iterable):
            spectra = [ spectra ]
        if not isinstance(templates, Iterable):
            templates = [ templates ]

        if step is None:
            step = 0.001 * rv

        # TODO: Verify math in sum over spectra - templates

        psi00 = 0.0
        psi01 = 0.0
        psi11 = 0.0
        psi02 = 0.0
        phi02 = 0.0
        phi00 = 0.0

        for k in spectra:
            temp = templates[k]
            for spec in spectra[k] if isinstance(spectra[k], list) else [spectra[k]]:
                spec = self.process_spectrum(spec)

                psf = self.template_psf[k] if self.template_psf is not None else None
                temp0 = self.process_template(temp, spec, rv, psf=psf)
                temp1 = self.process_template(temp, spec, rv + step, psf=psf)
                temp2 = self.process_template(temp, spec, rv - step, psf=psf)

                # Calculate the centered diffence of the flux
                d1  = 0.5 * (temp2.flux - temp1.flux) / step
                d2  = (temp1.flux + temp2.flux - 2 * temp0.flux) / step

                # Build the different terms
                s2 = spec.flux_err ** 2
                psi00 += np.sum(temp0.flux * temp0.flux / s2)
                psi01 += np.sum(temp0.flux * d1 / s2)
                psi11 += np.sum(d1 * d1 / s2)
                psi02 += np.sum(temp0.flux * d2 / s2)
                phi02 += np.sum(spec.flux * d2 / s2)
                phi00 += np.sum(spec.flux * temp0.flux / s2)
        
        chi   = psi00 
        a0    = phi00 / psi00
        
        F00 = psi00
        F01 = a0 * psi01
        F11 = a0 ** 2 * (psi02 - phi02 / a0 + psi11)

        F  = np.array([[F00, F01], [F01, F11]])

        return -F, np.linalg.inv(F)
       
    def calculate_F(self, spectra, templates, rv_0, step=None, mode='full', method='hessian'):
        # Calculate the Fisher matrix using different methods

        if mode == 'full' and method == 'phi_chi':
            return self.eval_F_full_phi_chi(spectra, templates, rv_0, step=step)
        elif mode == 'full' and method == 'alex':
            return self.eval_F_full_alex(spectra, templates, rv_0, step=step)
        else:
            return self.eval_F(spectra, templates, rv_0, step=step, mode=mode, method=method)

    def eval_rv_error_alex(self, spectra, templates, rv_0, step=1.0):
        """
        Calculate the RV fitting error around the best fit value using
        numerical differentiation of the matrix elements of phi and chi.
        """

        def nu(rv):
            phi, chi = self.eval_phi_chi(spectra, templates, rv)
            return np.sqrt(self.eval_nu2(phi, chi))

        # Second derivative by RV
        dd_nu = nd.Derivative(nu, step=step, n=2)

        nu_0 = nu(rv_0)
        dd_nu_0 = dd_nu(rv_0)

        return -1.0 / (nu_0 * dd_nu_0)
    
    #endregion
    
    @staticmethod
    def lorentz(x, a, b, c, d):
        return a / (1 + (x - b) ** 2 / c ** 2) + d

    def fit_lorentz(self, rv, y0):
        """
        Fit a Lorentz function to the log-likelihood to have a good initial guess for RV.
        """

        # Guess initial values from y0
        p0 = [
            np.max(y0),
            rv[np.argmax(y0)],
            0.5 * (rv[-1] - rv[0]),
            y0.min() + 0.5 * (y0.max() - y0.min())
        ]

        # Bounds
        bb = [
            (
                0,
                rv[0],
                0.2 * (rv[-1] - rv[0]),
                y0.min()
            ),
            (
                5 * np.max(y0),
                rv[-1],
                5.0 * (rv[-1] - rv[0]),
                y0.min() + 4 * (y0.max() - y0.min())
            )
        ]
        
        pp, pcov = curve_fit(self.lorentz, rv, y0, p0=p0, bounds=bb)

        return pp, pcov

    def guess_rv(self, spectra, templates, rv_bounds=(-500, 500), rv_steps=31):
        """
        Given a spectrum and a template, make a good initial guess for RV where a minimization
        algorithm can be started from.
        """

        rv = np.linspace(*rv_bounds, rv_steps)
        log_L, phi, chi = self.calculate_log_L(spectra, templates, rv)

        pp, pcov = self.fit_lorentz(rv, log_L)

        if self.trace is not None:
            self.trace.on_guess_rv(rv, log_L, self.lorentz(rv, *pp), 'lorentz', pp, pcov)

        return pp[1]

    def fit_rv(self, spectra, templates, rv_0=None, rv_bounds=(-500, 500), guess_rv_steps=31, method='bounded'):
        """
        Given a set of spectra and templates, find the best fit RV by maximizing the log likelihood.
        Spectra are assumed to be of the same object in different wavelength ranges.

        If no initial guess is provided, rv0 is determined automatically.
        """

        assert isinstance(spectra, dict)
        assert isinstance(templates, dict)

        rv_0 = rv_0 if rv_0 is not None else self.rv_0
        rv_bounds = rv_bounds if rv_bounds is not None else self.rv_bounds

        # No initial RV estimate provided, try to guess it and save it for later

        # if rv_0 is None:
        #     rv_0 = self.guess_rv(spectra, templates, rv_bounds=rv_bounds, rv_steps=guess_rv_steps)
        #     self.rv_0 = rv_0

        # Run optimization for log_L

        def llh(rv):
            log_L, phi, chi = self.calculate_log_L(spectra, templates, rv)
            return -log_L

        # Multivariate method
        #out = minimize(llh, [rv0], method=method)
        
        # Univariate method
        # NOTE: scipy.optimize is sensitive to type of args
        rv_bounds = tuple(float(b) for b in rv_bounds)
        bracket = None

        # if rv_bounds is not None and rv_bounds[0] < rv_0 and rv_0 < rv_bounds[1]:
        #     bracket = (rv_bounds[0], float(rv_0), rv_bounds[1])
        # else:
        #     bracket = None
            
        try:
            if method == 'grid':
                out = RVFit.minimize_gridsearch(llh, bounds=rv_bounds)
            else:
                out = minimize_scalar(llh, bracket=bracket, bounds=rv_bounds, method=method)
        except Exception as ex:
            raise ex

        if out.success:
            rv_fit = out.x

            # If tracing, evaluate the template at the best fit RV.
            # TODO: can we cache this for better performance?
            if self.trace is not None:
                tt = {}
                for k in spectra:
                    for spec in spectra[k] if isinstance(spectra[k], list) else [spectra[k]]:
                        temp = templates[k]
                        psf = self.template_psf[k] if self.template_psf is not None else None
                    
                        t = self.process_template(temp, spec, rv_fit, psf=psf)
                        tt[k] = t
                self.trace.on_fit_rv(rv_fit, spectra, tt)
        else:
            raise Exception(f"Could not fit RV using `{method}`")

        # Calculate the error from the Fisher matrix
        _, C = self.eval_F(spectra, templates, rv_fit, mode='rv', method='hessian')
        rv_err = np.sqrt(C[-1, -1]) # sigma

        return rv_fit, rv_err
    
    @staticmethod
    def minimize_gridsearch(fun, bounds, nstep=100):
        # Find minimum with direct grid search. For testing only

        x = np.linspace(bounds[0], bounds[1], nstep)
        y = fun(x)
        mi = np.argmin(y)
        
        class Result(): pass
        out = Result()
        out.success = True
        out.x = x[mi]

        return out

    def calculate_rv_bouchy(self, spectra, templates, rv_0):
        # Calculate a delta V correction by the method of Bouchy (2001)
        # TODO: we might need the flux correction coefficients here

        phi_0, chi_0 = self.eval_phi_chi(spectra, templates, rv_0)
        a_0 = self.eval_a(phi_0, chi_0)

        nom = 0
        den = 0
        sumw = 0

        for k in spectra:
            if not isinstance(spectra[k], list):
                specs = [spectra[k]]
            else:
                specs = spectra[k]

            psf = self.template_psf[k] if self.template_psf is not None else None
            
            for ei in range(len(specs)):
                spec = self.process_spectrum(specs[ei])
                temp = self.process_template(templates[k], spec, rv_0, psf=psf, diff=True)

                if self.use_mask:
                    mask = ~spec.mask if spec.mask is not None else np.s_[:]
                else:
                    mask = np.s_[:]

                if isinstance(mask, np.ndarray):
                    # We are going to calculate the central difference of flux for
                    # each pixel, so dilate the mask here by 1 pixel in both directions
                    mask = binary_dilation(mask)

                if self.use_flux_corr:
                    # Evaluate the correction function
                    basis = self.flux_corr_basis(spec.wave)
                    corr = np.dot(basis, a_0)
                else:
                    corr = a_0

                # Eq. 3
                _, dAdl = self.diff_template(temp)
                dV = (spec.flux - corr * temp.flux) / temp.wave / (corr * dAdl)

                # Eq. 8
                w = temp.wave**2 * (corr * dAdl)**2 / spec.flux_err**2

                # Eq. 4 nominator and denominator
                nom += np.sum(dV[mask] * w[mask])
                den += np.sum(w[mask])

                # Eq. 10 denominator
                sumw += np.sum(w[mask])

            # TODO: add trace function call

            # Eq. 9 and Eq. 10
            return 1e-3 * Physics.c * nom / den, 1e-3 * Physics.c / np.sqrt(sumw)