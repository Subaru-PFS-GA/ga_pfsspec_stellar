import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.optimize import minimize_scalar
import numdifftools as nd
from collections.abc import Iterable

from pfs.ga.pfsspec.core import Physics
from pfs.ga.pfsspec.core.obsmod.resampling import FluxConservingResampler

class RVFitTrace():
    def on_guess_rv(self, rv, log_L, fit, function, pp, pcov):
        pass

    def on_calculate_log_L(self, rv, spectra, templates, log_L, phi, chi):
        pass

    def on_fit_rv(self, rv, spectra, templates):
        pass

class RVFit():
    """
    A basic RV estimation based on template fitting using a non-linear maximum likelihood or
    maximum significance method. Templates are optionally convolved with the instrumental LSF.
    """

    def __init__(self, trace=None, orig=None):
        
        if not isinstance(orig, RVFit):
            self.trace = trace              # Collect debug info

            self.template_psf = None        # Dict of psf to downgrade templates with
            self.template_resampler = FluxConservingResampler()  # Resample template to instrument pixels
            self.template_cache = {}
            self.template_cache_resolution = 50  # Cache templates with this resolution in RV
            self.cache_templates = True    # Cache PSF-convolved templates

            self.rv_0 = None                # RV initial guess
            self.rv_bounds = None           # Find RV between these bounds

            self.flux_corr = False          # Flux correction on/off
            self.flux_corr_basis = None     # Callable that provides the basis functions for flux correction / continuum fitting.

            self.spec_norm = None           # Spectrum (observation) normalization factor
            self.temp_norm = None           # Template normalization factor

            self.use_mask = True            # Use mask from spectrum
        else:
            self.trace = orig.trace

            self.template_psf = orig.template_psf
            self.template_resampler = orig.template_resampler
            self.template_cache = {}
            self.template_cache_resolution = orig.template_cache_resolution
            self.cache_templates = orig.cache_templates

            self.rv_0 = orig.rv_0
            self.rv_bounds = orig.rv_bounds

            self.flux_corr = orig.flux_corr
            self.flux_corr_basis = orig.flux_corr_basis

            self.spec_norm = orig.spec_norm
            self.temp_norm = orig.temp_norm

            self.use_mask = orig.use_mask

    def get_normalization(self, spectra, templates):
        # Calculate a normalization factor for the spectra, as well as
        # the templates assuming an RV=0 from the median flux. This is
        # just a factor to bring spectra to the same scale and avoid
        # very large numbers in the Fisher matrix

        spec_flux = np.concatenate([spectra[k].flux for k in spectra])
        temp_flux = np.concatenate([templates[k].flux for k in templates])

        return np.median(spec_flux), np.median(temp_flux)

    def process_spectrum(self, spectrum):
        """
        Preprocess the spectrum.
        """

        spec = spectrum.copy()
        if self.spec_norm is not None:
            spec.multiply(1.0 / self.spec_norm)

        return spec

    def process_template(self, template, spectrum, rv, psf=None):
        """
        Preprocess the template to match the observation
        """

        def process_template_impl(rv):
            #   1. Make a copy, not in-place update
            t = template.copy()

            # Normalize
            if self.temp_norm is not None:
                t.multiply(1.0 / self.temp_norm)

            #   3. Shift template to the desired RV
            t.set_rv(rv)

            #   4. If a PSF if provided, convolve down template to the instruments
            #      resolution. Otherwise assume the template is already convolved.
            if psf is not None:
                t.convolve_psf(psf)

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
            else:
                temp = process_template_impl(rv_q)

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

        return temp

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

        if not self.flux_corr:
            bases = None

            phi = np.zeros(rvv.shape)
            chi = np.zeros(rvv.shape)
        else:
            # Evaluate the basis for each spectrum
            # TODO: this can be cached because the spectrum doesn't
            #       change between evaluations!
            bases = {}
            basis_size = None
            for k in spectra:
                spec = spectra[k]
                bases[k] = self.flux_corr_basis(spec.wave)
                if basis_size is None:
                    basis_size = bases[k].shape[-1]
                elif basis_size != bases[k].shape[-1]:
                    raise Exception('Inconsistent basis size')

            phi = np.zeros(rvv.shape + (basis_size,))
            chi = np.zeros(rvv.shape + (basis_size, basis_size))
                
        # For each value of rv0, sum up log_L contributions from spectrum - template pairs
        for i in range(rvv.size):
            tt = []         # Collect preprocessed templates for tracing
            cc = []         # Collect continuum basis functions for tracing
            for k in spectra:
                psf = self.template_psf[k] if self.template_psf is not None else None
                
                spec = self.process_spectrum(spectra[k])
                temp = self.process_template(templates[k], spec, rvv[i], psf=psf)
                tt.append(temp)

                if self.use_mask:
                    mask = ~spec.mask if spec.mask is not None else np.s_[:]
                else:
                    mask = np.s_[:]

                s2 = spec.flux_err ** 2
                
                if bases is None:
                    phi[i] += np.sum(spec.flux[mask] * temp.flux[mask] / s2[mask])
                    chi[i] += np.sum(temp.flux[mask] ** 2 / s2[mask])

                    if self.trace is not None:
                        log_L = self.eval_log_L(phi[i], chi[i])
                        self.trace.on_calculate_log_L(rvv[i], spectra, tt, log_L, phi[i], chi[i])
                else:
                    basis = bases[k]
                    # TODO: figure out where to use the mask on basis
                    raise NotImplementedError()
                    phi[i] += np.sum(spec.flux[mask, None] * temp.flux[mask, None] * basis / s2[mask, None], axis=0)
                    chi[i] += np.sum(temp.flux[mask, None, None] ** 2 * np.matmul(basis[:, :, None], basis[:, None, :]) / s2[mask, None, None], axis=0)

                    if self.trace is not None:
                        # First dimension must index rv items
                        log_L = self.eval_log_L(phi[np.newaxis, i], chi[np.newaxis, i])
                        self.trace.on_calculate_log_L(rvv[i], spectra, tt, log_L[0], phi[i], chi[i])

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
        if not self.flux_corr:
            a = phi / chi
        else:
            a = np.matmul(np.linalg.inv(chi), phi)
        return a

    def eval_nu2(self, phi, chi):
        # TODO: This doesn support vectorized inputs for phi and chi
        if not self.flux_corr:
            return phi ** 2 / chi
        else:
            chi_inv = np.linalg.inv(chi)
            return np.dot(phi, np.matmul(chi_inv, phi))

    def eval_log_L(self, phi, chi):
        if not self.flux_corr:
            log_L = np.empty(phi.shape)
        else:
            log_L = np.empty(phi.shape[:-1])

        for i in np.ndindex(log_L.shape):
            log_L[i] = 0.5 * self.eval_nu2(phi[i], chi[i])
        
        return log_L

    def calculate_log_L(self, spectra, templates, rv):
        phi, chi = self.eval_phi_chi(spectra, templates, rv)
        log_L = self.eval_log_L(phi, chi)

        return log_L, phi, chi

    def calculate_rv_error(self, spectra, templates, rv_0, rv_step=1.0):
        """
        Calculate the RV fitting error around the best fit value using
        numerical differentiation of the matrix elements of phi and chi.
        """

        def nu(rv):
            phi, chi = self.eval_phi_chi(spectra, templates, rv)
            return np.sqrt(self.eval_nu2(phi, chi))

        # Second derivative by RV
        dd_nu = nd.Derivative(nu, step=rv_step, n=2)

        nu_0 = nu(rv_0)
        dd_nu_0 = dd_nu(rv_0)

        return -1.0 / (nu_0 * dd_nu_0)

    def calculate_fisher(self, spectra, templates, rv_0, rv_step=1.0):
        """
        Calculate the full Fisher matrix using numerical differentiation around `rv_0`.
        """

        # We need to pack and unpack phi and chi because numdifftools don't
        # properly handle multivariate functions of higher dimensions.

        if not self.flux_corr:
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
        d_phi_chi = nd.Derivative(phi_chi, step=rv_step)
        dd_phi_chi = nd.Derivative(phi_chi, step=rv_step, n=2)

        d_phi_0, d_chi_0 = unpack_phi_chi(d_phi_chi(rv_0), phi_0.size)
        dd_phi_0, dd_chi_0 = unpack_phi_chi(dd_phi_chi(rv_0), phi_0.size)

        if not self.flux_corr:
            # TODO: use special calculations from Alex
            F = np.empty((2, 2), dtype=phi_0.dtype)
            F[0, 0] = chi_0
            F[1, 0] = F[0, 1] = d_phi_0   # TODO: is this correct here?
            F[1, 1] = -a_0 * dd_phi_0 + 0.5 * a_0**2 * dd_chi_0
        else:
            # Put together the Fisher matrix
            ndf = phi_0.size
            F = np.empty((ndf + 1, ndf + 1), dtype=phi_0.dtype)

            F[:ndf, :ndf] = chi_0
            F[-1, :-1] = F[:-1, -1] = -d_phi_0 + np.matmul(d_chi_0, a_0)
            F[-1, -1] = - np.dot(a_0, dd_phi_0) + 0.5 * np.dot(a_0, np.matmul(dd_chi_0, a_0))

        return F

    def calculate_fisher_special(self, spectra, templates, rv, rv_step=1.0):
        """
        Calculate the Fisher matrix numerically from a local finite difference
        around `rv` in steps of `step`.
        """

        # TODO: what if we are fitting the continuum as well?

        if not isinstance(spectra, Iterable):
            spectra = [ spectra ]
        if not isinstance(templates, Iterable):
            templates = [ templates ]

        # TODO: Verify math in sum over spectra - templates

        psi00 = 0.0
        psi01 = 0.0
        psi11 = 0.0
        psi02 = 0.0
        phi02 = 0.0
        phi00 = 0.0

        for k in spectra:
            spec = spectra[k]
            temp = templates[k]
            psf = self.template_psf[k] if self.template_psf is not None else None
            temp0 = self.process_template(temp, spec, rv, psf=psf)
            temp1 = self.process_template(temp, spec, rv + rv_step, psf=psf)
            temp2 = self.process_template(temp, spec, rv - rv_step, psf=psf)

            # Calculate the centered diffence of the flux
            d1  = 0.5 * (temp2.flux - temp1.flux) / rv_step
            d2  = (temp1.flux + temp2.flux - 2 * temp0.flux) / rv_step

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

        return F

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

    def fit_rv(self, spectra, templates, rv_0=None, rv_bounds=(-500, 500), guess_rv_steps=31, method='Golden'):
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

        if rv_0 is None:
            rv_0 = self.guess_rv(spectra, templates, rv_bounds=rv_bounds, rv_steps=guess_rv_steps)
            self.rv_0 = rv_0

        # Run optimization for log_L

        def llh(rv):
            log_L, phi, chi = self.calculate_log_L(spectra, templates, rv)
            return -log_L

        # Multivariate method
        #out = minimize(llh, [rv0], method=method)
        
        # Univariate
        if rv_bounds is not None:
            # NOTE: scipy.optimize is sensitive to type of args
            rv_bounds = tuple(float(b) for b in rv_bounds)
            bracket = (rv_bounds[0], float(rv_0), rv_bounds[1])
        else:
            bracket = None
            
        out = minimize_scalar(llh, bracket=bracket, bounds=rv_bounds, method=method)

        # direct argmin method
        # rvv = np.linspace(100, 300, 400)
        # L = llh(rvv)
        # mi = np.argmin(L)
        # class Result(): pass
        # out = Result()
        # out.success = True
        # out.x = rvv[mi]

        if out.success:
            # out.nit
            # rv_fit = out.x[0]
            rv_fit = out.x

            # If tracing, evaluate the template at the best fit RV.
            # TODO: can we cache this for better performance?
            if self.trace is not None:
                tt = {}
                for k in spectra:
                    spec = spectra[k]
                    temp = templates[k]
                    psf = self.template_psf[k] if self.template_psf is not None else None
                
                    t = self.process_template(temp, spec, rv_fit, psf=psf)
                    tt[k] = t
                self.trace.on_fit_rv(rv_fit, spectra, tt)
        else:
            raise Exception(f"Could not fit RV using `{method}`")

        # Calculate the error from the Fisher matrix
        F = self.calculate_fisher(spectra, templates, rv_fit)
        iF = np.linalg.inv(F)
        rv_err = np.sqrt(iF[-1, -1])  # sigma

        return rv_fit, rv_err

    def fit(self, spec):
        pass