import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.optimize import minimize_scalar
from collections.abc import Iterable

from pfs.ga.pfsspec.core.obsmod.resampling import FluxConservingResampler

class RVFitTrace():
    def on_guess_rv(self, rv, log_L, fit, function, pp, pcov):
        pass

    def on_calculate_log_L(self, rv, spec, temp, log_L, phi, chi):
        pass

    def on_fit_rv(self, rv, spec, temp):
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

            self.rv_0 = None                # RV initial guess
            self.rv_bounds = None           # Find RV between these bounds

            self.basis_functions = None     # Callable that provides the basis functions for flux correction / continuum fitting.
        else:
            self.trace = orig.trace

            self.template_psf = orig.template_psf
            self.template_resampler = orig.template_resampler

            self.rv_0 = orig.rv_0
            self.rv_bounds = orig.rv_bounds

            self.basis_functions = orig.basis_functions

    def process_template(self, template, spectrum, rv, psf=None):
        """
        Preprocess the template to match the observation
        """

        #   1. Make a copy, not in-place update
        t = template.copy()

        #   2. Shift template to the desired RV
        t.set_rv(rv)

        #   3. If a PSF if provided, convolve down template to the instruments
        #      resolution. Otherwise assume the template is already convolved.
        if psf is not None:
            t.convolve_psf(psf)

        #   4. Resample template to the binning of the observation
        t.apply_resampler(self.template_resampler, spectrum.wave, spectrum.wave_edges)

        return t    

    def calculate_log_L(self, spectra, templates, rv):
        """
        Calculate the log-likelihood of an observed spectrum for a template with RV.

        It assumes that the template is already convolved down to the instrumental
        resolution.
        """

        if not isinstance(rv, np.ndarray):
            rvv = np.array([rv])
        else:
            rvv = rv.flatten()

        if self.basis_functions is None:
            bases = None

            phi = np.zeros(rvv.shape)
            chi = np.zeros(rvv.shape)
            log_L = np.zeros(rvv.shape)
        else:
            # Evaluate the basis for each spectrum
            bases = {}
            basis_size = None
            for k in spectra:
                spec = spectra[k]
                bases[k] = self.basis_functions(spec.wave)
                if basis_size is None:
                    basis_size = bases[k].shape[-1]
                elif basis_size != bases[k].shape[-1]:
                    raise Exception('Inconsistent basis size')

            phi = np.zeros(rvv.shape + (basis_size,))
            chi = np.zeros(rvv.shape + (basis_size, basis_size))
            log_L = np.zeros(rvv.shape)
                
        # For each value of rv0, sum up log_L contributions from spectrum - template pairs
        for i in range(rvv.size):
            tt = []         # Collect preprocessed templates for tracing
            cc = []         # Collect continuum basis functions for tracing
            for k in spectra:
                spec = spectra[k]
                temp = templates[k]
                psf = self.template_psf[k] if self.template_psf is not None else None
                
                t = self.process_template(temp, spec, rvv[i], psf=psf)
                tt.append(t)

                s2 = spec.flux_err ** 2
                
                if bases is None:
                    phi[i] += np.sum(spec.flux * t.flux / s2)
                    chi[i] += np.sum(t.flux ** 2 / s2)
                else:
                    basis = bases[k]
                    phi[i] += np.sum(spec.flux[:, None] * t.flux[:, None] * basis / s2[:, None], axis=0)
                    chi[i] += np.sum(t.flux[:, None, None] ** 2 * np.matmul(basis[:, :, None], basis[:, None, :]) / s2[:, None, None], axis=0)

            if bases is None:
                log_L[i] = 0.5 * phi[i] ** 2 / chi[i]
            else:
                chi_inv = np.linalg.inv(chi[i])
                chi_inv_phi = np.matmul(chi_inv, phi[i])
                log_L[i] = 0.5 * np.dot(phi[i], chi_inv_phi)

            if self.trace is not None:
                self.trace.on_calculate_log_L(rvv[i], spectra, tt, log_L[i], phi[i], chi[i])

        if not isinstance(rv, np.ndarray):
            log_L = log_L[0]
            phi = phi[0]
            chi = chi[0]
        else:
            log_L = log_L.reshape(rv.shape)
            if bases is None:
                phi = phi.reshape(rv.shape)
                chi = chi.reshape(rv.shape)
            else:
                phi = phi.reshape(rv.shape + (basis_size,))
                chi = chi.reshape(rv.shape + (basis_size, basis_size))

        return log_L, phi, chi

    def get_fisher(self, spectra, templates, rv, rv_step=1.0):
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

        if out.success:
            # out.nit
            # rv_fit = out.x[0]
            rv_fit = out.x

            # If tracing, evaluate the template at the best fit RV.
            # TODO: can we cache this for better performance?
            if self.trace is not None:
                tt = []
                for k in spectra:
                    spec = spectra[k]
                    temp = templates[k]
                    psf = self.template_psf[k] if self.template_psf is not None else None
                
                    t = self.process_template(temp, spec, rv_fit, psf=psf)
                    tt.append(t)
                self.trace.on_fit_rv(rv_fit, spectra, tt)

            # Calculate the error from the Fisher matrix
            F = self.get_fisher(spectra, templates, rv_fit)
            iF = np.linalg.inv(F)
            rv_err = np.sqrt(iF[1, 1])  # sigma

            return rv_fit, rv_err
        else:
            raise Exception(f"Could not fit RV using `{method}`")

    def fit(self, spec):
        pass