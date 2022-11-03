import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.optimize import minimize_scalar
from collections.abc import Iterable

from pfs.ga.pfsspec.core.obsmod.resampling import FluxConservingResampler

class RVFitTrace():
    def on_guess_rv(self, rv, log_L, fit, function, pp, pcov):
        pass

    def on_calculate_log_L(self, rv, spec, temp):
        pass

    def on_fit_rv(self, rv, spec, temp):
        pass

class RVFit():
    """
    A basic RV fit implementation based on a non-linear maximum likelihood method.
    It takes a single, high resolution template which is fitted to the observation.
    Optionally, the template is convolved with the instrumental PSF after Doppler
    shifting but before convolution.
    """

    def __init__(self, trace=None, orig=None):
        
        if not isinstance(orig, RVFit):
            self.trace = trace              # Collect debug info

            self.template_psf = None        # Convolve template to instrument resolution
            self.template_resampler = FluxConservingResampler()  # Resample template to instrument pixels

            self.rv0 = None                 # RV initial guess
            self.rv_bounds = None           # Find RV between these bounds
        else:
            self.trace = orig.trace

            self.template_psf = orig.template_psf
            self.template_resampler = orig.template_resampler

            self.rv0 = orig.rv0
            self.rv_bounds = orig.rv_bounds

    def process_template(self, template, spectrum, rv):
        """
        Preprocess the template to match the observation
        """

        #   1. Make a copy, not in-place update
        t = template.copy()

        #   2. Shift template to the desired RV
        t.set_rv(rv)

        #   3. If a PSF if provided, convolve down template to the instruments
        #      resolution. Otherwise assume the template is already convolved.
        if self.template_psf is not None:
            t.convolve_psf(self.template_psf)

        #   4. Resample template to the binning of the observation
        t.apply_resampler(self.template_resampler, spectrum.wave, spectrum.wave_edges)

        return t

    def calculate_log_L(self, spectra, templates, rv):
        """
        Calculate the log-likelihood of an observed spectrum for a template with RV.
        Also return `phi` and `chi` for the Fisher matrix.

        It assumes that the template is already convolved down to the instrumental
        resolution.

        Not to be used for the rv fitting, but rather for testing and error estimation.
        """

        if not isinstance(spectra, Iterable):
            spectra = [ spectra ]
        if not isinstance(templates, Iterable):
            templates = [ templates ]

        if not isinstance(rv, np.ndarray):
            rv0 = np.array([rv])
        else:
            rv0 = rv.flatten()

        nu = np.zeros_like(rv0)
        phi = np.zeros_like(rv0)
        chi = np.zeros_like(rv0)

        # For each value of rv0, sum up log_L contributions from spectrum - template pairs
        for i in range(rv0.size):
            phi[i] = 0.0
            chi[i] = 0.0

            tt = []
            for spec, temp in zip(spectra, templates):
                t = self.process_template(temp, spec, rv0[i])
                tt.append(t)

                s2 = spec.flux_err ** 2
                
                phi[i] += np.sum(spec.flux * t.flux / s2)
                chi[i] += np.sum(t.flux ** 2 / s2)

            if self.trace is not None:
                self.trace.on_calculate_log_L(rv0[i], spectra, tt)
                
            nu[i] = phi[i] / np.sqrt(chi[i])

        if not isinstance(rv, np.ndarray):
            nu = nu[0]
            phi = phi[0]
            chi = chi[0]
        else:
            nu = nu.reshape(rv.shape)
            phi = phi.reshape(rv.shape)
            chi = chi.reshape(rv.shape)

        return nu, phi, chi

    def get_fisher(self, spectra, templates, rv, rv_step=1.0):
        """
        Calculate the Fisher matrix numerically from a local finite difference
        around `rv` in steps of `step`.
        """

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

        for spec, temp in zip(spectra, templates):
            temp0 = self.process_template(temp, spec, rv)
            temp1 = self.process_template(temp, spec, rv + rv_step)
            temp2 = self.process_template(temp, spec, rv - rv_step)

            # Calculate the centered diffence of the flux
            d1  = 0.5 * (temp2.flux - temp1.flux)
            d2  = (temp1.flux + temp2.flux - 2 * temp0.flux)

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
        y0, _, _ = self.calculate_log_L(spectra, templates, rv)

        pp, pcov = self.fit_lorentz(rv, y0)

        if self.trace is not None:
            self.trace.on_guess_rv(rv, y0, self.lorentz(rv, *pp), 'lorentz', pp, pcov)

        return pp[1]

    def fit_rv(self, spectra, templates, rv0=None, rv_bounds=(-500, 500), guess_rv_steps=31, method="Nelder-Mead"):
        """
        Given a set of spectra and templates, find the best fit RV by maximizing the log likelihood.
        Spectra are assumed to be of the same object in different wavelength ranges.

        If no initial guess is provided, rv0 is determined automatically.
        """

        if not isinstance(spectra, Iterable):
            spectra = [ spectra ]
        if not isinstance(templates, Iterable):
            templates = [ templates ]

        rv0 = rv0 if rv0 is not None else self.rv0
        rv_bounds = rv_bounds if rv_bounds is not None else self.rv_bounds

        # No initial RV estimate provided, try to guess it and save it for later

        if rv0 is None:
            rv0 = self.guess_rv(spectra, templates, rv_bounds=rv_bounds, rv_steps=guess_rv_steps)
            self.rv0 = rv0

        # Run optimization for log_L

        def llh(rv):
            nu, _, _ = self.calculate_log_L(spectra, templates, rv)
            return -nu

        # Multivariate method
        #out = minimize(llh, [rv0], method=method)
        
        # Univariate
        if rv_bounds is not None:
            # NOTE: scipy.optimize is sensitive to type of args
            rv_bounds = tuple(float(b) for b in rv_bounds)
            bracket = (rv_bounds[0], float(rv0), rv_bounds[1])
        else:
            bracket = None

        out = minimize_scalar(llh, bracket=bracket, bounds=rv_bounds, method='Golden')

        if out.success:
            # out.nit

            # Calculate the error from the Fisher matrix
            # rv_fit = out.x[0]
            rv_fit = out.x

            if self.trace is not None:
                tt = []
                for spec, temp in zip(spectra, templates):
                    t = self.process_template(temp, spec, rv_fit)
                    tt.append(t)
                self.trace.on_fit_rv(rv_fit, spectra, tt)

            F = self.get_fisher(spectra, templates, rv_fit)
            iF = np.linalg.inv(F)
            rv_err = np.sqrt(iF[1, 1])  # sigma

            return rv_fit, rv_err
        else:
            raise Exception(f"Could not fit RV using `{method}`")

    def fit(self, spec):
        pass