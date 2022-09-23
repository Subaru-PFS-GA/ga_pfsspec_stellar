import numpy as np
from scipy.optimize import curve_fit, minimize

class RVFit():
    def __init__(self, orig=None):
        
        if not isinstance(orig, RVFit):
            self.grid = None
            self.psf = None
        else:
            self.grid = orig.grid
            self.psf = orig.psf

    def get_template(self, convolve=True, **kwargs):
        """
        Generate a noiseless template spectrum with same line spread function as the
        observation but keep the original, high-resolution binning.
        """

        # TODO: add template caching

        temp = self.grid.get_nearest_model(**kwargs)
        temp.cont = None        # Make sure it's not passed around for better performance
        temp.mask = None

        if convolve:
            temp.convolve_psf(self.psf)

        return temp

    def rebin_template(self, template, rv, spec):
        """
        Shift and rebin template to match the wavelength grid of `spec`.
        """

        # Make a copy, not in-place update
        temp = template.copy()
        temp.set_rv(rv)

        # TODO: this uses pysynphot to rebin, which might be too slow
        temp.rebin(spec.wave, spec.wave_edges)

        return temp

    def get_log_L(self, spec, template, rv):
        """
        Calculate the log-likelihood of an observed spectrum for a template with RV.
        Also return `phi` and `chi` for the Fisher matrix.

        It assumes that the template is already convolved down to the instrumental
        resolution.

        Not to be used for the rv fitting, but rather for testing and error estimation.
        """

        if not isinstance(rv, np.ndarray):
            v0 = np.array([rv])
        else:
            v0 = rv.flatten()

        nu = np.zeros_like(v0)
        phi = np.zeros_like(v0)
        chi = np.zeros_like(v0)

        s2 = spec.flux_err ** 2

        for i in range(v0.size):
            temp = self.rebin_template(template, v0[i], spec)
            phi[i] = np.sum(spec.flux * temp.flux / s2)
            chi[i] = np.sum(temp.flux ** 2 / s2)
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

    def get_fisher(self, spec, template, rv, step=1.0):
        """
        Calculate the Fisher matrix numerically from a local finite difference
        around `rv` in steps of `step`.
        """

        temp0 = self.rebin_template(template, rv, spec)
        temp1 = self.rebin_template(template, rv + step, spec)
        temp2 = self.rebin_template(template, rv - step, spec)

        # Calculate the centered diffence of the flux
        d1  = 0.5 * (temp2.flux - temp1.flux)
        d2  = (temp1.flux + temp2.flux - 2 * temp0.flux)

        # Build the different terms
        vm = spec.flux_err ** 2
        psi00 = np.sum(temp0.flux * temp0.flux / vm)
        psi01 = np.sum(temp0.flux * d1 / vm)
        psi11 = np.sum(d1 * d1 / vm)
        psi02 = np.sum(temp0.flux * d2 / vm)
        phi02 = np.sum(spec.flux * d2 / vm)
        phi00 = np.sum(spec.flux * temp0.flux / vm)
        chi   = psi00 
        a0    = phi00 / psi00
        
        F00 = psi00
        F01 = a0 * psi01
        F11 = a0 ** 2 * (psi02 - phi02 / a0 + psi11)

        F  = np.array([[F00,F01],[F01,F11]])

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

    def guess_rv(self, spec, template, rv_bounds=(-500, 500), rv_steps=31):
        """
        Given a spectrum and a template, make a good initial guess for RV where a minimization
        algorithm can be started from.
        """

        rv = np.linspace(*rv_bounds, rv_steps)
        y0, _, _ = self.get_log_L(spec, template, rv)

        pp, pcov = self.fit_lorentz(rv, y0)

        return pp[1]

    def fit_rv(self, spec, template, rv0=None, rv_bounds=(-500, 500), rv_steps=31, method="Nelder-Mead"):
        """
        Given a spectrum and a template, find the best fit RV by maximizing the log likelihood.

        If no initial guess is provided, rv0 is determined automatically.
        """

        def llh(rv):
            nu, _, _ = self.get_log_L(spec, template, rv)
            return -nu

        if rv0 is None:
            rv0 = self.guess_rv(spec, template, rv_bounds=rv_bounds, rv_steps=rv_steps)

        out = minimize(llh, [rv0], method=method)
        
        if out.success:
            return out.x[0]         # covariance?
        else:
            raise Exception(f"Could not fit RV using `{method}`")

    def fit(self, spec):
        pass