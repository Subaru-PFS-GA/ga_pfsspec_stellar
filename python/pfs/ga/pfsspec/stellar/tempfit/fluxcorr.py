import numpy as np
import numdifftools as nd
from collections.abc import Iterable

from pfs.ga.pfsspec.core.util.copy import safe_deep_copy
from pfs.ga.pfsspec.core.util.args import get_arg
from ..fluxcorr import PolynomialFluxCorrection
from .tempfittrace import TempFitTrace
from .tempfittracestate import TempFitTraceState
from .correctionmodel import CorrectionModel

from .setup_logger import logger

class FluxCorr(CorrectionModel):
    """
    Model class for flux correction based stellar template fitting.
    """

    def __init__(self, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, FluxCorr):
            self.use_flux_corr = True           # Use flux correction. Scalar if no basis is provided, otherwise linear combination of basis functions
            self.flux_corr_type = PolynomialFluxCorrection
            self.flux_corr_degree = 5           # Flux correction degree
            self.flux_corr_per_arm = False      # Do flux correction independently for each arm
            self.flux_corr_per_fiber = False    # Do flux correction independently for each fiber
            self.flux_corr_per_exp = False      # Do flux correction independently for each exposure

            self.flux_corr = None               # Flux correction model, optionally for each arm and/or exposure
        else:
            self.use_flux_corr = orig.use_flux_corr
            self.flux_corr_type = orig.flux_corr_type
            self.flux_corr_degree = orig.flux_corr_degree
            self.flux_corr_per_arm = orig.flux_corr_per_arm
            self.flux_corr_per_fiber = orig.flux_corr_per_fiber
            self.flux_corr_per_exp = orig.flux_corr_per_exp

            self.flux_corr = safe_deep_copy(orig.flux_corr)

        self.reset()

    def reset(self):
        super().reset()

        self.flux_corr_basis_cache = None
        self.flux_corr_basis_size = None

    def add_args(self, config, parser):
        super().add_args(config, parser)

        parser.add_argument('--flux-corr', action='store_true', dest='flux_corr', help='Do flux correction.\n')
        parser.add_argument('--no-flux-corr', action='store_false', dest='flux_corr', help='No flux correction.\n')
        parser.add_argument('--flux-corr-deg', type=int, help='Degree of flux correction polynomial.\n')
        parser.add_argument('--flux-corr-per-arm', action='store_true', dest='flux_corr_per_arm', help='Flux correction per arm.\n')
        parser.add_argument('--flux-corr-per-fiber', action='store_true', dest='flux_corr_per_fiber', help='Flux correction per fiber.\n')
        parser.add_argument('--flux-corr-per-exp', action='store_true', dest='flux_corr_per_exp', help='Flux correction per exposure.\n')

    def init_from_args(self, script, config, args):
        super().init_from_args(script, config, args)

        self.use_flux_corr = get_arg('flux_corr', self.use_flux_corr, args)
        self.flux_corr_degree = get_arg('flux_corr_deg', self.flux_corr_degree, args)
        self.flux_corr_per_arm = get_arg('flux_corr_per_arm', self.flux_corr_per_arm, args)
        self.flux_corr_per_fiber = get_arg('flux_corr_per_fiber', self.flux_corr_per_fiber, args)
        self.flux_corr_per_exp = get_arg('flux_corr_per_exp', self.flux_corr_per_exp, args)

    def create_flux_corr(self, wlim):
        """
        Instantiate a flux correction model.
        """

        return self.flux_corr_type(degree=self.flux_corr_degree, wlim=wlim)

    def init_models(self, spectra, rv_bounds=None, force=False):
        """
        Initialize the flux correction model for each arm and exposure.
        """

        if self.use_flux_corr and (self.flux_corr is None or force):
            self.flux_corr = self.tempfit.init_correction_model(spectra,
                                                                per_arm=self.flux_corr_per_arm,
                                                                per_exp=self.flux_corr_per_exp,
                                                                rv_bounds=rv_bounds,
                                                                round_to=100,
                                                                create_model_func=self.create_flux_corr)
        elif self.flux_corr is not None:
            logger.info("Flux correction model is already initialized, skipping reinitialization.")

    def get_coeff_count(self):
        """
        Determine the number of linear coefficients of the flux correction. Even when flux correction
        is not used, the amplitude (absolute calibration) can very from arm to arm.

        Flux correction can be an overall model for all arms or different for each arm and the same
        amplitude and coefficients can be used for all exposures made with that arm, or a different
        amplitude for every single exposure.
        """
        
        if self.use_flux_corr:
            # Sum up the flux correction model parameters
            coeff_count = sum([ model.get_coeff_count() for mi, model in self.flux_corr.items() ])
        else:
            coeff_count = 0

        return coeff_count
    
    def eval_flux_corr_basis(self, spectra):
        """
        Evaluate the basis functions for each exposure of each arm on the wavelength
        grid of the actual exposure. The basis will have the size of the total number of
        linear coefficients, taking config options such as `amplutide_per_exp` and 
        `flux_corr_per_exp` into account. Basis vectors which correspond to a 
        different arm or exposure will be set to zero.

        Parameters
        ----------
        spectra : dict of list of Spectra
            Dictionary of spectra for each arm and exposure.

        Returns
        -------
        basis : dict of list of np.ndarray
            Dictionary of basis vectors, in a form of ndarray for each arm and exposure.

        The structure of the basis is determined by the configuration options
        `amplitude_per_arm`, `amplitude_per_exp`, `flux_corr_per_arm` and `flux_corr_per_exp`.
        First, the number of different amplitudes is determined from `amplitude_per_arm`,
        `amplitude_per_exp` and also from the number of arms and exposures in `spectra`.
        The first k rows of the basis are set to 1.0, where k is the number of amplitudes
        and the spectrum matches that arm and exposure id.


        Note, that the flux correction model's `get_basis_callable˙ returns a function
        that returns a basis that doesn't include the constant function, those are
        added separately to the basis based on `amplitude_per_arm` and `amplitude_per_exp`.
        """

        # Calculate the total number of amplitudes and linear coefficients
        amp_count = self.tempfit.get_amp_count(spectra)
        
        if self.use_flux_corr:
            coeff_count = [ model.get_coeff_count() for mi, model in self.flux_corr.items() ]
            coeff_count_cum = np.cumsum([0] + coeff_count)
            basis_size = amp_count + np.sum(coeff_count)
        else:
            basis_size = amp_count
        
        basis = { arm: [] for arm in spectra}       # dict of lists of basis vectors for each spectrum
        
        # Fill in the basis vectors belonging to the amplitudes first.
        # Since the correction is multiplicative, the basis vectors corresponding
        # to amplitudes are then constant.
        for arm, ei, mi, spec in self.tempfit.enumerate_spectra(spectra,
                                                                per_arm=self.tempfit.amplitude_per_arm,
                                                                per_exp=self.tempfit.amplitude_per_exp,
                                                                include_none=True):
            # `mi` now indexes the different amplitudes

            if spec is not None:
                bb = np.zeros((spec.wave.shape[0], basis_size), dtype=spec.flux.dtype)
                bb[:, mi] = 1.0
                basis[arm].append(bb)
            else:
                basis[arm].append(None)

        # Fill in the basis vectors for the flux correction coefficients
        if self.use_flux_corr:
            for arm, ei, mi, spec in self.tempfit.enumerate_spectra(spectra,
                                                                    per_arm=self.flux_corr_per_arm,
                                                                    per_exp=self.flux_corr_per_exp,
                                                                    include_none=True):
                
                # `mi` now indexes the different flux correction models

                if spec is not None:
                    bb = basis[arm][ei]

                    # Get the flux correction model
                    flux_corr = self.flux_corr[mi]

                    # Get the basis vector generator function
                    f = flux_corr.get_basis_callable()

                    # Determine the slice where the basis vectors go into the matrix
                    fr = amp_count + coeff_count_cum[mi]
                    to = amp_count + coeff_count_cum[mi] + coeff_count[mi]

                    # Evaluate the basis vector generator function
                    bb[:, fr:to] = f(spec.wave)

        return basis, basis_size

    def get_flux_corr_basis(self, spectra):
        if self.flux_corr_basis_cache is None:
            self.flux_corr_basis_cache, self.flux_corr_basis_size = self.eval_flux_corr_basis(spectra)

        bases, basis_size = self.flux_corr_basis_cache, self.flux_corr_basis_size

        return bases, basis_size

    def calculate_phi_chi(self, spectra, templates, rv):
        """
        Calculate the log-likelihood of an observed spectrum for a template with RV.
        """

        pp_specs = self.tempfit.preprocess_spectra(spectra)
        pp_temps = self.tempfit.preprocess_templates(spectra, templates, rv)
        phi_0, chi_0, ndf_0 = self.eval_phi_chi(pp_specs, pp_temps)
        return self.eval_phi_chi(pp_specs, pp_temps)

    def eval_phi_chi(self, spectra, templates):
        """
        Calculate the log-likelihood of an observed spectrum for a template with RV.

        It assumes that the templates are already convolved down to the instrumental
        resolution, shifted to some RV and resampled to the detector pixels.
        """

        # Evaluate the basis functions
        # TODO: move this outside of function and pass in as arguments
        bases, basis_size = self.get_flux_corr_basis(spectra)

        # TODO: move this outside of function and pass in as arguments
        # Calculate the number of flux-correction parameters, including
        # the amplitudes which might be different for each arm and exposure

        # Size of phi and chi is amp_count + coeff_count
        amp_count = self.tempfit.get_amp_count(spectra)
        param_count = self.get_coeff_count()
        coeff_count = amp_count + param_count

        # Sum up log_L contributions from spectrum - template pairs
        phi = np.zeros((coeff_count,))
        chi = np.zeros((coeff_count, coeff_count))
        ndf = np.zeros((1,))
    
        for arm, ei, mi, spec in self.tempfit.enumerate_spectra(spectra, per_arm=False, per_exp=False, include_none=False):
            temp = templates[arm][ei]
            basis = bases[arm][ei]

            # Combine masks
            mask = spec.mask & temp.mask

            # Weight from the spectrum (flux_err)
            if spec.sigma2 is not None:
                sigma2 = spec.sigma2
                mask &= ~np.isnan(sigma2)
            else:
                sigma2 = None

            # Weight from template (optional)
            if self.tempfit.use_weight and temp.weight is not None:
                weight = temp.weight / temp.weight.sum() * temp.weight.size
                mask &= ~np.isnan(weight)
            else:
                weight = None

            # Mask from flux correction basis
            # Be cautious in case any item in wave is nan
            mask &= ~np.any(np.isnan(basis), axis=-1)       

            # Verify that the mask is not empty or too few points to fit
            if mask.sum() == 0:
                raise Exception("Too few unmasked values to fit the spectrum.")
            
            # Calculate phi and chi and sum up along wavelength
            pp = spec.flux[mask] * temp.flux[mask]
            cc = temp.flux[mask] ** 2
            
            if weight is not None:
                pp *= weight[mask]
                cc *= weight[mask]
            
            if sigma2 is not None:
                pp /= sigma2[mask]
                cc /= sigma2[mask]
            
            bb =  basis[mask]
            phi += np.einsum('i,ij->j', pp, bb)
            chi += np.einsum('i,ij,ik->jk', cc, bb, bb)

            # Degrees of freedom
            ndf += mask.sum()

        if not self.use_flux_corr:
            ndf -= 1
        else:
            ndf -= basis_size

        if np.size(phi) == 1:
            phi = phi.item()

        if np.size(chi) == 1:
            chi = chi.item()

        if np.size(ndf) == 1:
            ndf = ndf.item()

        if self.trace is not None:
            self.trace.on_eval_phi_chi(spectra, templates, bases, phi, chi)

        return phi, chi, ndf
    
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
        
        if self.trace is not None:
            self.trace.on_eval_nu2(phi, chi, nu2)
        
        return nu2
    
    def eval_log_L_phi_chi(self, phi, chi):
        # Likelihood at the optimum in the flux amplitude / correction parameters.
        log_L = 0.5 * self.eval_nu2(phi, chi)

        if self.trace is not None:
            self.trace.on_eval_log_L_phi_chi(phi, chi, log_L)

        return log_L
    
    def eval_log_L_a(self, phi, chi, a):
        # Likelihood as a function of flux correction coefficients
        if not self.use_flux_corr:
            log_L = a * phi - 0.5 * a * a * chi
        else:
            log_L = np.squeeze(np.matmul(a[..., None, :], phi[..., :, None]) -
                               0.5 * np.matmul(a[..., None, :], np.matmul(chi, a[..., :, None])))

        if self.trace is not None:
            self.trace.on_eval_log_L_a(phi, chi, a, log_L)
        
        return log_L
    
    def eval_log_L(self, spectra, templates, /, a=None, return_phi_chi=False):
        """
        Calculate the value of the likelihood function given a the preprocessed exposures
        and templates.

        Templates are assumed to be Doppler shifted to a certain RV and resampled to
        the same grid as the spectra.
        
        This function does not add the log of priors and must not be called directly.
        """

        # Depending on whether the flux correction coefficients `a`
        # are supplied, calculate the log likelihood at the optimum of
        # flux correction or at the specified flux correction values.
        phi, chi, ndf = self.eval_phi_chi(spectra, templates)

        if a is None:
            log_L = self.eval_log_L_phi_chi(phi, chi)
        else:
            log_L = self.eval_log_L_a(phi, chi, a)

        if return_phi_chi:
            return log_L, phi, chi, ndf
        else:
            return log_L

    def calculate_coeffs(self, spectra, templates, a=None):
        """
        Given a set of spectra and preprocessed templates, calculate the
        flux correction coefficients.

        If flux correction coefficients are provided in `a`, they're returned
        without modification.
        """

        if a is None:
            phi, chi, ndf = self.eval_phi_chi(spectra, templates)
            a = self.eval_a(phi, chi)

        if np.ndim(a) > 1:
            a = a.squeeze(0)

        return a
    
    def eval_flux_corr(self, spectra, templates, rv, a=None):
        """
        Evaluate the flux correction for each exposure in each arm
        """

        if a is None:
            phi, chi, ndf = self.eval_phi_chi(spectra, templates, rv)
            a = self.eval_a(phi, chi)
        
        flux_corr = {}
        for arm in spectra:
            flux_corr[arm] = []
            ei = 0
            for ii, spec in enumerate(spectra[arm] if isinstance(spectra[arm], list) else [spectra[arm]]):
                if spec is not None:
                    flux_corr[arm].append(np.dot(self.flux_corr_basis_cache[arm][ei], a))
                    ei += 1
                else:
                    flux_corr[arm].append(None)

        return flux_corr

    def calculate_F_full_phi_chi(self, spectra, templates, rv_0, step=None):
        # Evaluate the Fisher matrix from the first and second derivatives of
        # phi and chi around rv_0, based on the flux correction formulate

        # We need to pack and unpack phi and chi because numdifftools don't
        # properly handle multivariate functions of higher dimensions.

        def pack_phi_chi(phi, chi):
            return np.concatenate([np.ravel(phi), np.ravel(chi)])

        def unpack_phi_chi(phi_chi, size):
            return phi_chi[:size], phi_chi[size:].reshape((size, size))

        def phi_chi(rv):
            phi, chi, ndf = self.calculate_phi_chi(spectra, templates, rv)
            return pack_phi_chi(phi, chi)

        # Calculate a_0
        phi_0, chi_0, ndf_0 = self.calculate_phi_chi(spectra, templates, rv_0)
        a_0 = self.eval_a(phi_0, chi_0)

        # First and second derivatives of the matrix elements by RV
        d_phi_chi = nd.Derivative(phi_chi, step=step)
        dd_phi_chi = nd.Derivative(phi_chi, step=step, n=2)

        d_phi_0, d_chi_0 = unpack_phi_chi(d_phi_chi(np.atleast_1d(rv_0)), np.size(phi_0))
        dd_phi_0, dd_chi_0 = unpack_phi_chi(dd_phi_chi(np.atleast_1d(rv_0)), np.size(phi_0))

        if not self.use_flux_corr:
            # TODO: use special calculations from Alex
            # Get the dtype from phi_0, even if it's a scalar
            np.dtype
            F = np.empty((2, 2), dtype=float)
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
    
    def calculate_F_full_alex(self, spectra, templates, rv, step=None):
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

        for arm in spectra:
            temp = templates[arm]
            for ei, spec in enumerate(spectra[arm] if isinstance(spectra[arm], list) else [spectra[arm]]):
                spec = self.tempfit.process_spectrum(arm, ei, spec)

                temp0 = self.tempfit.process_template(arm, temp, spec, rv)
                temp1 = self.tempfit.process_template(arm, temp, spec, rv + step)
                temp2 = self.tempfit.process_template(arm, temp, spec, rv - step)

                # Calculate the centered diffence of the flux
                d1 = 0.5 * (temp2.flux - temp1.flux) / step
                d2 = (temp1.flux + temp2.flux - 2 * temp0.flux) / step

                # Build the different terms
                s2 = spec.flux_err ** 2
                psi00 += np.sum(temp0.flux * temp0.flux / s2)
                psi01 += np.sum(temp0.flux * d1 / s2)
                psi11 += np.sum(d1 * d1 / s2)
                psi02 += np.sum(temp0.flux * d2 / s2)
                phi02 += np.sum(spec.flux * d2 / s2)
                phi00 += np.sum(spec.flux * temp0.flux / s2)
        
        chi = psi00
        a0 = phi00 / psi00
        
        F00 = psi00
        F01 = a0 * psi01
        F11 = a0 ** 2 * (psi02 - phi02 / a0 + psi11)

        F = np.array([[F00, F01], [F01, F11]])

        return -F, np.linalg.inv(F)
       
    def calculate_F(self, spectra, templates,
                    rv_0, rv_fixed=None,
                    step=None, mode='full', method='hessian'):
                    
        # Calculate the Fisher matrix using different methods

        if mode == 'full' and method == 'phi_chi':
            return self.calculate_F_full_phi_chi(spectra, templates, rv_0, step=step)
        elif mode == 'full' and method == 'alex':
            return self.calculate_F_full_alex(spectra, templates, rv_0, step=step)
        else:
            # Revert back to numeric differentiation
            return self.tempfit.calculate_F(spectra, templates,
                                            rv_0, rv_fixed=rv_fixed,
                                            step=step, mode=mode, method=method)

    def eval_rv_error_alex(self, spectra, templates, rv_0, step=1.0):
        """
        Calculate the RV fitting error around the best fit value using
        numerical differentiation of the matrix elements of phi and chi.
        """

        def nu(rv):
            phi, chi, ndf = self.eval_phi_chi(spectra, templates, rv)
            return np.sqrt(self.eval_nu2(phi, chi))

        # Second derivative by RV
        dd_nu = nd.Derivative(nu, step=step, n=2)

        nu_0 = nu(rv_0)
        dd_nu_0 = dd_nu(rv_0)

        return -1.0 / (nu_0 * dd_nu_0)
    
    def eval_correction(self, pp_specs, pp_temps, a=None):

        def eval_flux_corr(temp, basis, a):
            # Evaluate the flux correction model

            if self.use_flux_corr:
                # Full flux correction
                corr = np.dot(basis, a)
            else:
                # This is an amplitude only
                corr = a

            return corr

        if a is None:
            a = self.calculate_coeffs(pp_specs, pp_temps)

        bases, basis_size = self.get_flux_corr_basis(pp_specs)

        corrections = { arm: [] for arm in pp_specs }
        for arm in pp_specs:
            for ei, (spec, temp) in enumerate(zip(pp_specs[arm], pp_temps[arm])):
                if spec is not None:
                    corr = eval_flux_corr(temp, bases[arm][ei], a)
                    corrections[arm].append(corr)
                else:
                    corrections[arm].append(None)
                    
        return corrections
       
    def apply_correction(self, pp_specs, pp_temps, a=None):
        """
        Apply the flux correction to the templates. The templates are assumed to be
        already Doppler shifted and resampled to the same grid as the spectra.
        The function modifies `pp_temps` in place.

        Parameters
        ----------
        pp_specs : dict of list
            Dictionary of spectra for each arm and exposure.
        pp_temps : dict of list
            Dictionary of templates for each arm and exposure.
        a : array
            Flux correction coefficients.
        """

        corrections = self.eval_correction(pp_specs, pp_temps, a=a)

        for arm in pp_specs:
            for ei, (spec, temp, corr) in enumerate(zip(pp_specs[arm], pp_temps[arm], corrections[arm])):
                if temp is not None:
                    temp.multiply(corr)

