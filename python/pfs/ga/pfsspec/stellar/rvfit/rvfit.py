import os
import logging
from typing import Callable
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.optimize import minimize_scalar
from scipy.ndimage import binary_dilation
from scipy.interpolate import interp1d
from sklearn.preprocessing import PolynomialFeatures
import numdifftools as nd
from collections.abc import Iterable
from collections import namedtuple

from pfs.ga.pfsspec.core.util.args import *
from pfs.ga.pfsspec.core import Trace
from pfs.ga.pfsspec.core import Spectrum
import pfs.ga.pfsspec.core.plotting.styles as styles
from pfs.ga.pfsspec.core import Physics
from pfs.ga.pfsspec.core.caching import ReadOnlyCache
from pfs.ga.pfsspec.core.obsmod.resampling import FluxConservingResampler, Interp1dResampler
from pfs.ga.pfsspec.core.obsmod.fluxcorr import PolynomialFluxCorrection
from pfs.ga.pfsspec.core.sampling import Parameter, Distribution
from .rvfittrace import RVFitTrace
from .rvfittracestate import RVFitTraceState
from .rvfitresults import RVFitResults

class RVFit():
    """
    A basic RV estimation based on template fitting using a non-linear maximum likelihood or
    maximum significance method. Templates are optionally convolved with the instrumental LSF.
    The class supports various methods to determine the uncertainty of estimated parameters.
    """

    RESAMPLERS = {
        'interp': Interp1dResampler,
        'fluxcons': FluxConservingResampler,
    }

    def __init__(self, trace=None, orig=None):
        
        if not isinstance(orig, RVFit):
            self.trace = trace              # Collect debug info

            self.template_psf = None        # Dict of psf to downgrade templates with
            self.template_resampler = FluxConservingResampler()  # Resample template to instrument pixels
            self.template_cache_resolution = 50  # Cache templates with this resolution in RV
            self.template_wlim = None       # Model wavelength limits for each spectrograph arm
            self.template_wlim_buffer = 100 # Wavelength buffer in A, depends on line spread function
            
            self.cache_templates = False    # Cache PSF-convolved templates

            self.rv_0 = None                # RV initial guess
            self.rv_bounds = None           # Find RV between these bounds
            self.rv_prior = None
            self.rv_step = None             # RV step size for MCMC sampling

            self.use_flux_corr = False      # Use flux correction. Scalar if no basis is provided, otherwise linear combination of basis functions
            self.flux_corr = None           # Flux correction model

            self.spec_norm = None           # Spectrum (observation) normalization factor
            self.temp_norm = None           # Template normalization factor

            self.use_mask = True            # Use mask from spectrum, if available
            self.mask_bits = None           # Mask bits (None means any)
            self.use_error = True           # Use flux error from spectrum, if available
            self.use_weight = False         # Use weight from template, if available

            self.max_iter = 1000            # Maximum number of iterations to minimize significance

            self.mcmc_step = 10.0           # MCMC initial step size
            self.mcmc_walkers = 10          # Number of parallel walkers
            self.mcmc_burnin = 100          # Number of burn-in iterations
            self.mcmc_samples = 100         # Number of samples
            self.mcmc_gamma = 0.99          # Adaptive MCMC proposal memory
            self.mcmc_thin = 1              # MCMC trace thinning
        else:
            self.trace = orig.trace

            self.template_psf = orig.template_psf
            self.template_resampler = orig.template_resampler
            self.template_cache_resolution = orig.template_cache_resolution
            self.template_wlim = orig.template_wlims
            self.template_wlim_buffer = orig.template_wlim_buffer

            self.cache_templates = orig.cache_templates

            self.rv_0 = orig.rv_0
            self.rv_bounds = orig.rv_bounds
            self.rv_prior = orig.rv_prior
            self.rv_step = orig.rv_step

            self.use_flux_corr = orig.use_flux_corr
            self.flux_corr = orig.flux_corr

            self.spec_norm = orig.spec_norm
            self.temp_norm = orig.temp_norm

            self.use_mask = orig.use_mask
            self.mask_bits = orig.mask_bits
            self.use_error = orig.use_error
            self.use_weight = orig.use_weight

            self.max_iter = orig.max_iter

            self.mcmc_step = orig.mcmc_step
            self.mcmc_walkers = orig.mcmc_walkers
            self.mcmc_burnin = orig.mcmc_burnin
            self.mcmc_samples = orig.mcmc_samples
            self.mcmc_thin = orig.mcmc_thin
            self.mcmc_thin = orig.mcmc_gamma

        self.reset()

    def reset(self):
        self.template_cache = ReadOnlyCache()
        self.flux_corr_basis_cache = None
        self.flux_corr_basis_size = None

    def add_args(self, config, parser):
        Parameter('rv').add_args(parser)

        parser.add_argument('--flux-corr', action='store_true', dest='flux_corr', help='Do flux correction.\n')
        parser.add_argument('--no-flux-corr', action='store_false', dest='flux_corr', help='No flux correction.\n')
        parser.add_argument('--flux-corr-deg', type=int, help='Degree of flux correction polynomial.\n')

        parser.add_argument('--resampler', type=str, choices=list(RVFit.RESAMPLERS.keys()), default='fluxcons', help='Template resampler.\n')

        parser.add_argument('--mask', action='store_true', dest='use_mask', help='Use mask from spectra.\n')
        parser.add_argument('--no-mask', action='store_false', dest='use_mask', help='Do not use mask from spectra.\n')
        parser.add_argument('--mask-bits', type=int, help='Bit mask.\n')

        parser.add_argument('--mcmc-walkers', type=int, help='Number of MCMC walkers (min number of params + 1).\n')
        parser.add_argument('--mcmc-burnin', type=int, help='Number of MCMC burn-in samples.\n')
        parser.add_argument('--mcmc-samples', type=int, help='Number of MCMC samples.\n')
        parser.add_argument('--mcmc-thin', type=int, help='MCMC chain thinning interval.\n')
        parser.add_argument('--mcmc-gamma', type=float, help='Adaptive MC gamma.\n')

    def init_from_args(self, script, config, args):
        if self.trace is not None:
            self.trace.init_from_args(script, config, args)

        rv = Parameter('rv')
        rv.init_from_args(args)
        self.rv_0 = rv.value                        # RV initial guess
        self.rv_bounds = [rv.min, rv.max]           # Find RV between these bounds
        self.rv_prior = rv.get_dist()
        self.rv_step = None                         # RV step size for MCMC sampling


        self.use_flux_corr = get_arg('flux_corr', self.use_flux_corr, args)

        # TODO: add more options for flux correction model
        self.flux_corr = PolynomialFluxCorrection()
        self.flux_corr.degree = get_arg('flux_corr_deg', self.flux_corr.degree, args)

        # Use Interp1dResampler when template PSF accounts for pixelization
        resampler = get_arg('resampler', None, args)
        if resampler is None:
            pass
        elif resampler in RVFit.RESAMPLERS:
            self.template_resampler = RVFit.RESAMPLERS[resampler]()
        else:
            raise NotImplementedError()

        self.use_mask = get_arg('use_mask', self.use_mask, args)
        self.mask_bits = get_arg('mask_bits', self.mask_bits, args)
        self.use_error = get_arg('use_error', self.use_error, args)
        self.use_weight = get_arg('use_weight', self.use_weight, args)

        self.mcmc_walkers = get_arg('mcmc_walkers', self.mcmc_walkers, args)
        self.mcmc_burnin = get_arg('mcmc_burnin', self.mcmc_burnin, args)
        self.mcmc_samples = get_arg('mcmc_samples', self.mcmc_samples, args)
        self.mcmc_thin = get_arg('mcmc_thin', self.mcmc_thin, args)

    def create_trace(self):
        return RVFitTrace()

    def get_normalization(self, spectra, templates, rv_0=None):
        # Calculate a normalization factor for the spectra, as well as
        # the templates assuming an RV=0 from the median flux. This is
        # just a factor to bring spectra to the same scale and avoid
        # very large numbers in the Fisher matrix

        rv_0 = rv_0 if rv_0 is not None else self.rv_0

        s = []
        t = []
        for arm in spectra:
            for ei, spec in enumerate(spectra[arm] if isinstance(spectra[arm], list) else [spectra[arm]]):
                spec = self.process_spectrum(arm, ei, spec)

                # TODO: use mask from spectrum

                s.append(spec.flux[~np.isnan(spec.flux) & (spec.flux > 0)])
            
            if self.template_psf is not None:
                psf = self.template_psf[arm]
            else:
                psf = None

            if self.template_wlim is not None:
                wlim = self.template_wlim[arm]
            else:
                wlim = None
            
            if rv_0 is not None:
                rv = rv_0
            else:
                rv = 0.0

            temp = self.process_template(arm, templates[arm], spec, rv, psf=psf, wlim=wlim)
            t.append(temp.flux)

        spec_flux = np.concatenate(s)
        temp_flux = np.concatenate(t)

        return np.median(spec_flux), np.median(temp_flux)

    def process_spectrum(self, arm, i, spectrum):
        # Preprocess the spectrum to calculate the likelihood. This step
        # currently consist only of normalizing the flux with a factor.

        spec = spectrum.copy()
        if self.spec_norm is not None:
            spec.multiply(1.0 / self.spec_norm)

        if self.trace is not None:
            self.trace.on_process_spectrum(arm, i, spectrum, spec)

        return spec
    
    def determine_wlim(self, spectra, rv_bounds=None):
        # Determine wavelength limits necessary for the template LSF convolution

        if rv_bounds is not None:
            zmin = Physics.vel_to_z(rv_bounds[0])
            zmax = Physics.vel_to_z(rv_bounds[1])
        else:
            zmin = zmax = 1

        self.template_wlim = {}
        for k in spectra:
            wmin, wmax = None, None
            for s in spectra[k] if isinstance(spectra[k], list) else [ spectra[k] ]:
                w = s.wave[0] * (1 + zmin)
                wmin = w if wmin is None else min(wmin, w)

                w = s.wave[-1] * (1 + zmax)
                wmax = w if wmax is None else max(wmax, w)

            # Buffer the wave limits a little bit
            self.template_wlim[k] = (wmin - self.template_wlim_buffer, wmax + self.template_wlim_buffer)
    
    def process_template_impl(self, arm, template, spectrum, rv, psf=None, wlim=None):
        # 1. Make a copy, not in-place update
        t = template.copy()

        # 2. Shift template to the desired RV
        t.set_rv(rv)

        # 3. If a PSF if provided, convolve down template to the instruments
        #    resolution. Otherwise assume the template is already convolved.
        #    Not that psf is often None when the convolution is pushed down to
        #    the ModelGrid object for better cache performance.
        if psf is not None:
            t.convolve_psf(psf, wlim=wlim)

        # 4. Normalize by a factor
        if self.temp_norm is not None:
            t.multiply(1.0 / self.temp_norm)

        # TODO: add continuum normalization?

        if self.trace is not None:
            self.trace.on_process_template(arm, rv, template, t)
            
        return t

    def process_template(self, arm, template, spectrum, rv, psf=None, wlim=None, diff=False, resample=True, renorm=False, flux_corr=False, a=None):
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

        temp = None

        # If `cache_templates` is True, we can look up a template that's already been
        # convolved down with the PSF. Otherwise we just shift the template and do the
        # convolution on the fly.
        if self.cache_templates:
            # Quantize RV to the requested cache resolution
            rv_q = np.floor(rv / self.template_cache_resolution) * self.template_cache_resolution
            key = (template, rv_q)

            # Look up template in cache and shift to from quantized RV to actual RV
            if self.template_cache.is_cached(key):
                temp = self.template_cache.get(key)
                if self.trace is not None:
                    self.trace.on_template_cache_hit(template, rv_q, rv)
            else:
                temp = self.process_template_impl(arm, template, spectrum, rv_q, psf=psf, wlim=wlim)
                if self.trace is not None:
                    self.trace.on_template_cache_miss(template, rv_q, rv)

                self.template_cache.push(key, temp)

            # Shift from quantized RV to requested RV
            temp = temp.copy()
            temp.set_restframe()
            temp.set_redshift(Physics.vel_to_z(rv))
        else:
            # Compute convolved template from scratch
            temp = self.process_template_impl(arm, template, spectrum, rv, psf=psf, wlim=wlim)

        # Resample template to the binning of the observation
        if resample:
            temp.apply_resampler(self.template_resampler, spectrum.wave, spectrum.wave_edges)

        # Optionally apply flux correction. Note that a = None during the fitting process,
        # this feature is provided for evaluating the results only, so it's okay to
        # calculate the basis function on the fly.
        if a is not None:
            if flux_corr:
                # Evaluate the basis function
                f = self.flux_corr.get_basis_callable()
                basis = f(temp.wave)
                temp.multiply(np.dot(basis, a))
            else:
                temp.multiply(a)

        # Normalize template to match the flux scale of the fitted spectrum
        if renorm and self.spec_norm is not None:
            temp.multiply(self.spec_norm)

        if self.trace is not None:
            self.trace.on_resample_template(arm, rv, spectrum, template, temp)

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
        f = self.flux_corr.get_basis_callable()
        for k in spectra:
            basis[k] = []
            for spec in spectra[k] if isinstance(spectra[k], list) else [spectra[k]]:
                basis[k].append(f(spec.wave))
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

        if self.trace is not None:
            trace_state = RVFitTraceState()

        if not self.use_flux_corr:
            bases = None
            phi = np.zeros(rvv.shape)
            chi = np.zeros(rvv.shape)
            ndf = np.zeros(rvv.shape)
        else:
            # TODO: move this outside of function and pass in as arguments
            # Evaluate the basis for each spectrum
            if self.flux_corr_basis_cache is None:
                self.flux_corr_basis_cache, self.flux_corr_basis_size = self.eval_flux_corr_basis(spectra)
            bases, basis_size = self.flux_corr_basis_cache, self.flux_corr_basis_size            
            phi = np.zeros(rvv.shape + (basis_size,))
            chi = np.zeros(rvv.shape + (basis_size, basis_size))
            ndf = np.zeros(rvv.shape)
                
        # For each value of rv0, sum up log_L contributions from spectrum - template pairs
        for i in range(rvv.size):
            if self.trace is not None:
                trace_state.reset()

            for arm in spectra:
                if not isinstance(spectra[arm], list):
                    specs = [spectra[arm]]
                else:
                    specs = spectra[arm]

                # This is a generic call to preprocess the template which might or
                # might not include a convolution, depending on the RVFit implementation.
                # When template convolution is pushed down to the model grid to support
                # caching, convolution is skipped by the derived classes such as
                # ModelGridRVFit
                psf = self.template_psf[arm] if self.template_psf is not None else None
                wlim = self.template_wlim[arm] if self.template_wlim is not None else None
                
                for ei in range(len(specs)):
                    # TODO: Make sure template is not double-convolved in normal RVFit
                    spec = self.process_spectrum(arm, ei, specs[ei])
                    temp = self.process_template(arm, templates[arm], spec, rvv[i], psf=psf, wlim=wlim, diff=True)
                    
                    # Determine mask
                    mask = None
                    if self.use_mask and spec.mask is not None:
                        # TODO: allow specifying a bitmas
                        mask = spec.mask_as_bool(bits=self.mask_bits)
                    
                    if mask is None:
                        mask = np.full_like(spec.wave, True, dtype=bool)
                    
                    # Mask out nan values which might occur if spectrum mask is not properly defined
                    mask &= ~np.isnan(spec.wave)
                    mask &= ~np.isnan(spec.flux)
                    mask &= ~np.isnan(spec.flux_err)

                    mask &= ~np.isnan(temp.flux)

                    # Flux error
                    if self.use_error and spec.flux_err is not None:
                        sigma2 = spec.flux_err ** 2
                        mask &= ~np.isnan(sigma2)
                        
                        # TODO: add option to set this limit
                        # Mask out bins where sigma2 is unusually small
                        # Here we assume that flux is normalized in the unity range
                        mask &= sigma2 > 1e-5
                    else:
                        sigma2 = None

                    # Weight (optional)
                    if self.use_weight and temp.weight is not None:
                        weight = temp.weight / temp.weight.sum() * temp.weight.size
                        mask &= ~np.isnan(weight)
                    else:
                        weight = None

                    # Flux correction (optional)
                    if bases is not None:
                        basis = bases[arm][ei]
                        mask &= ~np.any(np.isnan(basis), axis=-1)       # Be cautious in case any item in wave is nan
                    else:
                        basis = None

                    # Verify that the mask is not empty or too few points to fit
                    if mask.sum() < 10:
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

                    if basis is None:
                        pass
                    else:
                        pp = pp[:, None] * basis[mask, :]
                        cc = cc[:, None, None] * np.matmul(basis[mask, :, None], basis[mask, None, :])

                    phi[i] += np.sum(pp, axis=0)
                    chi[i] += np.sum(cc, axis=0)

                    # Degrees of freedom
                    ndf[i] += mask.sum()

                    if self.trace is not None:
                        trace_state.append(arm, spec, temp, sigma2, weight, mask, basis)

            if not self.use_flux_corr:
                ndf[i] -= 1
            else:
                ndf[i] -= basis_size

            # Only trace when all arms are fitted but for each rv
            if self.trace is not None:
                # First dimension must index rv items
                log_L = self.eval_log_L(phi[np.newaxis, i], chi[np.newaxis, i])
                self.trace.on_eval_phi_chi(rvv[i], trace_state.spectra, trace_state.templates, trace_state.bases, trace_state.sigma2, trace_state.weights, trace_state.masks, log_L[0], phi[i], chi[i])

        if not isinstance(rv, np.ndarray):
            phi = phi[0]
            chi = chi[0]
            ndf = ndf[0]
        else:
            if bases is None:
                phi = phi.reshape(rv.shape)
                chi = chi.reshape(rv.shape)
                ndf = ndf.reshape(rv.shape)
            else:
                phi = phi.reshape(rv.shape + (basis_size,))
                chi = chi.reshape(rv.shape + (basis_size, basis_size))
                ndf = ndf.reshape(rv.shape)

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
            log_L = np.squeeze(
                np.matmul(a[..., None, :], phi[..., :, None]) - 
                0.5 * np.matmul(a[..., None, :], np.matmul(chi, a[..., :, None])))

        if self.trace is not None:
            self.trace.on_eval_log_L_a(phi, chi, a, log_L)
        
        return log_L
    
    def eval_prior(self, prior, x):
        if prior is None:
            return 0
        elif isinstance(prior, Distribution):
            return prior.log_pdf(x)
        elif isinstance(prior, Callable):
            return prior(x)
        else:
            raise NotImplementedError()
    
    def calculate_log_L(self, spectra, templates, rv, rv_prior=None, a=None):
        
        rv_prior = rv_prior if rv_prior is not None else self.rv_prior
        
        # Depending on whether the flux correction coefficients `a`
        # are supplied, calculate the log likelihood at the optimum of
        # flux correction or at the specified flux correction values.
        phi, chi, ndf = self.eval_phi_chi(spectra, templates, rv)
        if a is None:
            log_L = self.eval_log_L(phi, chi)
        else:
            log_L = self.eval_log_L_a(phi, chi, a)
        
        log_L += self.eval_prior(rv_prior, rv)

        return log_L, phi, chi, ndf
        
    def eval_flux_corr(self, spectra, templates, rv, a=None):
        # Evaluate the basis for each spectrum
        if self.flux_corr_basis_cache is None:
            self.flux_corr_basis_cache, self.flux_corr_basis_size = self.eval_flux_corr_basis(spectra)

        if a is None:
            phi, chi, ndf = self.eval_phi_chi(spectra, templates, rv)
            a = self.eval_a(phi, chi)
        
        flux_corr = {}
        for k in spectra:
            flux_corr[k] = []
            for i, spec in enumerate(spectra[k] if isinstance(spectra[k], list) else [spectra[k]]):
                flux_corr[k].append(np.dot(self.flux_corr_basis_cache[k][i], a))

        return flux_corr
    
    # def sample_log_L(self, log_L_fun, x_0,
    #                  walkers=None, burnin=None, samples=None, thin=None, cov=None):
        
    #     walkers = walkers if walkers is not None else self.mcmc_walkers
    #     burnin = burnin if burnin is not None else self.mcmc_burnin
    #     samples = samples if samples is not None else self.mcmc_samples
    #     thin = thin if thin is not None else self.mcmc_thin
    
    #     ndim = x_0.size
        
    #     # TODO: it's good to run many walkers but it increases time spent
    #     #       in burn-in a lot
    #     # nwalkers = max(walkers, 2 * x_0.size + 1)
    #     nwalkers = walkers

    #     # TODO: delete, moved to init_mcmc        
    #     # # Generate an initial state a little bit off of x_0
    #     # if step is None:
    #     #     p_0 = x_0 * (1 + 0.05 * np.random.uniform(-1.0, 1.0, size=(nwalkers, ndim)))
    #     # else:
    #     #     p_0 = x_0 + np.random.uniform(-1.0, 1.0, size=(nwalkers, ndim)) * step

    #     p_0 = np.broadcast_to(x_0, (nwalkers, ndim))
        
    #     # # Make sure the initial state is inside the bounds
    #     # if bounds is not None:
    #     #     p_0 = np.where(bounds[..., 0] < p_0, p_0, bounds[..., 0])
    #     #     p_0 = np.where(bounds[..., 1] > p_0, p_0, bounds[..., 1])

    #     if cov is not None:
    #         # moves = [ emcee.moves.GaussianMove(cov), emcee.moves.StretchMove(live_dangerously=True) ]
    #         moves = [ 
    #             # emcee.moves.GaussianMove(cov),
    #             # emcee.moves.GaussianMove(10 * cov),
    #             # emcee.moves.GaussianMove(0.1 * cov)
    #             emcee.moves.GaussianMove(np.diag(cov), mode="sequential"),
    #         ]
    #     else:
    #         # moves = [ emcee.moves.StretchMove(live_dangerously=True), emcee.moves.WalkMove(live_dangerously=True) ]
    #         moves = [ emcee.moves.StretchMove(live_dangerously=True) ]
            
    #     sampler = emcee.EnsembleSampler(nwalkers, ndim, log_L_fun, moves=moves)

    #     # Run the burn-in with the original proposal
    #     # Allocate array for the adaptive run
    #     state = sampler.run_mcmc(p_0, burnin, skip_initial_state_check=True)

    #     batch = 100
    #     nbatch = samples // batch
    #     ndim = sampler.chain.shape[-1]

    #     sampler.reset()

    #     # Periodically update the proposal distribution
        
    #     for b in range(nbatch):
    #         state = sampler.run_mcmc(state, batch, skip_initial_state_check=True)

    #         # if np.mean(sampler.acceptance_fraction) < 1e-2:
    #         #     C = moves[0].get_proposal.scale * 0.8
    #         # else:
    #         #     x = sampler.chain.reshape(-1, ndim)
    #         #     m = np.mean(x, axis=0)
    #         #     C = np.matmul((x - m).T, x - m) / x.shape[0]
    #         # # # TODO: add some logic to test if C is a valid proposal
    #         # moves[0].get_proposal.scale = (2.38**2 / ndim) * C

    #     if thin is not None:
    #         s = np.s_[..., ::thin, :]
    #     else:
    #         s = ()

    #     # shape: (chains, samples, params)
    #     return sampler.chain[s], sampler.lnprobability[s]
        
    #region Fisher matrix evaluation

    # There are multiple ways of evaluating the Fisher matrix. Functions
    # with _full_ return the full Fisher matrix including matrix elements for
    # the flux correction coefficients and rv. Functions with _rv_ return
    # the (single) matrix element corresponding to the RV only. Functions with
    # _hessian depend on the numerical evaluation of the Hessian with respect
    # to the parameters of the fitted model.

    def get_packing_functions(self, mode='full'):
        if mode == 'full' or mode == 'a_rv':
            def pack_params(a, rv):
                rv = np.atleast_1d(rv)
                if rv.size > 1:
                    rv = np.reshape(rv, (-1,) + rv.shape)

                a = np.atleast_1d(a)
                a = np.reshape(a, (-1,) + rv.shape[1:])

                return np.concatenate([a, rv])

            def unpack_params(a_rv):
                a = a_rv[:-1]
                if a.ndim == 2:
                    a = np.squeeze(a)
                elif a.size == 1:
                    a = a.item()

                rv = a_rv[-1]
                if rv.size == 1:
                    rv = rv.item()

                return a, rv
            
            def pack_bounds(a_bounds, rv_bounds):
                if a_bounds is None:
                    raise NotImplementedError()
                else:
                    bounds = a_bounds

                bounds += [ rv_bounds ]

                return bounds
        elif mode == 'rv':
            def pack_params(rv):
                return np.atleast_1d(rv)

            def unpack_params(params):
                rv = params
                if rv.size == 1:
                    rv = rv.item()
                return rv
            
            def pack_bounds(rv_bounds):
                return [ rv_bounds ]
        else:
            raise NotImplementedError()
        
        return pack_params, unpack_params, pack_bounds

    def get_objective_function(self, spectra, templates, rv_prior, mode='full'):
        # Return the objection function and parameter packing/unpacking for optimizers
        # pack_params: convert individual arguments into a single 1d array
        # unpack_params: get individual arguments from 1d array
        # pack_bounds: pack parameters bounds into a list of tuples
        # log_L: evaluate the log likelihood

        pack_params, unpack_params, pack_bounds = self.get_packing_functions(mode=mode)

        if mode == 'full' or mode == 'a_rv':
            def log_L(a_rv):
                a, rv = unpack_params(a_rv)
                log_L, _, _, _ = self.calculate_log_L(spectra, templates, rv, rv_prior=rv_prior, a=a)
                return log_L
        elif mode == 'rv':
            def log_L(params):
                rv = unpack_params(params)
                log_L, _, _, _ = self.calculate_log_L(spectra, templates, rv, rv_prior=rv_prior)
                return log_L
        else:
            raise NotImplementedError()
        
        return log_L, pack_params, unpack_params, pack_bounds
    
    def get_bounds_array(self, bounds):
        if bounds is not None:
            bb = []
            for b in bounds:
                if b is None:
                    bb.append((-np.inf, np.inf))
                else:
                    bb.append((
                        b[0] if b[0] is not None else -np.inf,
                        b[1] if b[1] is not None else np.inf,
                    ))
            return np.array(bb)
        else:
            return None

    def eval_F(self, spectra, templates, rv_0=None, rv_bounds=None, rv_prior=None, step=None, mode='full', method='hessian'):
        # Evaluate the Fisher matrix around the provided rv_0. The corresponding
        # a_0 best fit flux correction will be evaluated at the optimum.
        # The Hessian will be calculated wrt either RV only, or rv and the
        # flux correction coefficients. Alternatively, the covariance
        # matrix will be determined using MCMC.

        rv_0 = rv_0 if rv_0 is not None else self.rv_0
        rv_bounds = rv_bounds if rv_bounds is not None else self.rv_bounds
        rv_prior = rv_prior if rv_prior is not None else self.rv_prior

        # Get objective function
        log_L, pack_params, unpack_params, pack_bounds = self.get_objective_function(
            spectra, templates, rv_prior, mode=mode)

        if mode == 'full' or mode == 'a_rv':
            # Calculate a_0
            phi_0, chi_0, ndf_0 = self.eval_phi_chi(spectra, templates, rv_0)
            a_0 = self.eval_a(phi_0, chi_0)
            x_0 = pack_params(a_0, rv_0)
            bounds = pack_bounds(a_0.size * [(-np.inf, np.inf)], rv_bounds)
        elif mode == 'rv':
            x_0 = pack_params(rv_0)
            bounds = pack_bounds(rv_bounds)
        else:
            raise NotImplementedError()
        
        bounds = self.get_bounds_array(bounds)
                
        return self.eval_F_dispatch(x_0, log_L, step, method, bounds)

    def eval_F_dispatch(self, x_0, log_L_fun, step, method, bounds):
        if method == 'hessian':
            return self.eval_F_hessian(x_0, log_L_fun, step)
        elif method == 'emcee':
            return self.eval_F_emcee(x_0, log_L_fun, step, bounds)
        elif method == 'sampling':
            return self.eval_F_sampling(x_0, log_L_fun, step, bounds)
        else:
            raise NotImplementedError()

    def eval_F_hessian(self, x_0, log_L_fun, step):
        # Evaluate the Fisher matrix by calculating the Hessian numerically

        # Default step size is 1% of optimum values
        if step is None:
            step = 0.01 * x_0

        dd_log_L = nd.Hessian(log_L_fun, step=step)
        dd_log_L_0 = dd_log_L(x_0)

        return dd_log_L_0, np.linalg.inv(-dd_log_L_0)
    
    def eval_F_emcee(self, x_0, log_L_fun, step, bounds):
        # Evaluate the Fisher matrix by MC sampling around the optimum

        x, log_L = self.sample_log_L(log_L_fun, x_0=x_0)

        if self.trace is not None:
            self.trace.on_eval_F_mcmc(x, log_L)

        # This is already the covanriance, we need its inverse here
        C = np.cov(x.T)
        return -np.linalg.inv(C), C
    
    def eval_F_sampling(self, x_0, log_L_fun, step, bounds):
        # Sample a bunch of RVs around the optimum and fit a parabola
        # to obtain the error of RV            

        # Default step size is 1% of optimum values
        if step is None:
            step = 1.0e-2 * x_0

        # TODO: what do we do with the original bounds?
        nbounds = np.stack([x_0 - step, x_0 + step], axis=-1)
        nbounds[..., 0] = np.where(bounds[..., 0] > nbounds[..., 0], bounds[..., 0], nbounds[..., 0])
        nbounds[..., 1] = np.where(bounds[..., 1] < nbounds[..., 1], bounds[..., 1], nbounds[..., 1])
            
        x = np.random.uniform(nbounds[..., 0], nbounds[..., 1], size=(500,) + x_0.shape)
        log_L = np.empty(x.shape[:-1])
        for ix in range(x.shape[0]):
            log_L[ix] = log_L_fun(np.atleast_1d(x[ix]))

        n = x_0.size
        if n == 1:
            p = np.polyfit(x, log_L, 2)
            return np.array([[2.0 * p[0]]]), np.array([[-0.5 / p[0]]])
        else:
            # Fit n-D quadratic formula
            # Create the design matrix and solve the least squares problem
            pf = PolynomialFeatures(degree=2)
            X = pf.fit_transform(x - x_0)
            p = np.linalg.solve(X.T @ X, X.T @ (log_L - np.min(log_L)))

            # Construct the hessian from the p parameters
            F = np.empty((n, n), dtype=x.dtype)
            for pi, pw in zip(p[n + 1:], pf.powers_[n + 1:]):
                idx = np.where(pw)[0]       # Index of item in the Hessian
                if idx.size == 1:
                    # Diagonal item:
                    F[idx[0], idx[0]] = 2 * pi
                else:
                    # Mixed item
                    F[idx[0], idx[1]] = F[idx[1], idx[0]] = pi

            return F, np.linalg.inv(-F)


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
            phi, chi, ndf = self.eval_phi_chi(spectra, templates, rv)
            return pack_phi_chi(phi, chi)

        # Calculate a_0
        phi_0, chi_0, ndf_0 = self.eval_phi_chi(spectra, templates, rv_0)
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

        for arm in spectra:
            temp = templates[arm]
            for ei, spec in enumerate(spectra[arm] if isinstance(spectra[arm], list) else [spectra[arm]]):
                spec = self.process_spectrum(arm, ei, spec)

                psf = self.template_psf[arm] if self.template_psf is not None else None
                wlim = self.template_wlim[arm] if self.template_wlim is not None else None

                temp0 = self.process_template(arm, temp, spec, rv, psf=psf, wlim=wlim)
                temp1 = self.process_template(arm, temp, spec, rv + step, psf=psf, wlim=wlim)
                temp2 = self.process_template(arm, temp, spec, rv - step, psf=psf, wlim=wlim)

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
            phi, chi, ndf = self.eval_phi_chi(spectra, templates, rv)
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
                rv[0] - 1,
                0.2 * (rv[-1] - rv[0]),
                y0.min()
            ),
            (
                5 * np.max(y0),
                rv[-1] + 1,
                5.0 * (rv[-1] - rv[0]),
                y0.min() + 4 * (y0.max() - y0.min())
            )
        ]
        
        pp, pcov = curve_fit(self.lorentz, rv, y0, p0=p0, bounds=bb)

        return pp, pcov
    
    def sample_rv_prior(self, rv_prior, bounds=None):
        rv_0 = rv_prior.sample()
        if bounds is not None:
            if bounds[0] is not None:
                rv_0 = max(rv_0, bounds[0])
            if bounds[1] is not None:
                rv_0 = min(rv_0, bounds[1])
        return rv_0

    def guess_rv(self, spectra, templates, /, rv_bounds=(-500, 500), rv_prior=None, rv_steps=31, method='lorentz'):
        """
        Given a spectrum and a template, make a good initial guess for RV where a minimization
        algorithm can be started from.
        """

        rv_bounds = rv_bounds if rv_bounds is not None else self.rv_bounds
        rv_prior = rv_prior if rv_prior is not None else self.rv_prior
        method = method if method is not None else 'lorentz'

        rv = np.linspace(*rv_bounds, rv_steps)
        log_L, phi, chi, ndf = self.calculate_log_L(spectra, templates, rv, rv_prior=rv_prior)
        a = self.eval_a(phi, chi)

        if method == 'lorentz':
            # Mask out infs here in case the prior is very narrow
            mask = (~np.isnan(log_L) & ~np.isinf(log_L))    
            if mask.sum() < 10:
                raise Exception("Too few values to guess RV. Consider changing the bounds.")     
            
            pp, pcov = self.fit_lorentz(rv[mask], log_L[mask])
            rv_guess = pp[1]

            # The maximum of the Lorentz curve might be outside the bounds
            if rv_bounds is not None and rv_bounds[0] is not None:
                rv_guess = max(rv_guess, rv_bounds[0])
            if rv_bounds is not None and rv_bounds[1] is not None:
                rv_guess = min(rv_guess, rv_bounds[1])
                
            if self.trace is not None:
                self.trace.on_guess_rv(rv, log_L, rv_guess, self.lorentz(rv, *pp), 'lorentz', pp, pcov)
        elif method == 'max':
            rv_guess = rv[np.argmax(log_L)]
            if self.trace is not None:
                self.trace.on_guess_rv(rv, log_L, rv_guess, None, 'max', None, None)
        else:
            raise NotImplementedError()

        return rv, log_L, a, rv_guess

    def fit_rv(self, spectra, templates, rv_0=None, rv_bounds=(-500, 500), rv_prior=None,
               method='bounded', calculate_error=True):
        """
        Given a set of spectra and templates, find the best fit RV by maximizing the log likelihood.
        Spectra are assumed to be of the same object in different wavelength ranges.

        If no initial guess is provided, rv0 is determined automatically.
        """

        assert isinstance(spectra, dict)
        assert isinstance(templates, dict)

        rv_0 = rv_0 if rv_0 is not None else self.rv_0
        rv_bounds = rv_bounds if rv_bounds is not None else self.rv_bounds
        rv_prior = rv_prior if rv_prior is not None else self.rv_prior

        # Get objective function
        log_L, pack_params, unpack_params, pack_bounds = self.get_objective_function(
            spectra, templates, rv_prior, mode='rv')

        # Cost function
        def llh(rv):
            return -log_L(pack_params(rv))

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
            rv_fit = unpack_params(out.x)

            # Calculate the flux correction coefficients at best fit values
            phi_fit, chi_fit, ndf_fit = self.eval_phi_chi(spectra, templates, rv_fit)
            a_fit = self.eval_a(phi_fit, chi_fit)

            # If tracing, evaluate the template at the best fit RV.
            # TODO: can we cache this for better performance?
            if self.trace is not None:
                tt = {}
                for arm in spectra:
                    for spec in spectra[arm] if isinstance(spectra[arm], list) else [spectra[arm]]:
                        temp = templates[arm]
                        
                        psf = self.template_psf[arm] if self.template_psf is not None else None
                        wlim = self.template_wlim[arm] if self.template_wlim is not None else None
                    
                        # This is a generic call to preprocess the template which might or
                        # might not include a convolution, depending on the RVFit implementation.
                        # When template convolution is pushed down to the model grid to support
                        # caching, convolution is skipped by the derived classes such as
                        # ModelGridRVFit
                        t = self.process_template(arm, temp, spec, rv_fit, psf=psf, wlim=wlim)
                        tt[arm] = t
                self.trace.on_fit_rv(rv_fit, spectra, tt)
        else:
            raise Exception(f"Could not fit RV using `{method}`")
        
        # Calculate the error from the Fisher matrix
        if calculate_error:
            _, C = self.eval_F(spectra, templates, rv_fit, rv_bounds=rv_bounds, rv_prior=rv_prior, mode='rv', method='hessian')

            with np.errstate(invalid='warn'):
                rv_err = np.sqrt(C[-1, -1])         # sigma
        else:
            rv_err = None
            C = None

        return RVFitResults(rv_fit=rv_fit, rv_err=rv_err,
                            a_fit=a_fit, a_err=np.full_like(a_fit, np.nan),
                            cov=C)

    def run_mcmc(self, spectra, templates, *,
                 rv_0=None, rv_bounds=(-500, 500), rv_prior=None, rv_step=None,
                 walkers=None, burnin=None, samples=None, thin=None):
        """
        Given a set of spectra and templates, sample from the posterior distribution of RV.

        If no initial guess is provided, an initial state is generated automatically.
        """

        assert isinstance(spectra, dict)
        assert isinstance(templates, dict)

        rv_0 = rv_0 if rv_0 is not None else self.rv_0
        rv_bounds = rv_bounds if rv_bounds is not None else self.rv_bounds
        rv_prior = rv_prior if rv_prior is not None else self.rv_prior

        if rv_0 is None:
            _, _, _, rv_0 = self.guess_rv(spectra, None, 
                                          rv_bounds=rv_bounds, rv_prior=rv_prior)

        # Get objective function
        log_L_fun, pack_params, unpack_params, pack_bounds = self.get_objective_function(
            spectra, templates, rv_prior, mode='rv')
        
        # Initial values
        if rv_0 is not None:
            x_0 = pack_params(rv_0)
        else:
            x_0 = None

        bounds = pack_bounds(rv_bounds)
        bounds = self.get_bounds_array(bounds)
        
        # Run sampling
        x, log_L =  self.sample_log_L(log_L_fun, x_0,
            walkers=walkers, burnin=burnin, samples=samples, thin=thin)
        
        rv = unpack_params(x.T)
        
        return RVFitResults(rv_mcmc=rv, log_L_mcmc=log_L)
    
    @staticmethod
    def minimize_gridsearch(fun, bounds, nstep=100):
        # Find minimum with direct grid search. For testing only

        x = np.linspace(bounds[0], bounds[1], nstep)

        # TODO: rewrite log_l function to take vectorized input
        # raise NotImplementedError
        # y = fun(x)

        y = np.empty_like(x)
        for i in range(x.size):
            y[i] = fun(x[i])

        mi = np.argmin(y)
        
        class Result(): pass
        out = Result()
        out.success = True
        out.x = x[mi]

        return out

    def calculate_rv_bouchy(self, spectra, templates, rv_0):
        # Calculate a delta V correction by the method of Bouchy (2001)
        # TODO: we might need the flux correction coefficients here

        phi_0, chi_0, ndf_0 = self.eval_phi_chi(spectra, templates, rv_0)
        a_0 = self.eval_a(phi_0, chi_0)

        nom = 0
        den = 0
        sumw = 0

        for arm in spectra:
            if not isinstance(spectra[arm], list):
                specs = [spectra[arm]]
            else:
                specs = spectra[arm]

            psf = self.template_psf[arm] if self.template_psf is not None else None
            wlim = self.template_wlim[arm] if self.template_wlim is not None else None
            
            for ei in range(len(specs)):
                spec = self.process_spectrum(arm, ei, specs[ei])
                temp = self.process_template(arm, templates[arm], spec, rv_0, psf=psf, wlim=wlim, diff=True)

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