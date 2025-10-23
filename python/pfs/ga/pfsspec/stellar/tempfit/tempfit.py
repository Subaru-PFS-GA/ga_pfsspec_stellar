from deprecated import deprecated
from typing import Callable
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.optimize import minimize_scalar
from scipy.ndimage import binary_dilation
from scipy.interpolate import interp1d
from sklearn.preprocessing import PolynomialFeatures
import numdifftools as nd

from pfs.ga.pfsspec.core.util.copy import safe_deep_copy
from pfs.ga.pfsspec.core.util.args import get_arg
from pfs.ga.pfsspec.core import Physics
from pfs.ga.pfsspec.core.sampling import MCMC
from pfs.ga.pfsspec.core.caching import ReadOnlyCache
from pfs.ga.pfsspec.core.obsmod.resampling import RESAMPLERS
from pfs.ga.pfsspec.core.sampling import Parameter, Distribution
from .tempfitflag import TempFitFlag
from .tempfittrace import TempFitTrace
from .tempfittracestate import TempFitTraceState
from .tempfitstate import TempFitState
from .tempfitresults import TempFitResults
from .fluxcorr import FluxCorr

from .setup_logger import logger

class TempFit():
    """
    This is the base class for basic RV estimation based on template fitting using a non-linear
    maximum likelihood or maximum significance method. Templates are optionally convolved with
    the instrumental LSF. The class supports various methods to determine the uncertainty of
    RV estimates.

    The input should be presented as a dictionary of lists of spectra with the dictionary keys
    being the various spectrograph arms whereas the spectra within the lists are the single
    exposures.

    When the objective is fitting the template parameters or extinction, use ModelGridTempFit instead.

    Functions of the class often have parameters that are the same as variables defined on the
    class itself. This is to simplify function calls but also to allow overriding some parameters
    defined on the class. When an optional argument is passed to the function, it will override
    the class variable of the same name.

    Depending on the continuum correction object passed to the constructor, the class can fit
    fluxed templates with a multiplicative flux correction function, or normalize the observations
    with a continuum model and fit the absorption lines with continuum-normalized templates.
    
    Derived classes, with the use of the appropriate mix-ins are capable of fitting templates
    with a multiplicative flux correction function or a continuum model. In either case, a
    multiplicative scalar amplitude is fitted, either

    Terminology: amplitudes are scalar factors to be applied to observed spectra to scale to
    the templates. Coefficients are the flux correction or continuum fit model parameters.

    Variables
    ---------
    trace : TempFitTrace
        Trace object that implements tracing callbacks to collect debug info
    correction_model : FluxCorr, ContNorm
        Flux correction or continuum normalization model
    template_psf : dict of Psf
        Instrumental line spread function to convolve templates with, for each arm
    template_resampler : Resampler
        Resampler to resample templates to the instrument pixels. FluxConservingResampler is the default
        but other resamplers can be used.
    template_cache_resolution : float
        Cache templates with this resolution in RV. Templates are quantized to this resolution and
        stored in a cache to avoid recomputing the same template multiple times.
    template_wlim : dict of tuple
        Model wavelength limits for each spectrograph arm when performing the convolution.
    template_wlim_buffer : float
        Wavelength buffer in A for each spectrograph arm when performing the convolution.
    cache_templates : bool
        Set to True to enable caching PSF-convolved templates.
    rv_0 : float
        Initial guess for the radial velocity. Will be estimated automatically if not set.
    rv_fixed : bool
        When True, do not optimize for RV but fix its value to the initial value (or the value
        determined by `guess_rv`).
    rv_bounds : tuple
        Find RV between these bounds.
    rv_prior : Distribution or callable
        Prior distribution for the RV.
    rv_step : float
        Initial step size for the MCMC sampling of the RV and estimating the uncertainty from
        the Hessian numerically.
    amplitude_per_arm : bool
        Estimate flux multiplier for each arm independently.
    amplitude_per_fiber : bool
        Estimate flux multiplier for each fiber independently (reserved for future use).
    amplitude_per_exp : bool
        Estimate flux multiplier for each exposure independently.
    spec_norm : float
        Spectrum (observation) normalization factor. Use `get_normalization` to initialize it.
    temp_norm : float
        Template normalization factor. Use `get_normalization` to initialize it.
    wave_include: list of tuples
        List of wavelength ranges to fit the template to. If None or empty, fit the entire spectrum.
    wave_exclude: list of tuples
        List of wavelength ranges to exclude from the fit.
    use_mask : bool
        Use mask from spectrum, if available.
    mask_bits : int
        Mask bits to observe when converting spectrum flags to a boolean mask (None means any).
    use_error : bool
        Use flux error from spectrum to weight fitting, if available.
    use_weight : bool
        Use weight from template for fitting, if available. Templates can define weights to
        modify the likelihood function.
    max_iter : int
        Maximum number of iterations of the optimization algorithm.
    mcmc_walkers : int
        Number of parallel walkers for the MCMC sampling.
    mcmc_burnin : int
        Number of burn-in iterations for the MCMC sampling.
    mcmc_samples : int
        Number of samples for the MCMC sampling.
    mcmc_thin : int
        MCMC chain thinning interval.
    mcmc_gamma : float
        Adaptive MCMC proposal memory.
    """

    def __init__(
            self,
            trace=None,
            correction_model=None,
            extinction_model=None,
            orig=None
    ):
        
        """
        Initializes the template fitting problem.

        Parameters
        ----------
        trace : TempFitTrace
            Trace object that implements tracing callbacks to collect debug info
        correction_model : FluxCorr, ContNomr
            Flux correction or continuum normalization model
        orig : TempFit
            Original object to copy from
        """
        
        if not isinstance(orig, TempFit):
            self.trace = trace                              # Collect debug info
            self.correction_model = correction_model        # Flux correction or continuum fitting model
            self.extinction_model = extinction_model    # Extinction model

            self.template_psf = None                                # Dict of psf to downgrade templates with
            self.template_resampler = RESAMPLERS['fluxcons']()      # Resample template to instrument pixels
            self.template_cache_resolution = 50                     # Cache templates with this resolution in RV
            self.template_wlim = None                               # Model wavelength limits for each spectrograph arm to limit convolution
            self.template_wlim_buffer = 100                         # Wavelength buffer in A, depends on line spread function
            
            self.cache_templates = True     # Cache PSF-convolved templates

            self.rv_0 = None                # RV initial guess
            self.rv_fixed = False           # Fix RV to the initial guess or default value
            self.rv_bounds = None           # Find RV between these bounds
            self.rv_prior = None
            self.rv_step = None             # RV step size for MCMC sampling

            # Variables controlling flux correction

            self.amplitude_per_arm = False      # Estimate flux multiplier for each arm independently
            self.amplitude_per_fiber = False
            self.amplitude_per_exp = False

            # Other variables

            self.spec_norm = None           # Spectrum (observation) normalization factor
            self.temp_norm = None           # Template normalization factor

            self.wave_include = None        # List of wavelength ranges to fit the template to
            self.wave_exclude = None        # List of wavelength ranges to exclude from the fit
            self.use_mask = True            # Use mask from spectrum, if available
            self.mask_bits = None           # Mask bits (None means any)
            self.use_error = True           # Use flux error from spectrum, if available
            self.use_weight = False         # Use weight from template, if available

            self.max_iter = 1000            # Maximum number of iterations of the optimization algorithm

            self.mcmc_walkers = 10          # Number of parallel walkers
            self.mcmc_burnin = 100          # Number of burn-in iterations
            self.mcmc_samples = 100         # Number of samples
            self.mcmc_gamma = 0.99          # Adaptive MCMC proposal memory
            self.mcmc_thin = 1              # MCMC trace thinning
        else:
            self.trace = trace if trace is not None else orig.trace
            self.correction_model = correction_model if correction_model is not None else orig.correction_model
            self.extinction_model = extinction_model if extinction_model is not None else orig.extinction_model

            self.template_psf = orig.template_psf
            self.template_resampler = orig.template_resampler
            self.template_cache_resolution = orig.template_cache_resolution
            self.template_wlim = orig.template_wlim
            self.template_wlim_buffer = orig.template_wlim_buffer

            self.cache_templates = orig.cache_templates

            self.rv_0 = orig.rv_0
            self.rv_bounds = orig.rv_bounds
            self.rv_prior = orig.rv_prior
            self.rv_step = orig.rv_step

            self.amplitude_per_arm = orig.amplitude_per_arm
            self.amplitude_per_fiber = orig.amplitude_per_fiber
            self.amplitude_per_exp = orig.amplitude_per_exp

            self.spec_norm = orig.spec_norm
            self.temp_norm = orig.temp_norm

            self.wave_include = orig.wave_ranges
            self.wave_exclude = orig.wave_exclude
            self.use_mask = orig.use_mask
            self.mask_bits = orig.mask_bits
            self.use_error = orig.use_error
            self.use_weight = orig.use_weight

            self.max_iter = orig.max_iter

            self.mcmc_walkers = orig.mcmc_walkers
            self.mcmc_burnin = orig.mcmc_burnin
            self.mcmc_samples = orig.mcmc_samples
            self.mcmc_thin = orig.mcmc_thin
            self.mcmc_thin = orig.mcmc_gamma

        self.reset()

    def reset(self):
        """
        Reset the state of the object
        """

        self.template_cache = ReadOnlyCache()

        if self.correction_model is not None:
            self.correction_model.reset()

        if self.extinction_model is not None:
            self.extinction_model.reset()

    def add_args(self, config, parser):
        """
        Register arguments for the command line parser

        Parameters
        ----------
        config : dict
            Script configuration object
        parser : argparse.ArgumentParser
            Command line parser
        """

        Parameter('rv').add_args(parser)

        parser.add_argument('--amplitude-per-arm', action='store_true', dest='amplitude_per_arm', help='Flux correction per arm.\n')
        parser.add_argument('--amplitude-per-fiber', action='store_true', dest='amplitude_per_fiber', help='Flux correction per fiber.\n')
        parser.add_argument('--amplitude-per-exp', action='store_true', dest='amplitude_per_exp', help='Flux correction per exposure.\n')

        parser.add_argument('--resampler', type=str, choices=list(RESAMPLERS.keys()), default='fluxcons', help='Template resampler.\n')

        # TODO: add argument to specify the wavelength ranges

        parser.add_argument('--mask', action='store_true', dest='use_mask', help='Use mask from spectra.\n')
        parser.add_argument('--no-mask', action='store_false', dest='use_mask', help='Do not use mask from spectra.\n')
        parser.add_argument('--mask-bits', type=int, help='Bit mask.\n')

        parser.add_argument('--max-iter', type=int, help='Maximum number of iterations of the optimization algorithm.\n')

        parser.add_argument('--mcmc-walkers', type=int, help='Number of MCMC walkers (min number of params + 1).\n')
        parser.add_argument('--mcmc-burnin', type=int, help='Number of MCMC burn-in samples.\n')
        parser.add_argument('--mcmc-samples', type=int, help='Number of MCMC samples.\n')
        parser.add_argument('--mcmc-thin', type=int, help='MCMC chain thinning interval.\n')
        parser.add_argument('--mcmc-gamma', type=float, help='Adaptive MC gamma.\n')

        if self.correction_model is not None:
            self.correction_model.add_args(config, parser)

        if self.extinction_model is not None:
            self.extinction_model.add_args(config, parser)

    def init_from_args(self, script, config, args):
        """
        Initialize the object from command line arguments

        Parameters
        ----------
        script : Script
            Script object
        config : dict
            Script configuration object
        args : dict
            Command line arguments
        """

        if self.trace is not None:
            self.trace.init_from_args(script, config, args)

        rv = Parameter('rv')
        rv.init_from_args(args)
        
        self.rv_0 = rv.value                        # RV initial guess
        self.rv_bounds = [rv.min, rv.max]           # Find RV between these bounds
        self.rv_prior = rv.get_dist()
        self.rv_step = rv.generate_step_size(step_size_factor=0.1)     # RV step size for RV guess, MCMC sampling, etc.

        self.amplitude_per_arm = get_arg('amplitude_per_arm', self.amplitude_per_arm, args)
        self.amplitude_per_fiber = get_arg('amplitude_per_fiber', self.amplitude_per_fiber, args)
        self.amplitude_per_exp = get_arg('amplitude_per_exp', self.amplitude_per_exp, args)

        # Use Interp1dResampler when template PSF accounts for pixelization
        resampler = get_arg('resampler', None, args)
        if resampler is None:
            pass
        elif resampler in RESAMPLERS:
            self.template_resampler = RESAMPLERS[resampler]()
        else:
            raise NotImplementedError()
        
        # TODO: parse wavelength ranges from argument
        #       this needs a new type of parser, see gapipe search filters for example
        # self.wave_include = None
        # self.wave_exclude = None

        self.use_mask = get_arg('use_mask', self.use_mask, args)
        self.mask_bits = get_arg('mask_bits', self.mask_bits, args)
        self.use_error = get_arg('use_error', self.use_error, args)
        self.use_weight = get_arg('use_weight', self.use_weight, args)

        self.max_iter = get_arg('max_iter', self.max_iter, args)

        self.mcmc_walkers = get_arg('mcmc_walkers', self.mcmc_walkers, args)
        self.mcmc_burnin = get_arg('mcmc_burnin', self.mcmc_burnin, args)
        self.mcmc_samples = get_arg('mcmc_samples', self.mcmc_samples, args)
        self.mcmc_thin = get_arg('mcmc_thin', self.mcmc_thin, args)

        # Initialize the continuum correction model
        if self.correction_model is not None:
            self.correction_model.init_from_args(script, config, args)

        # Initialize the extinction model
        if self.extinction_model is not None:
            self.extinction_model.init_from_args(script, config, args)

    def create_trace(self):
        """
        Create a trace object that has the type with all necessary callbacks.
        """

        return TempFitTrace()

    def enumerate_spectra(self, spectra: dict, per_arm, per_exp, include_none=False, include_masked=False, mask_bits=None):
        """
        Enumerate all spectra in the dictionary of spectra. Return the arm,
        the exposure index, as well as an index that uniquely identifies the
        correction model given that the model is parameterized per arm and/or
        per exposure. This generator function is used to iterate over all
        amplitudes or coefficients in the correction models.

        Parameters
        ----------
        spectra : dict of Spectrum or dict of list of Spectrum
            Observed spectra
        per_arm : bool
            Whether to enumerate correction models per arm
        per_exp : bool
            Whether to enumerate correction models per exposure
        include_none : bool
            Include None values in the enumeration
        include_masked : bool
            Include spectra that are completely masked in the enumeration
        mask_bits : int
            Mask bits to observe when converting spectrum flags to a boolean mask (None means any)

        Returns
        -------
        arm : str
            Arm key
        ei : int
            Exposure index
        mi : int
            Model index
        spec : Spectrum
            Spectrum object
        """

        def is_good(spec):
            if spec is None:
                raise ValueError("Spectrum is None")
            elif include_masked:
                # Include spectra that may be completely masked
                return True
            elif spec.mask is None:
                # No mask, assume the spectrum is good
                return True
            else:
                # Only include spectra with some valid data

                # TODO: right now the mask is calculated every time the spectra are enumerated
                #       this could be cached, given that the mask_bits are the same

                mask = spec.mask_as_bool(bits=mask_bits)
                mask &= self.get_wave_mask(spec.wave, self.wave_include, self.wave_exclude)
                return mask.sum() > 0
            
        # Find those exposures where all spectra are bad in all arms
        # and exclude them from the model enumeration
        exp_exl = set()
        exp_count = np.max([ len(spectra[arm]) if isinstance(spectra[arm], list) else 1 for arm in spectra ])
        for ei in range(exp_count):
            found = False
            for arm in spectra:
                if isinstance(spectra[arm], list):
                    if ei < len(spectra[arm]):
                        spec = spectra[arm][ei]
                    else:
                        spec = None
                else:
                    spec = spectra[arm]

                if spec is not None and is_good(spec):
                    found = True
                    break

            if not found:
                exp_exl.add(ei)

        if per_arm:
            if per_exp:
                # Increase index for each arm and each exposure
                mi = 0
                for arm in spectra:
                    for ei, spec in enumerate(spectra[arm] if isinstance(spectra[arm], list) else [spectra[arm]]):
                        if spec is None and include_none:
                            yield arm, ei, None, None
                        elif spec is not None and is_good(spec):
                            yield arm, ei, mi, spec
                            mi += 1
            else:
                # Increase index for each arm but not for exposures
                # Count only arms where at least one spectrum is good
                mi = 0
                for arm in spectra:
                    found = False
                    for ei, spec in enumerate(spectra[arm] if isinstance(spectra[arm], list) else [spectra[arm]]):
                        if spec is None and include_none:
                            yield arm, ei, None, None
                        elif spec is not None and is_good(spec):
                            found = True
                            yield arm, ei, mi, spec

                    if found:
                        mi += 1
        else:
            if per_exp:
                # Increase index for each exposure but not for arms
                # The trick here is to skip the exposures which are None in every arm
                # because it would cause a singular matrix in linear flux correction models

                for arm in spectra:
                    mi = 0
                    for ei in range(exp_count):
                        spec = None
                        if isinstance(spectra[arm], list) :
                            if ei < len(spectra[arm]):
                                spec = spectra[arm][ei]
                        else:
                            spec = spectra[arm]

                        if spec is None and include_none:
                            yield arm, ei, None, None
                        elif spec is not None and is_good(spec):
                            yield arm, ei, mi, spec

                        # This will increse the model index even if the spectrum is None
                        # or not good but there is at least one good spectrum in some arm
                        if ei not in exp_exl:
                            mi += 1
            else:
                # Use a single model, regardless of the arm or exposure
                mi = 0
                for arm in spectra:
                    for ei, spec in enumerate(spectra[arm] if isinstance(spectra[arm], list) else [spectra[arm]]):
                        if spec is None and include_none:
                            yield arm, ei, None, None
                        elif spec is not None and is_good(spec):
                            yield arm, ei, mi, spec

    def get_wave_include_exclude(self):
        wave_include = []
        wave_exclude = []
        wave_include.extend(self.wave_include or [])
        wave_exclude.extend(self.wave_exclude or [])

        if self.correction_model is not None:
            wave_include.extend(self.correction_model.get_wave_include() or [])
            wave_exclude.extend(self.correction_model.get_wave_exclude() or [])

        if self.extinction_model is not None:
            wave_include.extend(self.extinction_model.get_wave_include() or [])
            wave_exclude.extend(self.extinction_model.get_wave_exclude() or [])

        return wave_include, wave_exclude
    
    def determine_wlim(self, spectra: dict, /, per_arm=True, per_exp=True,
                       rv_bounds=None, wlim_buffer=None, round_to=None):
        """
        Determine the wave limits for each arm separately or all arms together, for each
        exposure or all exposures together.

        This function does not take the masks or included and excluded wavelength ranges into
        account, only the valid values of the wavelength vectors. The wmlim is used to instantiate
        the correction models (per arm or per exposure).

        Parameters
        ----------
        spectra : dict or dict of lists
            Dictionary of spectra, keyed by arm name, optionally with multiple exposures in a list.
        per_arm : bool
            Whether to determine the wave limits separately for each arm
        per_exp : bool
            Whether to determine the wave limits separately for each exposure
        rv_bounds : tuple
            RV bounds to extend the wavelength limits with
        wlim_buffer : float
            Buffer to add to the wavelength limits

        Returns
        -------
        dict
            Dictionary of wavelength limits for each correction model index.
        """

        rv_bounds = rv_bounds if rv_bounds is not None else self.rv_bounds
        wlim_buffer = wlim_buffer if wlim_buffer is not None else self.template_wlim_buffer

        if rv_bounds is not None:
            zmin = Physics.vel_to_z(rv_bounds[0])
            zmax = Physics.vel_to_z(rv_bounds[1])
        else:
            zmin = zmax = None

        def buffer_limits(wmin, wmax):
            # Given the wavelength coverage and rv_bounds, calculate the appropriate
            # domain for the flux correction basis or continuum model

            if zmin is not None:
                wmin = wmin * (1 + zmin)

            if zmax is not None:
                wmax = wmax * (1 + zmax)

            if wlim_buffer is not None:
                wmin -= wlim_buffer
                wmax += wlim_buffer

            if round_to is not None:
                wmin = np.floor(wmin / round_to) * round_to
                wmax = np.ceil(wmax / round_to) * round_to
        
            return (wmin, wmax)
        
        # Enumerate every spectra in each arm and determine the wavelength limits for each correction model,
        # determined by the model index `mi`.
        wlim = {}
        for arm, ei, mi, spec in self.enumerate_spectra(spectra, per_arm=per_arm, per_exp=per_exp, include_none=False):
            wmin, wmax = spec.wave[~np.isnan(spec.wave)][[0, -1]]
            wmin, wmax = buffer_limits(wmin, wmax)
            
            if mi in wlim:
                wmin = min(wlim[mi][0] or wmin, wmin)
                wmax = max(wlim[mi][1] or wmax, wmax)

            wlim[mi] = (wmin, wmax)

        return wlim

    def init_correction_models(self, spectra, rv_bounds=None, force=False):
        """
        Initialize the models for flux correction or continuum fitting

        Parameters
        ----------
        spectra : dict of Spectrum or dict of list of Spectrum
            Observed spectra
        rv_bounds : tuple
            RV bounds to extend the wavelength limits with
        force : bool
            Force reinitialization of the models
        """

        rv_bounds = rv_bounds if rv_bounds is not None else self.rv_bounds

        if self.correction_model is not None:
            self.correction_model.trace = self.trace
            self.correction_model.tempfit = self

            self.correction_model.init_models(spectra, rv_bounds=rv_bounds, force=force)

    def init_correction_model(self, spectra, /, per_arm=True, per_exp=True,
                   rv_bounds=None, wlim_buffer=None, round_to=None,
                   create_model_func=None):
        """
        Create model instances for flux correction or continuum fitting for each arm
        and exposure, depending on the settings.

        The returned data structure is either a dict or lists, or model type or the
        combination of these.

        Parameters
        ----------
        spectra : dict or dict of lists
            Dictionary of spectra, keyed by arm name
        per_arm : bool
            Whether to initialize a model for each arm
        per_exp : bool
            Whether to initialize a model for each exposure
        rv_bounds : tuple
            RV bounds to extend the wavelength limits with
        wlim_buffer : float
            Buffer to add to the wavelength limits
        round_to : float
            Round the wavelength limits to this quantum
        create_model_func : callable
            Function to create the model instance

        Returns
        -------
        dict
            Dictionary of models for each arm and/or exposure, indexed by the model index.
        """

        # Determine the wavelength limits for each model
        wlim = self.determine_wlim(spectra, per_arm=per_arm, per_exp=per_exp,
                                   rv_bounds=rv_bounds, wlim_buffer=wlim_buffer, round_to=round_to)
        
        # Initialize the correction models
        models = {}
        for arm, ei, mi, spec in self.enumerate_spectra(spectra, per_arm=per_arm, per_exp=per_exp, include_none=False):
            models[mi] = create_model_func(wlim[mi])

        return models

    def get_correction_model(self, model, arm, ei, per_arm=True, per_exp=True):
        """
        Get the model for the given arm and exposure index

        Parameters
        ----------
        model : dict or list or tuple
            Model objects
        arm : str
            Spectrograph arm
        ei : int
            Exposure index
        per_arm : bool
            Whether the model is defined per arm
        per_exp : bool
            Whether the model is defined per exposure

        Returns
        -------
        object
            Model object associated with the spectrum (arm, ei)
        """

        if isinstance(model, dict):
            return model[arm][ei]
        elif isinstance(model, list):
            return model[ei]
        else:
            return model

    def init_extinction_curves(self, spectra, force=False):
        """
        Calculate the extinction curve for each exposure.

        Parameters
        ----------
        spectra : dict of Spectrum or dict of list of Spectrum
            Observed spectra
        force : bool
            Force reinitialization of the models
        """

        if self.extinction_model is not None:
            self.extinction_model.trace = self.trace
            self.extinction_model.tempfit = self

            self.extinction_model.init_curves(spectra, force=force)

    def get_normalization(self, spectra, templates, rv_0=None):
        """
        Calculate a normalization factor for the spectra, as well as
        the templates assuming an RV=0 from the median flux.

        Before calculating the normalization, templates are convolved down to
        the resultion of the instrument and resampled to the same wavelength
        grid as the corresponding spectra.

        Make sure to call this function while `spec_norm` and `temp_norm` are
        set to None.

        Parameters
        ----------
        spectra : dict of Spectrum or dict of list of Spectrum
            Observed spectra
        templates : dict of Spectrum
            Synthetic stellar templates for each arm
        rv_0 : float
            Radial velocity

        Returns
        -------
        float, float
            Normalization factors for spectra and templates
        
        This is just a factor to bring spectra to the same scale and avoid
        very large numbers in the Fisher matrix or elsewhere.
        """

        # Do not allow recalculating the normalization when it has already been done
        assert self.spec_norm is None 
        assert self.temp_norm is None

        rv_0 = rv_0 if rv_0 is not None else (self.rv_0 if self.rv_0 is not None else 0.0)

        s = []
        t = []

        for arm, ei, mi, spec in self.enumerate_spectra(spectra, per_arm=False, per_exp=False, include_none=False):
            spec = self.process_spectrum(arm, ei, spec)
            s.append(spec.flux[spec.mask & (spec.flux > 0)])
            
            psf = self.template_psf[arm] if self.template_psf is not None else None
            wlim = self.template_wlim[arm] if self.template_wlim is not None else None    
            temp = self.process_template(arm, templates[arm], spec, rv_0, psf=psf, wlim=wlim)
            t.append(temp.flux[temp.mask])

        if len(s) == 0 or len(t) == 0:
            raise ValueError('No valid flux values found in spectra or templates to determine the optimal normalization.')

        spec_flux = np.concatenate(s)
        temp_flux = np.concatenate(t)

        return np.median(spec_flux), np.median(temp_flux)

    def process_spectrum(self, arm, i, spectrum):
        """
        Preprocess the spectrum for the purposes of evaluating likelihood function.

        Parameters
        ----------
        arm : str
            Spectrograph arm
        i : int
            Exposure index
        spectrum : Spectrum
            Observed spectrum

        Returns
        -------
        Spectrum
            Preprocessed spectrum
        
        This step currently consist only of normalizing the flux with a factor.
        """

        # Do not modifiy the original observed spectrum
        spec = spectrum.copy()

        # Normalize flux and flux_err
        if self.spec_norm is not None:
            spec.multiply(1.0 / self.spec_norm)

        # Determine the binary mask used for template fitting
        spec.mask = self.get_full_mask(spec)

        # Take flux error and calculate its squared
        if self.use_error and spec.flux_err is not None:
            spec.sigma2 = spec.flux_err ** 2
            spec.mask &= ~np.isnan(spec.sigma2)
        else:
            spec.sigma2 = None

        if self.trace is not None:
            self.trace.on_process_spectrum(arm, i, spectrum, spec)

        return spec

    def preprocess_spectra(self, spectra):
        """
        Preprocess the observed spectra for the purposes of evaluating the likelihood function.

        Parameters
        ----------
        spectra : dict of Spectrum or dict of list of Spectrum
            Observed spectra

        Returns
        -------
        dict of list of Spectrum
            Preprocessed spectra
        """

        # Preprocess the observed spectra, mask, weight etc.
        # These are independent of RV 
        pp_spec = { arm: [] for arm in spectra }

        for arm, ei, mi, spec in self.enumerate_spectra(spectra, per_arm=False, per_exp=False, include_none=True):
            if spec is not None:
                pp_spec[arm].append(self.process_spectrum(arm, ei, spec))
            else:
                pp_spec[arm].append(None)

        return pp_spec
    
    def process_template_impl(self, arm, template, spectrum, rv, psf=None, wlim=None):
        """
        Preprocesses a template for the purposes of evaluting the likelihood function.

        Parameters
        ----------
        arm : str
            Spectrograph arm
        template : dict of Spectrum
            Synthetic stellar template, for each arm
        spectrum : dict of Spectrum or dict of list of Spectrum
            Observed spectrum
        rv : float
            Radial velocity
        psf : dict of Psf
            Instrumental line spread function, for each arm
        wlim : tuple
            Wavelength limits for the convolution

        Returns
        -------
        Spectrum
            Preprocessed template
        
        This step is essentially a simulating an observation without calculating any
        uncertainties which consits of a convolution and a normalization step.
        """

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

        # TODO: calculate weight vector from the template
            
        return t

    def process_template(self, arm, template, spectrum, rv, psf=None, wlim=None, resample=True):
        """
        Preprocess the template to match the observation.

        Parameters
        ----------
        arm : str
            Spectrograph arm
        template : Spectrum
            Synthetic stellar template
        spectrum : dict of Spectrum or dict of list of Spectrum
            Observed spectrum
        rv : float
            Radial velocity
        psf : Psf
            Instrumental line spread function
        wlim : tuple
            Wavelength limits for the convolution
        diff : bool
            Calculate the derivative of the template
        resample : bool
            Resample the template to the observation pixels

        Returns
        -------
        Spectrum
            Preprocessed template

        This step involves shifting the high resolution template the a given RV,
        convolving it with the PSF, still at high resolution, and normalizing
        by a factor.

        To improve performance, pre-processed templates are cached at quantized 
        values of RV and reused when the requested RV is close to one in the
        cache. This is fine as long as the PFS changes slowly with wavelength
        which is most often the case. Still, templates sometimes get shifted
        significantly when the target's RV is large so a single precomputed
        template is not enough.

        The template is always kept at high resolution during transformations
        until it's resampled to the spectrum pixels using a flux-conserving
        or an interpolating resampler.
        """

        psf = psf if psf is not None else (self.template_psf[arm] if self.template_psf is not None else None)
        wlim = wlim if wlim is not None else (self.template_wlim[arm] if self.template_wlim is not None else None)

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
                self.template_cache.push(key, temp)

                if self.trace is not None:
                    self.trace.on_template_cache_miss(template, rv_q, rv)

            # Shift from quantized RV to requested RV
            temp = temp.copy()
            temp.set_restframe()
            temp.apply_redshift(Physics.vel_to_z(rv))
        else:
            # Compute convolved template from scratch
            temp = self.process_template_impl(arm, template, spectrum, rv, psf=psf, wlim=wlim)

        if self.trace is not None:
            self.trace.on_process_template(arm, rv, template, temp)

        # Resample template to the binning of the observation
        if resample:
            temp.apply_resampler(self.template_resampler, spectrum.wave, spectrum.wave_edges)

            if self.trace is not None:
                self.trace.on_resample_template(arm, rv, spectrum, template, temp)

        # Create a mask based on the wavelength coverage
        temp.mask = self.get_full_mask(temp)

        if temp.weight is not None:
            temp.mask &= ~np.isnan(temp.weight)

        return temp

    def preprocess_templates(self, spectra, templates, rv, ebv=None):
        """
        Preprocess the templates to match the observations. This is basically simulating
        and observation.

        Parameters
        ----------
        spectra : dict of Spectrum or dict of list of Spectrum
            Observed spectra for each arm and exposure
        templates : dict of Spectrum
            Synthetic stellar templates for each arm
        rv : float
            Radial velocity

        Returns
        -------
        dict of list of Spectrum
            Preprocessed templates
        """

        # Pre-process each exposure in each arm and pass them to the log L calculator
        # function of the flux correction or continuum fit model.
        pp_temp = { arm: [] for arm in spectra }

        for arm, ei, mi, spec in self.enumerate_spectra(spectra, per_arm=False, per_exp=False, include_none=True):
            if spec is not None:
                temp = self.process_template(arm, templates[arm], spec, rv)
                pp_temp[arm].append(temp)
            else:
                pp_temp[arm].append(None)

        # Apply the extinction
        if self.extinction_model is not None and ebv is not None:
            self.extinction_model.apply_extinction(pp_temp, ebv)

        if self.trace is not None:
            for arm, ei, mi, spec in self.enumerate_spectra(spectra, per_arm=False, per_exp=False, include_none=True):
                temp = pp_temp[arm][ei] if spec is not None else None
                if temp is not None:
                    self.trace.on_extinction_template(arm, rv, ebv, templates[arm], temp)

        return pp_temp
    
    def diff_template(self, template, wave=None):
        """
        Calculate the numerical derivative of the template and optionally resample
        to a new wavelength grid using linear interpolation

        Parameters
        ----------
        template : Spectrum
            Template spectrum
        wave : ndarray
            New wavelength grid
        """

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
        """
        Calculate the numerical log-derivative of the template and optionally resample
        to a new wavelength grid using linear interpolation

        Parameters
        ----------
        template : Spectrum
            Template spectrum
        wave : ndarray
            New wavelength grid
        """

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
    
    def get_amp_count(self, spectra: dict):
        """
        Calculate the number of different amplitudes to be fitted.

        Parameters
        ----------
        spectra : dict of Spectrum or dict of list of Spectrum
            Observed spectra

        Returns
        -------
        int
            Number of amplitudes to be fitted

        Depending on the mode of fitting, different amplitudes are fitted on a per arm
        and/or per exposure basis to correct for fluxing artefacts.

        Note, that amplitudes are independent of flux correction models (which are linear combinations
        of function excluding the constant function) and continuum models (which don't account
        for a multiplier) but handled differently for flux correction models and continuum models.
        """

        if self.amplitude_per_fiber:
            raise NotImplementedError()

        # Enumerate the spectra and count the unique model indices `mi`
        # to determine the number of amplitudes to be fitted.
        mmi = []
        for arm, ei, mi, spec in self.enumerate_spectra(spectra, per_arm=self.amplitude_per_arm, per_exp=self.amplitude_per_exp, include_none=False):
            mmi.append(mi)

        amp_count = np.unique(mmi).size

        if amp_count == 0:
            raise ValueError('No valid spectra found')

        return amp_count
    
    @staticmethod
    def get_wave_mask(wave, wave_include, wave_exclude):
        """
        Based on wave_include and wave_exclude, generate a binary mask for the specific
        wavelength vector.
        """

        if wave_include is None:
            mask = np.full_like(wave, True, dtype=bool)
        else:
            mask = np.full_like(wave, False, dtype=bool)
            for wmin, wmax in wave_include:
                mask |= (wmin <= wave) & (wave <= wmax)     # Must be inside either of the included ranges

        if wave_exclude is not None:
            for wmin, wmax in wave_exclude:
                mask &= (wave < wmin) | (wmax < wave)       # Must be outside all of the exluded ranges

        return mask

    def get_full_mask(self, spec, mask_bits=None):
        """
        Return the full mask for the spectrum, including the mask from the spectrum
        and the mask nan and inf values.

        Parameters
        ----------
        spec : Spectrum
            Spectrum
        mask_bits : int
            Flag bits to consider as masked pixels.

        Returns
        -------
        ndarray
            Boolean mask in the shape of the wave array. True values mean that the
            pixel must be included in the fit.
        """
        
        if mask_bits is not None:
            pass
        elif self.mask_bits is not None:
            mask_bits = self.mask_bits
        elif spec.mask_bits is not None:
            mask_bits = spec.mask_bits
        
        mask = None

        if self.use_mask and spec.mask is not None:
            # TODO: mask bits can be different for each spectrum
            mask = spec.mask_as_bool(bits=mask_bits)
        
        if mask is None:
            mask = np.full_like(spec.wave, True, dtype=bool)

        # Included and excluded wavelength ranges
        mask &= self.get_wave_mask(spec.wave, self.wave_include, self.wave_exclude)
        
        # Mask out nan values which might occur if spectrum mask is not properly defined
        mask &= ~(np.isnan(spec.wave) | np.isinf(spec.wave))
        mask &= ~(np.isnan(spec.flux) | np.isinf(spec.flux))
        if spec.flux_err is not None:
            mask &= ~(np.isnan(spec.flux_err) | np.isinf(spec.flux_err))

        # Flux error
        if self.use_error and spec.flux_err is not None:
            mask &= (spec.flux_err / spec.flux) > 1e-5

        return mask
    
    def eval_prior(self, prior, x):
        """
        Evaluates a prior in the form of a distribution of callable function.

        Parameters
        ----------
        prior : Distribution or callable
            Prior distribution or function that returns a normalized probability.
        x : float
            Value to evaluate the prior at

        Returns
        -------
        float
            Log of the prior probability density at x
        """

        if prior is None:
            return 0
        elif isinstance(prior, Distribution):
            return prior.log_pdf(x)
        elif isinstance(prior, Callable):
            return prior(x)
        else:
            raise NotImplementedError()
        
    def sample_prior(self, prior, size=1, bounds=None):
        """
        Draw random number based on a prior distribution.

        Parameters
        ----------
        prior : Distribution or callable
            Prior distribution or function that returns a normalized probability.
        size : int
            Number of samples to draw
        bounds : tuple
            Bounds for the random numbers
        """

        if isinstance(prior, Distribution):
            if bounds is not None:
                prior = prior.copy()
                if bounds is not None:
                    if bounds[0] is not None:
                        prior.min = bounds[0]
                (prior.min, prior.max) = bounds[0]

            raise NotImplementedError()
            
            # x = prior.sample(size)
            # if bounds is not None:
            #     if bounds[0] is not None:
            #         x = max(x, bounds[0])
            #     if bounds[1] is not None:
            #         x = min(x, bounds[1])
            # return x
        elif isinstance(prior, Callable):
            # TODO: implement rejection sampling
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        
    # TODO: this uses emcee but isn't very good and has been replaced by a simple
    #       adaptive MCMC that works better; consider deleting
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

    # region Parameter packing function and objective functions
        
    def get_param_packing_functions(self, mode='full'):
        """
        Return parameter packing functions that combine the correction model parameters
        with the rest of the parameters. These functions are used by the optimizer to
        compose the array of parameters to optimize over.

        Parameters
        ----------
        mode : str
            Optimization mode, either 'full', 'a', 'rv', or the combination of the two,
            where 'full' means pack all parameters, 'a' means pack only the correction
            model parameters, 'rv' means pack only the RV, and 'a_rv' means pack both the
            correction model parameters and the RV, etc.

        Returns
        -------
        tuple
            Tuple of functions that pack and unpack the parameters, and a function that
            packs the parameter bounds.

        This function returns 2d arrays, where the first index runs over the multiple
        sets of parameters.
        """

        # Correction model packing functions
        pp_a, up_a, pb_a = self._get_param_packing_functions_a(mode=mode)

        # Other parameter packing functions
        # pack_params, unpack_params, pack_bounds
        pp_rv, up_rv, pb_rv = self._get_param_packing_functions_rv(mode=mode)

        if mode == 'full' or (mode == 'a_rv' and pp_a is not None and pp_rv is not None):
            # Combine correction model coefficients with RV
            def pack_params(a, rv):
                rv = pp_rv(rv)
                
                a = pp_a(a)
                if a.ndim < 2 and rv.size > 1:
                    a = a[None, :]

                return np.concatenate([ a, rv ], axis=-1)
            
            def unpack_params(a_rv):
                rv = up_rv(a_rv[..., -1])

                if np.ndim(rv) > 1:
                    rv = rv.squeeze(0)
                if isinstance(rv, np.ndarray) and np.size(rv) == 1:
                    rv = rv.item()

                a = up_a(a_rv[..., :-1])

                return a, rv

            def pack_bounds(a_bounds, rv_bounds):
                return pb_a(a_bounds) + pb_rv(rv_bounds)
        elif pp_a is not None:
            pack_params, unpack_params, pack_bounds = pp_a, up_a, pb_a
        elif pp_rv:
            pack_params, unpack_params, pack_bounds = pp_rv, up_rv, pb_rv
        else:
            raise NotImplementedError()
        
        return pack_params, unpack_params, pack_bounds
    
    def _get_param_packing_functions_a(self, mode='full'):

        modes = mode.split('_')

        if mode == 'full' or 'a' in modes:
            def pack_params(a):
                a = np.atleast_1d(a)
                if np.ndim(a) < 2:
                    a = a[None, :]
                return a

            def unpack_params(a):
                if a.ndim == 2 and a.shape[0] == 1:
                    a = np.squeeze(a, 0)
                if a.size == 1:
                    a = a.item()
                return a

            def pack_bounds(a_bounds):
                if a_bounds is None:
                    raise NotImplementedError()
                else:
                    bounds = a_bounds
                return bounds
        else:
            pack_params, unpack_params, pack_bounds = None, None, None
            
        return pack_params, unpack_params, pack_bounds
    
    def _get_param_packing_functions_rv(self, mode='full'):

        modes = mode.split('_')

        if mode == 'full' or 'rv' in modes:
            def pack_params(rv):
                rv = np.atleast_1d(rv)
                if np.ndim(rv) < 2:
                    rv = rv[:, None]
                return rv
                
            def unpack_params(rv):
                if np.ndim(rv) > 1 and rv.shape[0] == 1:
                    rv = rv.squeeze(0)
                if isinstance(rv, np.ndarray) and np.size(rv) == 1:
                    rv = rv.item()
                return rv
            
            def pack_bounds(rv_bounds):
                return [ rv_bounds ]
            
        elif 'rv' not in modes:
            # Only pack amplitude and flux correction or continuum parameters
            pack_params, unpack_params, pack_bounds = None, None, None
        else:
            raise NotImplementedError()

        return pack_params, unpack_params, pack_bounds
    
    def get_objective_function(self,
                               spectra, templates,
                               /,
                               rv_prior=None,
                               mode='full',
                               pp_spec=None):
        """
        Return the objective function and parameter packing/unpacking functions for optimizers

        Parameters
        ----------
        spectra : dict or dict of list
            Dictionary of spectra for each arm and exposure
        templates : dict
            Dictionary of templates for each arm
        rv_prior : Distribution or callable
            RV prior function
        mode : str
            Determines how the model parameters are packed.

        Returns
        -------
        log_L : callable
            Log likelihood function
        pack_params : callable
            Function to pack individual parameters into a 1d array
        unpack_params : callable
            Function to unpack individual parameters from a 1d array
        pack_bounds : callable
            Function to pack parameter bounds into a list of tuples
        """

        pack_params, unpack_params, pack_bounds = self.get_param_packing_functions(mode=mode)

        if mode == 'full' or mode == 'a_rv':
            def log_L(a_rv):
                a, rv = unpack_params(a_rv)
                log_L = self.calculate_log_L(spectra, templates, rv, rv_prior=rv_prior, a=a, pp_spec=pp_spec)
                return log_L
        elif mode == 'rv':
            def log_L(rv):
                rv = unpack_params(rv)
                log_L = self.calculate_log_L(spectra, templates, rv, rv_prior=rv_prior, pp_spec=pp_spec)
                return log_L
        elif mode == 'a':
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        
        return log_L, pack_params, unpack_params, pack_bounds
            
    def get_bounds_array(self, bounds):
        """
        Convert a list of bounds to a 2d array.

        Parameters
        ----------
        bounds : list of tuple
            List of bounds for each parameter
        
        Returns
        -------
        ndarray
            2d array of bounds

        If an element of `bounds` is None, or and element of the tuples is None,
        the array will contain -inf and/or inf.
        """

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

    def _check_param_bounds_edge(
        self,
        param_name,
        param_fixed,
        param_fit,
        param_bounds,
        param_prior,
        params_flags,
        flags,
        eps=1e-3
    ):
        
        if not param_fixed and param_fit is not None and param_bounds is not None:
            if param_bounds[0] is not None and np.abs(param_fit - param_bounds[0]) < eps:
                logger.warning(f"{param_name} {param_fit} is at the lower bound {param_bounds[0]}.")
                params_flags |= TempFitFlag.PARAMEDGE
            if param_bounds[1] is not None and np.abs(param_fit - param_bounds[1]) < eps:
                logger.warning(f"{param_name} {param_fit} is at the upper bound {param_bounds[1]}.")
                params_flags |= TempFitFlag.PARAMEDGE

        # Check if the param is at the edge of the prior distribution
        if not param_fixed and param_fit is not None and isinstance(param_prior, Distribution):
            if param_prior.is_at_edge(param_fit):
                logger.warning(f"{param_name} {param_fit} is at the edge of the prior distribution.")
                params_flags |= TempFitFlag.PARAMEDGE

        if params_flags != 0:
            flags |= TempFitFlag.BADCONVERGE

        return params_flags, flags

    def check_bounds_edge(self, state, eps=1e-3):
        """
        Check if the fitted parameters are at the edge of the bounds.
        """

        # Check if the RV is at the edge of the bounds
        state.rv_flags, state.flags = self._check_param_bounds_edge(
            'RV', state.rv_fixed, state.rv_fit, state.rv_bounds, state.rv_prior,
            state.rv_flags, state.flags,
            eps=eps)

    def _check_param_prior_unlikely(
        self,
        param_name,
        param_fixed,
        param_fit,
        param_prior,
        param_flags,
        flags,
        lp_limit=-5
    ):
        
        if not param_fixed and param_fit is not None and param_prior is not None:
            lp = self.eval_prior(param_prior, param_fit)
            if lp is not None and lp / np.log(10) < lp_limit:
                logger.warning(f"Prior for {param_name} {param_fit} is very unlikely with lp {lp}.")
                param_flags |= TempFitFlag.UNLIKELYPRIOR

        return param_flags, flags

    def check_prior_unlikely(self, state, lp_limit=-5):
        
        """
        Check if the fitted parameters are unlikely a priori. This means
        that the convergence might be bad.
        """
        
        state.rv_flags, state.flags = self._check_param_prior_unlikely(
            'RV', state.rv_fixed, state.rv_fit, state.rv_prior, state.rv_flags, state.flags)
        
    # endregion
    # region Fisher matrix evaluation

    # There are multiple ways of evaluating the Fisher matrix. Functions
    # with _full_ return the full Fisher matrix including matrix elements for
    # the flux correction coefficients and rv. Functions with _rv_ return
    # the (single) matrix element corresponding to the RV only. Functions with
    # _hessian depend on the numerical evaluation of the Hessian with respect
    # to the parameters of the fitted model.

    def calculate_F(self, spectra, templates,
                    rv_0=None, rv_fixed=None, rv_bounds=None, rv_prior=None,
                    step=None, mode='full', method='hessian',
                    pp_spec=None):
        
        """
        Evaluate the Fisher matrix around the provided rv_0.
        
        Parameters
        ----------
        spectra : dict of Spectrum or dict of list of Spectrum
            Observed spectra
        templates : dict of Spectrum
            Synthetic stellar templates for each arm
        rv_0 : float
            Radial velocity around which the Hessian is evaluated
        rv_fixed: bool
            Whether to fix the RV during the optimization
        rv_bounds : tuple
            RV bounds to limit the numerical differentiation
        rv_prior : Distribution or callable
            Prior distribution for the RV
         
        The corresponding a_0 best fit flux correction or continuum model
        will be evaluated at the optimum. The Hessian is calculated wrt either
        RV only, or rv and the flux correction coefficients.
        Alternatively, the covariance matrix will be determined using MCMC.
        """

        rv_0 = rv_0 if rv_0 is not None else self.rv_0
        rv_fixed = rv_fixed if rv_fixed is not None else self.rv_fixed
        rv_bounds = rv_bounds if rv_bounds is not None else self.rv_bounds
        rv_prior = rv_prior if rv_prior is not None else self.rv_prior

        # Get objective function
        log_L, pack_params, unpack_params, pack_bounds = self.get_objective_function(
            spectra, templates,
            rv_prior,
            mode=mode,
            pp_spec=pp_spec)
            
        if mode == 'full' or mode == 'a_rv':
            # Determine the flux correction or continuum fit parameters at the
            # optimum RV
            a_0, _, _ = self.calculate_coeffs(spectra, templates, rv_0, pp_spec=pp_spec)
            x_0 = pack_params(a_0, rv_0)[0]
            bounds = pack_bounds(np.size(a_0) * [(-np.inf, np.inf)], rv_bounds)
        elif mode == 'rv':
            x_0 = pack_params(rv_0)[0]
            bounds = pack_bounds(rv_bounds)
        else:
            raise NotImplementedError()
        
        bounds = self.get_bounds_array(bounds)

        F, C = self.eval_F_dispatch(x_0, log_L, step, method, bounds)

        if F.ndim > 2:
            F = F.squeeze(0)
            C = C.squeeze(0)
                
        return F, C

    def eval_F_dispatch(self, x_0, log_L_fun, step, method, bounds):
        if method == 'hessian':
            return self.eval_F_hessian(x_0, log_L_fun, step)
        elif method == 'sampling':
            return self.eval_F_sampling(x_0, log_L_fun, step, bounds)
        else:
            raise NotImplementedError()

    def eval_F_hessian(self, x_0, log_L_fun, step, inverse=True):
        """
        Evaluate the Fisher matrix by calculating the Hessian numerically

        Parameters
        ----------
        x_0 : ndarray
            Initial parameters
        log_L_fun : callable
            Log likelihood function
        step : float or ndarray
            Step size for numerical differentiation
        inverse : bool
            Whether to calculate the inverse of the Hessian (the covariance)

        Returns
        -------
        ndarray
            Fisher matrix
        ndarray
            Inverse of the Fisher matrix (covariance matrix)
        """

        # Default step size is 1% of optimum values
        if step is None:
            step = 0.01 * x_0

        dd_log_L = nd.Hessian(log_L_fun, step=step)
        dd_log_L_0 = dd_log_L(x_0)

        if inverse:
            try:
                inv = np.linalg.inv(-dd_log_L_0)
            except np.linalg.LinAlgError:
                logger.warning("Fisher matrix is singular and cannot be inverted.")
                inv = None
        else:
            inv = None

        return dd_log_L_0, inv
        
    def eval_F_sampling(self, x_0, log_L_fun, step, bounds, inverse=True):
        """
        Sample a bunch of RVs around the optimum and fit a parabola
        to obtain the error of RV.

        Parameters
        ----------
        x_0 : ndarray
            Initial parameters
        log_L_fun : callable
            Log likelihood function
        step : float or ndarray
            Step size for numerical differentiation
        bounds : ndarray
            Bounds for the parameters
        inverse : bool
            Whether to calculate the inverse of the Hessian (the covariance)

        Returns
        -------
        ndarray
            Fisher matrix
        """

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
            p = np.polyfit(x.item(), log_L, 2)
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

            if inverse:
                inv = np.linalg.inv(-F)
            else:
                inv = None

            return F, inv
    
    #endregion
    
    @staticmethod
    def lorentz(x, a, b, c, d):
        return a / (1 + (x - b) ** 2 / c ** 2) + d

    def fit_lorentz(self, rv, y0):
        """
        Fit a Lorentz function to the log-likelihood to have a good initial guess for RV.

        Parameters
        ----------
        rv : ndarray
            Radial velocity
        y0 : ndarray
            Log-likelihood

        Returns
        -------
        ndarray
            Best fit parameters
        """

        # Guess initial values from y0
        p0 = [
            y0.max() - y0.min(),
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
                5 * (y0.max() - y0.min()),
                rv[-1] + 1,
                5.0 * (rv[-1] - rv[0]),
                y0.min() + 4 * (y0.max() - y0.min())
            )
        ]

        logger.info(f"Fitting Lorentz function to the log likelihood in {len(rv)} points "
                    f"with initial values {p0} and bounds {bb}.")

        # TODO: verify that params within the bounds
        #       for some reason the initial amplitude becomes negative
        
        pp, pcov = curve_fit(self.lorentz, rv, y0, p0=p0, bounds=bb)

        logger.info(f"Lorentz function fitted to the log likelihood with best fit parameters {pp}.")

        return pp, pcov

    def calculate_log_L(
        self,
        spectra,
        templates,
        rv, /, rv_prior=None,
        ebv=None,
        a=None,
        pp_spec=None
    ):
        
        """
        Calculate the logarithm of the likelihood at the given values of RV.

        Parameters
        ----------
        spectra : dict of Spectrum or dict of list of Spectrum
            Observed spectra
        templates : dict of Spectrum
            Synthetic stellar templates for each arm
        rv : float or ndarray
            Radial velocity
        rv_prior : Distribution or callable
            Prior distribution for the RV
        a : ndarray
            Correction model parameters (optional)
        
        The function calls into the underlying correction model to calculate log L
        and adds the contribution of the priors. When the correction model parameters
        are provided, they are used directly, otherwise they are calculated from the
        spectra and templates.
        """
        
        if not isinstance(rv, np.ndarray):
            rvv = np.atleast_1d(rv)
        else:
            rvv = rv.flatten()

        rv_prior = rv_prior if rv_prior is not None else self.rv_prior

        if self.trace is not None:
            trace_state = TempFitTraceState()

        if pp_spec is None:
            pp_spec = self.preprocess_spectra(spectra)

        # Calculate log L at each RV
        log_L = np.full(rvv.shape, np.nan)
        for i in range(rvv.size):
            if self.trace is not None:
                trace_state.reset()

            # Pre-process each exposure in each arm and pass them to the log L calculator
            # function of the flux correction or continuum fit model.
            pp_temp = self.preprocess_templates(spectra, templates, rvv[i], ebv)

            # Save everything to the trace
            if self.trace is not None:
                for arm in pp_spec:
                    for ei, (spec, temp) in enumerate(zip(pp_spec[arm], pp_temp[arm])):
                        trace_state.append(arm, spec, temp)

            # Calculate log L
            lp = self.correction_model.eval_log_L(pp_spec, pp_temp, a=a)

            # Add the priors
            # TODO: factor this into a function that can be overridden
            lp += self.eval_prior(rv_prior, rvv[i])

            log_L[i] = lp

            if self.trace is not None:
                self.trace.on_calculate_log_L(trace_state.spectra, trace_state.templates, rvv[i], lp)

        log_L = np.reshape(log_L, np.shape(rv))
        if isinstance(log_L, np.ndarray) and log_L.size == 1:
            log_L = log_L.item()

        return log_L

    def calculate_coeffs(self, spectra, templates, rv, ebv=None, pp_spec=None):
        """
        Given a set of observed spectra and templates, evaluate the correction model
        to determine the correction model parameters at the given RV.

        Parameters
        ----------
        spectra : dict of Spectrum or dict of list of Spectrum
            Observed spectra
        templates : dict of Spectrum
            Synthetic stellar templates for each arm
        rv : float
            Radial velocity

        Returns
        -------
        ndarray or dict of ndarray
            Correction model parameters, in a format depending on the correction model.
        dict of Spectra
            Preprocessed templates that match the observed spectra in wavelength
        """

        if pp_spec is None:
            pp_spec = self.preprocess_spectra(spectra)
        pp_temp = self.preprocess_templates(spectra, templates, rv, ebv)
        a = self.correction_model.calculate_coeffs(pp_spec, pp_temp)
        
        return a, pp_spec, pp_temp
    
    def init_state(self, spectra, templates, /, fluxes=None,
                    rv_0=None, rv_bounds=None, rv_prior=None, rv_step=None, rv_fixed=None,
                    pp_spec=None):
        """
        This function is called before template fitting to normalize all parameters and
        prepare them for the optimization.

        Parameters
        ----------
        spectra : dict of Spectrum or dict of list of Spectrum
            Observed spectra
        templates : dict of Spectrum
            Synthetic stellar templates for each arm
        rv_0 : float
            Initial guess for the RV
        rv_bounds : tuple
            RV bounds to limit the search for initial RV if not provided as well as
            limits to the fit.
        rv_prior : Distribution or callable
            Prior distribution for the RV
        rv_step : float
            Step size for MCMC or numerical differentiation
        rv_fixed : bool
            If True, the RV is fixed and no optimization is performed.
        pp_spec : dict of Spectrum or dict of list of Spectrum
            Preprocessed spectra, if already available

        Returns
        -------
        TempFitState
            State object that contains all the information needed to run the fitting
        """

        assert isinstance(spectra, dict)
        assert isinstance(templates, dict)

        state = TempFitState(
            spectra=spectra,
            templates=templates,
            fluxes=fluxes
        )
        
        state.rv_0 = rv_0 if rv_0 is not None else self.rv_0
        state.rv_fixed = rv_fixed if rv_fixed is not None else self.rv_fixed
        state.rv_bounds = rv_bounds if rv_bounds is not None else (self.rv_bounds if self.rv_bounds is not None else (-500, 500))
        state.rv_prior = rv_prior if rv_prior is not None else self.rv_prior
        state.rv_step = rv_step if rv_step is not None else self.rv_step

        # Initialize flux correction or continuum models for each arm and exposure
        self.init_correction_models(spectra, rv_bounds=rv_bounds)

        # Calculate the pre-normalization constants. This is only here to make sure that the
        # matrix elements during calculations stay around unity
        # TODO: this has side effect, should add spec_norm and temp_norm to the state instead?
        if self.spec_norm is None or self.temp_norm is None:
            self.spec_norm, self.temp_norm = self.get_normalization(spectra, templates, state.rv_0)

        # Determine the (buffered) wavelength limit in which the templates will be convolved
        # with the PSF. This should be slightly larger than the observed wavelength range.
        # TODO: this has side-effect, add template_wlim to the state instead?
        if self.template_wlim is None:
            # Use different template wlim for each arm but same for each exposure
            self.template_wlim = {}
            wlim = self.determine_wlim(spectra, per_arm=True, per_exp=False, rv_bounds=state.rv_bounds)
            for mi, arm in enumerate(spectra):
                self.template_wlim[arm] = wlim[mi]

        for mi, arm in enumerate(spectra):
            logger.debug(f"Template wavelength limits for {arm}: {wlim[mi]}")

        # Preprocess the spectra
        if pp_spec is not None:
            state.pp_spec = pp_spec
        else:
            state.pp_spec = self.preprocess_spectra(spectra)

        # Get objective function, etc
        state.log_L_fun, state.pack_params, state.unpack_params, state.pack_bounds = self.get_objective_function(
            spectra, templates,
            rv_prior=state.rv_prior,
            mode='rv',
            pp_spec=state.pp_spec)
        
        self._pack_everything(state)

        # Set default values of the flags
        state.flags = TempFitFlag.OK
        state.rv_flags = TempFitFlag.OK

        return state

    def _pack_everything(self, state):
        """
        Pack all parameters from the state into a single array for optimization.
        """

        # Initial values
        if state.rv_0 is not None:
            state.x_0 = state.pack_params(state.rv_0)[0]
        else:
            state.x_0 = None

        # Step size for MCMC
        if state.rv_step is not None:
            state.steps = state.pack_params(state.rv_step)[0]
        else:
            state.steps = None

        # Parameter bounds for optimizers, bounds is a list of tuples, convert to an array
        bounds = state.pack_bounds(state.rv_bounds)
        state.bounds = self.get_bounds_array(bounds)

    @deprecated("Use `guess_ml` instead.")
    def guess_rv(self, spectra, templates, /,
                 rv_bounds=None, rv_prior=None, rv_step=None,
                 steps=None,
                 method='lorentz',
                 pp_spec=None):

        state = self.init_state(spectra, templates,
                                rv_bounds=rv_bounds,
                                rv_prior=rv_prior,
                                rv_step=rv_step,
                                pp_spec=pp_spec)

        state, rv, log_L, = self.guess_ml(
            state,
            steps=steps,
            method=method)

        return rv, log_L, state.rv_guess, state.log_L_guess
        
    @deprecated("Use `run_ml` instead.")
    def fit_rv(self, spectra, templates,
               fluxes=None,
               rv_0=None, rv_bounds=None, rv_prior=None, rv_fixed=None,
               method='bounded',
               calculate_error=True,
               calculate_cov=True):

        """
        Given a set of spectra and templates, find the best fit RV by maximizing the log likelihood.

        This function is kept for backward compatibility. Use `run_ml` instead.
        """

        state = self.init_state(
                spectra, templates,
                fluxes=fluxes,
                rv_0=rv_0,
                rv_fixed=rv_fixed,
                rv_bounds=rv_bounds,
                rv_prior=rv_prior)

        res, state = self.run_ml(state)

        if calculate_error:
            res, state = self.calculate_error_ml(state)

        if calculate_cov:
            res, state = self.calculate_cov_ml(state)

        res, state = self.finish_ml(state)

        return res

    def guess_ml(self, state, steps=None, method='lorentz'):
        """
        Given a spectrum and a template, make a good initial guess for RV where a minimization
        algorithm can be started from.

        Parameters
        ----------
        state: TempFitState
            Current state of the template fitting
        method : str
            Method to use to guess the RV: 'lorentz' or 'max'

        Returns
        -------
        TempFitState, ndarray, ndarray, float
            current state, RV grid, log likelihood values, guessed RV
        """

        method = method if method is not None else 'lorentz'

        logger.info(f"Guessing RV with method `{method}` using a fixed template. RV bounds are {state.rv_bounds} km/s, "
                    f"number of steps is {steps}.")

        if state.rv_bounds is None:
            state.rv_bounds = (-500, 500)
        else:
            if state.rv_bounds[0] is None or not np.isfinite(state.rv_bounds[0]):
                state.rv_bounds = (-500, state.rv_bounds[1])
            if state.rv_bounds[1] is None or not np.isfinite(state.rv_bounds[1]):
                state.rv_bounds = (state.rv_bounds[0], 500)
        
        # Calculate the number of steps for the RV grid
        if steps is None and state.rv_bounds is not None and state.rv_step is not None \
            and state.rv_bounds[0] is not None and state.rv_bounds[1] is not None \
            and np.isfinite(state.rv_bounds[0]) and np.isfinite(state.rv_bounds[1]):

            steps = int((state.rv_bounds[1] - state.rv_bounds[0]) / state.rv_step)
        else:
            steps = 31
    
        rv = np.linspace(*state.rv_bounds, steps)
        log_L = self.calculate_log_L(state.spectra, state.templates, rv, rv_prior=state.rv_prior, pp_spec=state.pp_spec)
        
        # Mask out infs here in case the prior is very narrow
        mask = (~np.isnan(log_L) & ~np.isinf(log_L))
        if mask.sum() < 10:
            raise Exception("Too few valid values to guess RV. Consider changing the bounds.")

        if method == 'lorentz':
            pp, pcov = self.fit_lorentz(rv[mask], log_L[mask])
            rv_guess = pp[1]
            log_L_guess = self.lorentz(rv_guess, *pp)

            # The maximum of the Lorentz curve might be outside the bounds
            outside = False
            if state.rv_bounds is not None and state.rv_bounds[0] is not None:
                rv_guess = max(rv_guess, state.rv_bounds[0])
                outside = True
            if state.rv_bounds is not None and state.rv_bounds[1] is not None:
                rv_guess = min(rv_guess, state.rv_bounds[1])
                outside = True

            if outside:
                logger.warning(f"RV guess from method `{method}` is {rv_guess:0.3f} km/s, "
                               f"which is outside the search bounds {state.rv_bounds}.")
                
            if self.trace is not None:
                self.trace.on_guess_rv(rv, log_L, rv_guess, log_L_guess, self.lorentz(rv, *pp), 'lorentz', pp, pcov)
        elif method == 'max':
            imax = np.argmax(log_L)
            rv_guess = rv[imax]
            log_L_guess = log_L[imax]

            if imax == 0 or imax == log_L.size - 1:
                logger.warning(f"RV guess form method `{method}` is {rv_guess:0.3f} km/s, "
                               f"which is at the edge of the search bounds {state.rv_bounds}.")
          
            if self.trace is not None:
                self.trace.on_guess_rv(rv, log_L, rv_guess, log_L_guess, None, 'max', None, None)
        else:
            raise NotImplementedError()
        
        logger.info(f"Initial guess for RV using the method `{method}` is {rv_guess:0.3f} km/s.")

        state.rv_guess = rv_guess
        state.log_L_guess = log_L_guess

        return state, rv, log_L

    def run_ml(self, state, method='bounded'):
        """
        Given a set of spectra and templates, find the best fit RV by maximizing the log likelihood.
        Spectra are assumed to be of the same object in different wavelength ranges, with multiple
        exposures.

        Parameters
        ----------
        state: TempFitState
            State initialized by `init_state`

        Returns
        -------
        TempFitResults
            Results of the template fitting

        If no initial guess is provided, `rv_0` is determined automatically. If `rv_fixed` is
        `True`, the radial velocity is not fitted but the best flux correction or continuum
        model parameters are determined as if the provided `rv_0` was the best fit.
        """
                            
        if state.rv_0 is None:
            state, rv, log_L = self.guess_ml(state, method='max')
            state.rv_0 = state.rv_guess

            if state.rv_fixed:
                logger.warning("No value of RV is provided, yet not fitting RV. The guessed value will be used.")

        self._pack_everything(state)
            
        if self.trace is not None:
            wave_include, wave_exclude = self.get_wave_include_exclude()
            self.trace.on_fit_rv_start(state.spectra, state.templates,
                                       state.rv_0, state.rv_bounds, state.rv_prior, state.rv_step,
                                       state.log_L_0, state.log_L_fun,
                                       wave_include=wave_include,
                                       wave_exclude=wave_exclude)

        if state.rv_fixed:
            # Only calculate the flux correction or continuum model coefficients at rv_0
            results, state = self._fit_rv_fixed(state)
        else:
            # Optimize for RV
            results, state = self._fit_rv_optimize(state, method)

        return results, state

    def finish_ml(self, state):
        
        if self.trace is not None:
            # If tracing, evaluate the template at the best fit RV.
            ss, tt = self.append_corrections_and_templates(state.spectra, state.templates,
                                                           state.rv_fit, a_fit=state.a_fit,
                                                           match='spectrum',
                                                           apply_correction=False)

            # Pass the log likelihood function to the trace
            def log_L_fun(rv):
                return self.calculate_log_L(state.spectra, state.templates, rv, rv_prior=state.rv_prior)

            self.trace.on_fit_rv_finish(ss, tt, 
                                        state.rv_0, state.rv_fit, state.rv_err, state.rv_bounds, state.rv_prior, state.rv_step, False,
                                        state.log_L_0, state.log_L_fit, log_L_fun)

        return TempFitResults.from_state(state), state


    def calculate_error_ml(self, state):
        if not state.rv_fixed:
            # Calculate the error only when the RV is not fixed
            state.cov_params = [0]   # RV is the only parameter
            _, state.cov = self.calculate_F(
                state.spectra, state.templates,
                state.rv_fit, rv_bounds=state.rv_bounds, rv_prior=state.rv_prior,
                mode='rv', method='hessian')
            state.cov_params = ['v_los']

            with np.errstate(invalid='warn'):
                state.rv_err = np.sqrt(state.cov[-1, -1])         # sigma

        return TempFitResults.from_state(state), state

    def calculate_cov_ml(self, state):
        return TempFitResults.from_state(state), state

    def _fit_rv_fixed(self, state):
        """
        Calculate the log likelihood at the provided RV. This requires evaluating the 
        correction model (flux correction or continuum normalization).

        Evan though the RV is fixed, the contribution of the prior is added to the log likelihood.

        Parameters
        ----------
        state : TempFitState
            State of the fit containing all parameters
        """


        # Calculate log_L in two steps, this requires passing around the flux correction or
        # continuum fit parameters
        # a_fit, _, _ = self.calculate_coeffs(spectra, templates, rv_0)
        # a_err = np.full_like(a_fit, np.nan)
        # log_L_fit = self.calculate_log_L(spectra, templates, rv_0, rv_prior=rv_prior, a=a_fit, pp_spec=pp_spec)
        
        # Calculate log_L in a single step, this won't provide the flux correction or
        # continuum fit parameters
        state.rv_fit = state.rv_0
        state.rv_err = 0.0
        state.rv_flags = TempFitFlag.OK
        state.a_fit = None
        state.a_err = None
        state.cov = None
        state.cov_params = None
        state.log_L_fit = self.calculate_log_L(
            state.spectra, state.templates,
            state.rv_0, rv_prior=state.rv_prior, pp_spec=state.pp_spec)

        return TempFitResults.from_state(state), state
        
    def _fit_rv_optimize(self, state, method):

        """
        Optimize for the RV. This function is called when the RV is not fixed.

        Parameters
        ----------
        state : TempFitState
            State of the fit containing all parameters
        method : str
            Optimization method to use: 'grid', otherwise scalar
        """
                
        # Cost function
        def llh(rv):
            return -state.log_L_fun(state.pack_params(rv))
        
        # Univariate method
        # NOTE: scipy.optimize is sensitive to type of args
        rv_bounds = tuple(float(b) for b in state.rv_bounds)
        bracket = None
            
        try:
            if method == 'grid':
                out = TempFit.minimize_gridsearch(llh, bounds=rv_bounds)
            else:
                out = minimize_scalar(llh, bracket=bracket, bounds=rv_bounds, method=method)
        except Exception as ex:
            raise ex

        if out.success:
            state.rv_fit = state.unpack_params(out.x)
            state.log_L_fit = -out.fun
        
            if method == 'grid':
                pass
            else:
                logger.debug(f"Optimizer {method} message: {out.message}")
                logger.debug(f"Optimizer {method} number of iterations: {out.nit}, "
                             f"number of function evaluations: {out.nfev}")
        else:
            raise Exception(f"Could not fit RV using `{method}`")

        self.check_bounds_edge(state)
        self.check_prior_unlikely(state)

        # TODO: check if log_L decreased

        # Calculate the flux correction or continuum fit coefficients at best fit values
        state.a_fit, _, _ = self.calculate_coeffs(state.spectra, state.templates,
                                                  state.rv_fit,
                                                  pp_spec=state.pp_spec)
        
        # Set the errors to NaN for now
        state.rv_err = np.nan
        state.a_err = None
        state.cov = None
        state.cov_params = None
        
        return TempFitResults.from_state(state), state

    def eval_correction(self, spectra, templates, rv, ebv=None, a=None):
        """
        Evaluate the correction model at the given RV on the wavelength grid
        of the observed spectra.

        This function serves mainly debugging purposes as the meaning of the correction
        model is different for flux correction or continuum normalization.

        Parameters
        ----------
        spectra : dict of Spectrum or dict of list of Spectrum
            Observed spectra
        templates : dict of Spectrum
            Synthetic stellar templates for each arm
        rv : float
            Radial velocity
        a : ndarray
            Correction model parameters (optional)

        Returns
        -------
        pp_specs: dict of list of Spectrum
            Preprocessed spectra with normalization applied.
        pp_temps: dict of list of Spectrum
            Preprocessed templates that match the wave grid of the observed spectra
        corrections: dict of list of np.ndarray
            Correction model for each arm and exposure
        correction_masks: dict of list of np.ndarray
            Correction model masks for each arm and exposure
        """

        pp_specs = self.preprocess_spectra(spectra)
        pp_temps = self.preprocess_templates(spectra, templates, rv, ebv)
        
        corrections, correction_masks = self.correction_model.eval_correction(pp_specs, pp_temps, a=a)

        return pp_specs, pp_temps, corrections, correction_masks

    def append_corrections_and_templates(
            self,
            spectra,
            templates,
            rv_fit,
            ebv_fit=None,
            a_fit=None,
            match=None,
            apply_correction=True):
        
        """
        Evaluate the correction model and best fit template and append them to the spectra.

        The flux is not modified, only the correction model and best fit template are
        appended. The mask of the correction model is applied to the spectrum mask.

        Parameters
        ----------
        spectra : dict of Spectrum or dict of list of Spectrum
            Observed spectra
        templates : dict of Spectrum
            Synthetic stellar templates for each arm
        rv_fit : float
            Best fit radial velocity
        ebv_fit : float
            Best fit E(B-V)
        a_fit : ndarray
            Best fit correction model parameters
        match : str or None
            If 'spectrum', the correction model is applied to the templates so that they match
            the observed flux. If 'template', the correction model is applied to the observed
            spectra so that they match the template. If None, the correction model is just appended
            to the spectra and templates but not applied.
        apply_correction : bool
            If True and `match` is 'template', the correction model is applied to the observed
            spectra so that they match the template. If True and `match` is 'spectrum', the correction
            model is applied to the templates so that they match the observed flux. If False, the
            correction model is just appended to the spectra and templates but not applied to the flux.
        """

        spectra = safe_deep_copy(spectra)

        # Evaluate the correction model at the best fit parameters
        pp_spec, pp_temp, corrections, correction_masks = self.eval_correction(spectra, templates, rv_fit, ebv=ebv_fit, a=a_fit)

        # At this point spectra.flux is in physical, observed units,
        # pp_spec.flux is observed flux scaled to unity and pp_temp.flux is scaled to unity.

        # Append the correction model to the original spectra but do not change the flux
        self.correction_model.append_model(spectra, corrections, correction_masks, apply_mask=True,
                                               normalization=None, apply_normalization=False)
        
        # Append the flux correction to the templates so that they match the observed flux but
        # do not actually change the flux
        self.correction_model.append_model(pp_temp, corrections, correction_masks, apply_mask=False,
                                            normalization=None, apply_normalization=False)

        if apply_correction:
            if match is None:
                # Only append the correction model, do not scale or correct the flux
                pass
            elif match == 'spectrum':           
                # The correction is applied to the templates
                # TODO: why don't we have an `if apply_correction` here?
                self.correction_model.apply_correction(pp_temp, template=True)
            elif match == 'template':
                # If requested, apply the flux correction to the observed spectra
                # so that they match the template
                self.correction_model.apply_correction(spectra, template=False)
            else:
                raise NotImplementedError()

        # Scale the templates to the spectra
        self.multiply_spectra(pp_temp, self.spec_norm)

        # Append the best fit template to the original spectra. If the correction is not applied,
        # the template is just scaled to the observed flux but won't match the observed flux.
        for arm in spectra:
            for ei, spec in enumerate(spectra[arm] if isinstance(spectra[arm], list) else [spectra[arm]]):
                if spec is not None:
                    spec.flux_model = pp_temp[arm][ei].flux
                    spec.cont = pp_temp[arm][ei].cont

        return spectra, pp_temp
    
    def multiply_spectra(self, spectra, factor):
        for arm in spectra:
            for spec in spectra[arm]:
                if spec is not None:
                    spec.multiply(factor)

    def randomize_init_params(self, spectra,
                              rv_0=None, rv_bounds=None, rv_prior=None, rv_step=None, rv_fixed=None, rv_err=None,
                              randomize=False, random_size=()):
        
        """
        Randomize the initial parameters for MCMC.

        Parameters
        ----------
        spectra : dict of Spectrum or dict of list of Spectrum
            Observed spectra
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
        rv_err : float
            RV error
        randomize : bool
            Whether to randomize the initial parameters. When False, this function is just a pass-through
            unless the initial value is None.
        random_size : tuple
            Size of the random array to generate

        Returns
        -------
        rv
            Randomized initial parameter
        rv_err
            Initial sigma for adaptive parameter sampling
        """
        
        rv_fixed = rv_fixed if rv_fixed is not None else self.rv_fixed
        rv_bounds = rv_bounds if rv_bounds is not None else self.rv_bounds
        rv_prior = rv_prior if rv_prior is not None else self.rv_prior
        rv_step = rv_step if rv_step is not None else self.rv_step

        # Generate an initial state for MCMC by sampling the prior randomly

        if rv_0 is None or np.isnan(rv_0):
            if rv_prior is not None:
                rv = self.sample_prior(rv_prior, bounds=rv_bounds)
            else:
                if self.rv_0 is not None:
                    rv = self.rv_0
                else:
                    raise NotImplementedError()
        else:
            rv = rv_0
                
        if randomize and not rv_fixed:
            if rv_step is not None:
                rv = rv + rv_step * (np.random.rand(*random_size) - 0.5)
            else:
                rv = rv * (1.0 + 0.05 * (np.random.rand(*random_size) - 0.5))

        if rv_bounds is not None:
            if rv_bounds[0] is not None:
                rv = np.maximum(rv, rv_bounds[0])
            if rv_bounds[1] is not None:
                rv = np.minimum(rv, rv_bounds[1])

        if rv_err is None or np.any(np.isnan(rv_err)):
            if rv_step is not None:
                rv_err = rv_step ** 2
            else:
                rv_err = (0.05 * np.mean(rv)) ** 2 + 1.0

        return rv, rv_err

    def run_mcmc(self, spectra, templates, *,
                 fluxes=None,
                 rv_0=None, rv_bounds=None, rv_prior=None, rv_step=None, rv_fixed=None,
                 cov=None,
                 walkers=None, burnin=None, samples=None, thin=None, gamma=None):
        """
        Given a set of spectra and templates, sample from the posterior distribution of RV.

        If no initial guess is provided, an initial state is generated automatically.

        Parameters
        ----------
        spectra : dict of Spectrum or dict of list of Spectrum
            Observed spectra
        templates : dict of Spectrum
            Synthetic stellar templates for each arm
        rv_0 : float
            Initial guess for the RV
        rv_bounds : tuple
            RV bounds to limit the search for initial RV if not provided as well as
            limits to the fit.
        rv_prior : Distribution or callable
            Prior distribution for the RV
        rv_step : float
            Step size for MCMC or numerical differentiation
        rv_fixed : bool
            If True, the RV is fixed and no optimization is performed.
        cov : ndarray
            Covariance matrix for the parameters (placeholder for future use)
        walkers : int
            Number of walkers
        burnin : int
            Number of burn-in steps
        samples : int
            Number of MCMC samples to generate
        thin : int
            Thinning factor
        gamma : float
            Adaptive memory factor for the MCMC sampler
        """

        assert isinstance(spectra, dict)
        assert isinstance(templates, dict)

        walkers = walkers if walkers is not None else self.mcmc_walkers
        burnin = burnin if burnin is not None else self.mcmc_burnin
        samples = samples if samples is not None else self.mcmc_samples
        thin = thin if thin is not None else self.mcmc_thin
        gamma = gamma if gamma is not None else self.mcmc_gamma

        # Initialize flux correction or continuum models for each arm and exposure
        self.init_correction_models(spectra, rv_bounds=rv_bounds)

        state = self.init_state(
                spectra, templates,
                fluxes=fluxes,
                rv_0=rv_0, rv_bounds=rv_bounds,
                rv_prior=rv_prior, rv_step=rv_step,
                rv_fixed=rv_fixed)
        
        if state.bounds is not None and \
            np.any((np.transpose(state.x_0) < state.bounds[..., 0]) |
                   (state.bounds[..., 1] < np.transpose(state.x_0))):

            raise Exception("Initial state is outside bounds.")
        
        # We only have a single parameter, there's no need for Gibbs sampling
        gibbs_blocks = [[ 0 ]]

        mcmc = MCMC(state.log_L_fun, step_size=state.steps,
                    gibbs_blocks=gibbs_blocks,
                    walkers=walkers, gamma=0.99)
        
        # A leading dimension is required by the mcmc sampler in x_0
        # res_x.shape: (num_vars, samples, walkers)
        # res_lp.shape: (samples, walkers)
        res_x, res_lp, accept_rate = mcmc.sample(state.x_0[None, ...],
                                                 burnin, samples, gamma=gamma)

        rv = state.unpack_params(res_x)

        # TODO: assign flags, calculate moments, etc.

        return TempFitResults(rv_mcmc=rv, rv_flags=state.rv_flags,
                              log_L_mcmc=res_lp,
                              accept_rate=accept_rate,
                              flags=state.flags)
    
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
        #       this function depends on a flux correction model

        a_0, _, _ = self.calculate_coeffs(spectra, templates, rv_0)

        nom = 0
        den = 0
        sumw = 0

        for arm in spectra:
            if not isinstance(spectra[arm], list):
                specs = [spectra[arm]]
            else:
                specs = spectra[arm]
            
            for ei in range(len(specs)):
                spec = self.process_spectrum(arm, ei, specs[ei])
                temp = self.process_template(arm, templates[arm], spec, rv_0)

                if self.use_mask:
                    mask = ~spec.mask if spec.mask is not None else np.s_[:]
                else:
                    mask = np.s_[:]

                if isinstance(mask, np.ndarray):
                    # We are going to calculate the central difference of flux for
                    # each pixel, so dilate the mask here by 1 pixel in both directions
                    mask = binary_dilation(mask)

                if isinstance(self.correction_model, FluxCorr) and self.correction_model.use_flux_corr:
                    # Evaluate the correction function
                    raise NotImplementedError()
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
