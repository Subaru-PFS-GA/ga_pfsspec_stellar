import numpy as np

from pfs.ga.pfsspec.core.util.copy import safe_deep_copy
from pfs.ga.pfsspec.core.util.args import get_arg
from ..continuum.models import Spline
from ..continuum.finders import Uniform
from .correctionmodel import CorrectionModel
from .tempfit import TempFit

class ContNorm(CorrectionModel):
    """
    Model class for continuum normalization based stellar template fitting.
    """

    def __init__(self, trace=None, orig=None):
        super().__init__(trace=trace, orig=orig)

        if not isinstance(orig, ContNorm):
            self.use_cont_norm = True           # Fit the continuum, use with normalized templates
            self.cont_finder_type = None
            self.cont_finder = None             # Continuum pixel finder, optionally for each arm
            self.cont_model_type = Spline
            self.cont_per_arm = True            # Do continuum fit independently for each arm
            self.cont_per_fiber = False         # Do continuum fit independently for each fiber
            self.cont_per_exp = True            # Do continuum fit independently for each exposurepass

            self.cont_model = None              # Continuum model, optionally for each arm

            self.cont_wave_include = None            # Include only these wavelengths in the continuum fit
            self.cont_wave_exclude = None            # Exclude these wavelengths from the continuum fit

        else:
            self.use_cont_norm = orig.use_cont_norm
            self.cont_finder = orig.cont_finder
            self.cont_per_arm = orig.cont_per_arm
            self.cont_per_fiber = orig.cont_per_fiber
            self.cont_per_exp = orig.cont_per_exp

            self.cont_model = orig.cont_model

            self.cont_wave_include = orig.wave_include
            self.cont_wave_exclude = orig.wave_exclude

        self.reset()

    def reset(self):
        super().reset()

        pass

    def add_args(self, config, parser):
        super().add_args(config, parser)

        parser.add_argument('--cont-norm', action='store_true', dest='cont_norm', help='Do continuum normalization.\n')
        parser.add_argument('--no-cont-norm', action='store_false', dest='cont_norm', help='No continuum normalization.\n')

        # TODO: add option to set continuum finder and continuum type

        parser.add_argument('--cont-per-arm', action='store_true', dest='cont_per_arm', help='Continuum normalization per arm.\n')
        parser.add_argument('--cont-per-fiber', action='store_true', dest='cont_per_fiber', help='Continuum normalization per fiber.\n')
        parser.add_argument('--cont-per-exp', action='store_true', dest='cont_per_exp', help='Continuum normalization per exposure.\n')

    def init_from_args(self, script, config, args):
        super().init_from_args(script, config, args) 

        self.use_cont_norm = get_arg('cont_norm', self.use_cont_norm, args)
        
        # TODO

        self.cont_per_arm = get_arg('cont_per_arm', self.cont_per_arm, args)
        self.cont_per_fiber = get_arg('cont_per_fiber', self.cont_per_fiber, args)
        self.cont_per_exp = get_arg('cont_per_exp', self.cont_per_exp, args)

        self.cont_wave_include = get_arg('cont_wave_include', self.cont_wave_include, args)
        self.cont_wave_exclude = get_arg('cont_wave_exclude', self.cont_wave_exclude, args)

    def create_continuum_finder(self, wlim):
        """
        Instantiate a continuum finder
        """

        # TODO: pass in argumens

        if self.cont_finder_type is None:
            return None
        else:
            return self.cont_finder_type()

    def create_continuum_model(self, wlim):
        """
        Instantiate a continuum model.
        """

        # TODO: pass in argumens

        return self.cont_model_type(trace=self.trace)

    def init_models(self, spectra, rv_bounds=None, force=False):
        if self.use_cont_norm:
            # Initialize the continuum model and continuum finder
            if self.use_cont_norm and (self.cont_model is None or force):
                self.cont_model = self.tempfit.init_correction_model(spectra,
                                                                     per_arm=self.cont_per_arm,
                                                                     per_exp=self.cont_per_exp,
                                                                     rv_bounds=rv_bounds,
                                                                     round_to=100,
                                                                     create_model_func=self.create_continuum_model)

            if self.use_cont_norm and (self.cont_finder is None or force):
                self.cont_finder = self.tempfit.init_correction_model(spectra,
                                                                      per_arm=self.cont_per_arm,
                                                                      per_exp=self.cont_per_exp,
                                                                      rv_bounds=rv_bounds,
                                                                      round_to=100,
                                                                      create_model_func=self.create_continuum_finder)
                
    def get_coeff_count(self):
        """
        Return the number of continuum model parameters
        """
        
        # The continuum model is considered to depend only on the observation, hence it
        # has no free parameters
            
        return 0

    def eval_log_L(self, spectra, templates, /, a=None):
        """
        Calculate the value of the likelihood function at a given value of RV.

        Templates are assumed to be Doppler shifted to a certain RV and resampled to
        the same grid as the spectra.

        This function does not add the log of priors and must not be called directly.
        """

        if a is not None:
            a = self.split_coeffs(spectra, a)
            continua = self.eval_continuum_fit(spectra, templates, a)
        else:
            # If the coefficients are not supplied, we have to fit the continuum
            # to the ratio of the observed flux and the templates
            coeffs, continua, masks = self.fit_continuum(spectra, templates)

        # Sum up the log likelihoods for each arm and exposure
        log_L = 0
        for arm, ei, mi, spec in self.tempfit.enumerate_spectra(spectra,
                                                                per_arm=self.cont_per_arm,
                                                                per_exp=self.cont_per_exp,
                                                                include_none=False):
            temp = templates[arm][ei]
            cont = continua[arm][ei]
            mask = masks[arm][ei] & spec.mask & temp.mask

            log_L -= 0.5 * np.sum((spec.flux[mask] - cont[mask] * temp.flux[mask]) ** 2 / spec.sigma2[mask])

        return log_L
    
    def fit_continuum(self, pp_spec, pp_temp):
        """
        Fit the continuum model to the ratio of the spectra and the templates or,
        if the templates are not provided, to the continuum directly.
        
        Templates are assumed to be Doppler shifted to a certain RV and resampled to
        the same grid as the spectra. Templates are also assumed to be normalized so
        taking the ratio of the measured flux effectively removes the absorption lines.

        Depending on the configuration, the models are fitted to single exposures or
        to all exposures simultaneously but the continuum model is always defined on a
        per arm basis and, unlike flux correction, cannot be done for all arms at once.

        Parameters
        ----------
        pp_spec : dict of list
            Dictionary of preprocessed spectra for each arm and exposure.
        pp_temp : dict of list
            Dictionary of preprocessed templates for each arm and exposure.
        """

        def get_pixels_per_exp(arm, spec, temp):
            """
            Return the continuum normalized flux in every pixel of the spectrum.
            """

            flux = spec.flux / temp.flux
            flux_err = spec.flux_err / temp.flux if spec.flux_err is not None else None

            return spec.wave, flux, flux_err, spec.mask & temp.mask
        
        def fit_continuum_all_exp(continuum_model, continuum_finder, wave, flux, flux_err, mask):
            """
            Fit the continuum of multiple exposures simultaneously
            """

            all_wave = np.concatenate(wave)
            all_flux = np.concatenate(flux)
            all_flux_err = np.concatenate(flux_err)

            # Instead of passing the continuum finder to the continuum model to handle the
            # iterations of continuum fitting, we do it here since we have to fit multiple
            # exposures in parallel.

            cont = [ None for f in flux ]      # initial continuum
            cont_mask = [ None for f in flux ] # mask for the continuum pixels

            if continuum_finder is None:
                all_mask = np.concatenate(mask)
                params = continuum_model.fit(all_wave, all_flux, all_flux_err, all_mask)
                
                # Evaluate model for each exposure
                for i in range(len(flux)):
                    _, cont[i], cont_mask[i] = continuum_model.eval(params, wave=wave[i])
            else:
                # Perform iterations of the continuum fit
                iter = 0
                more_iter = True
                while more_iter:
                    # Fit the model to all pixels at once
                    # TODO: make sure that continuum models don't depend on the order of wave
                    all_mask = np.concatenate(mask)
                    params = continuum_model.fit(all_wave, all_flux, all_flux_err, all_mask)

                    # Evaluate model for each exposure
                    for i in range(len(flux)):
                        _, cont[i], cont_mask[i] = continuum_model.eval(params, wave=wave[i])

                    # Find the continuum pixels for each exposure
                    more_iter = False
                    for i in range(len(flux)):
                        mask[i], mi = continuum_finder.find(iter, wave[i], flux[i], mask=mask[i], cont=cont[i])
                        more_iter |= mi

                    iter += 1

            return params, cont, cont_mask
        
        if not self.cont_per_arm:
            raise Exception('Continuum fitting using a single model for all arms is not supported yet.')

        # The list `cont_model` is already populated for each model index
        # Loop through all spectra and add the flux of unmasked pixels to a list
        # that corresponds to the continuum models.
        
        r = range(len(self.cont_model))
        key = { i: [] for i in r }
        wave = { i: [] for i in r }
        flux = { i: [] for i in r }
        flux_err = { i: [] for i in r }
        mask = { i: [] for i in r }
        
        for arm, ei, mi, spec in self.tempfit.enumerate_spectra(pp_spec,
                                                                per_arm=self.cont_per_arm,
                                                                per_exp=self.cont_per_exp,
                                                                include_none=False):
            
            key[mi].append((arm, ei))

            temp = pp_temp[arm][ei]
            w, f, fe, m = get_pixels_per_exp(arm, spec, temp)

            # Apply the continuum fit range definitions
            # TODO: this could be cached but it's not clear where's the best place to do so
            m &= TempFit.get_wave_mask(w, self.cont_wave_include, self.cont_wave_exclude)

            wave[mi].append(w)
            flux[mi].append(f)
            flux_err[mi].append(fe)
            mask[mi].append(m)

        # Combine all pixels for each model index and fit the model and fit the model
        # Model parameter are returned in a dictionary keyed by the model index,
        # whereas the continua are returned in a dict of lists keyed by the arm.
        coeffs = {}
        continua = { arm: [] for arm in pp_spec }
        masks = { arm: [] for arm in pp_spec }
        for arm, ei, _, _ in self.tempfit.enumerate_spectra(pp_spec,
                                                            per_arm=False, per_exp=False,
                                                            include_none=True):
            continua[arm].append(None)
            masks[arm].append(None)

        # Perform the fitting for each continuum model. This will results in a continuum fit
        # per arm and/or per exposure, depending on the settings
        for mi in r:
            model = self.cont_model[mi]
            finder = self.cont_finder[mi]
            params, cont, cont_mask = fit_continuum_all_exp(model, finder, wave[mi], flux[mi], flux_err[mi], mask[mi])
            coeffs[mi] = params

            # Continua are returned in a list so they have to be sorted out by arms and exposures
            for i, (arm, ei) in enumerate(key[mi]):
                continua[arm][ei] = cont[i]
                masks[arm][ei] = cont_mask[i]

        return coeffs, continua, masks
    
    def eval_continuum_fit(self, spectra, templates, coeffs):
        """
        Evaluate the continuum fit for each exposure in each arm. Templates
        are assumed to be Doppler shifted to a certain RV and resampled to
        the same grid as the spectra.
        """

        if not self.cont_per_arm:
            raise Exception('Continuum fitting using a single model for all arms is not supported yet.')

        continua = { arm: [] for arm in spectra }
        masks = { arm: [] for arm in spectra }

        for arm, ei, mi, spec in self.tempfit.enumerate_spectra(spectra,
                                                                per_arm=self.cont_per_arm,
                                                                per_exp=self.cont_per_exp,
                                                                include_none=True):
            
            if spec is not None:
                model = self.cont_model[mi]
                a = coeffs[mi]
                _, cont, mask = model.eval(a, wave=spec.wave)
                continua[arm].append(cont)
                masks[arm].append(mask)
            else:
                continua[arm].append(None)
                masks[arm].append(None)
        
        return continua, masks
    
    def concat_coeffs(self, a):
        """
        Concatenate the continuum model parameters into a single 1d array.
        """
        
        def concat_coeffs_helper(a):
            if isinstance(a, dict):
                return np.concatenate([ concat_coeffs_helper(a[arm]) for arm in sorted(a.keys()) ])
            elif isinstance(a, list):
                return np.concatenate([ concat_coeffs_helper(x) for x in a ])
            else:
                return np.atleast_1d(a)
            
        return concat_coeffs_helper(a)

    def split_coeffs(self, spectra, a):
        """
        Split the continuum model parameters into a dictionary.
        """

        raise NotImplementedError

    def calculate_coeffs(self, spectra, templates, a=None):
        """
        Given a set of spectra and preprocessed templates, determine the
        continuum model parameters.

        The continuum model parameters are determined by actual fitting of the
        continuum.

        If the continuum model parameters are provided in `a`, they're returned
        without modification.
        """

        if a is None:
            a, _, _ = self.fit_continuum(spectra, templates)
        
        return a

    def eval_correction(self, pp_specs, pp_temps, a=None):
        if self.use_cont_norm:
            if a is None:
                a, continua, masks = self.fit_continuum(pp_specs, pp_temps)
            else:
                continua, masks = self.eval_continuum_fit(pp_specs, pp_temps, a)
        else:
            continua = None
            masks = None

        return continua, masks

    def _apply_correction_impl(self, spec, corr, template=False, apply_flux=False):
        super()._apply_correction_impl(spec, corr, template=template, apply_flux=apply_flux)

        spec.cont = corr    

    def _apply_normalization_impl(self, spec, norm, template=False):
        spec.cont *= norm
                            
    def get_wave_include(self):
        return self.cont_wave_include
    
    def get_wave_exclude(self):
        return self.cont_wave_exclude