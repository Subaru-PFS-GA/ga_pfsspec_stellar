import numpy as np

from pfs.ga.pfsspec.core.util.copy import safe_deep_copy
from pfs.ga.pfsspec.core.util.args import get_arg
from ..continuum.models import Spline
from ..continuum.finders import Uniform
from .correctionmodel import CorrectionModel

class ContNorm(CorrectionModel):
    """
    Model class for continuum normalization based stellar template fitting.
    """

    def __init__(self, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, ContNorm):
            self.use_cont_norm = False          # Fit the continuum, use with normalized templates, exclusive with use_flux_corr
            self.cont_finder_type = None
            self.cont_finder = None             # Continuum pixel finder, optionally for each arm
            self.cont_model_type = Spline
            self.cont_model = None              # Continuum model, optionally for each arm
            self.cont_per_arm = False           # Do continuum fit independently for each arm
            self.cont_per_fiber = False         # Do continuum fit independently for each fiber
            self.cont_per_exp = False           # Do continuum fit independently for each exposurepass
        else:
            self.use_cont_norm = orig.use_cont_norm
            self.cont_finder = orig.cont_finder
            self.cont_model = orig.cont_model
            self.cont_per_arm = orig.cont_per_arm
            self.cont_per_fiber = orig.cont_per_fiber
            self.cont_per_exp = orig.cont_per_exp

        self.reset()

    def reset(self):
        super().reset()

        pass

    def add_args(self, config, parser):
        super().add_args(config, parser)

    def init_from_args(self, script, config, args):
        super().init_from_args(script, config, args) 

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

        return self.cont_model_type()

    def init_models(self, spectra, rv_bounds=None, force=False):
        if self.use_cont_norm:
            # Initialize the continuum model and continuum finder
            if self.use_cont_norm and (self.cont_model is None or force):
                self.cont_model = self.tempfit.init_model(spectra,
                                                          per_arm=self.cont_per_arm,
                                                          per_exp=self.cont_per_exp,
                                                          rv_bounds=rv_bounds,
                                                          round_to=100,
                                                          create_model_func=self.create_continuum_model)

            if self.use_cont_norm and (self.cont_finder is None or force):
                self.cont_finder = self.tempfit.init_model(spectra,
                                                           per_arm=self.cont_per_arm,
                                                           per_exp=self.cont_per_exp,
                                                           rv_bounds=rv_bounds,
                                                           round_to=100,
                                                           create_model_func=self.create_continuum_finder)
                
    def get_coeff_count(self, spectra: dict):
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
            self.eval_continuum_fit(spectra, templates, a)
            # TODO: evaluate the models to get cont
            raise NotImplementedError()
        else:
            # If the coefficients are not supplied, we have to fit the continuum
            # to the ratio of the observed flux and the templates
            coeffs, continua = self.fit_continuum(spectra, templates)

        # Sum up the log likelihoods for each arm and exposure
        log_L = 0
        for arm in spectra:
            for ei, (spec, temp, cont) in enumerate(zip(spectra[arm], templates[arm], continua[arm])):
                if spec is not None:
                    flux = spec.flux
                    sigma2 = spec.sigma2
                    mask = spec.mask
                    log_L += 0.5 * np.sum((flux[mask] - cont[mask] * temp.flux[mask]) ** 2 / sigma2[mask])
                else:
                    # Missing exposure has no contribution to the likelihood
                    pass
                
        return log_L

    def fit_continuum(self, spectra, templates):
        """
        Fit the continuum model to the ratio of the spectra and the templates or,
        if the templates are not provides, to the continuum directly.
        
        Templates are assumed to be Doppler shifted to a certain RV and resampled to
        the same grid as the spectra. Templates are also assumed to be normalized so
        taking the ratio of the measured flux effectively removes the absorption lines.
        """

        def get_pixels_per_exp(arm, spec, temp):
            """
            Return the valid pixels for a single exposure
            """

            if temp is not None:
                flux = spec.flux / temp.flux
                flux_err = spec.flux_err / temp.flux if spec.flux_err is not None else None
            else:
                flux = spec.flux
                flux_err = spec.flux_err

            return spec.wave, flux, flux_err, spec.mask
        
        def get_pixels_all_exp(arm, spectra, templates):
            """
            Return the valid pixels for all exposures in the arm
            """

            wave = []
            flux = []
            flux_err = []
            mask = []
            for ie, (spec, temp) in enumerate(zip(spectra, templates)):
                if spec is not None:
                    w, f, fe, m = get_pixels_per_exp(arm, spec, temp)
                    wave.append(w)
                    flux.append(f)
                    flux_err.append(fe)
                    mask.append(m)
                else:
                    wave.append(None)
                    flux.append(None)
                    flux_err.append(None)
                    mask.append(None)

            return wave, flux, flux_err, mask
        
        def fit_continuum_per_exp(continuum_model, continuum_finder, wave, flux, flux_err, mask):
            """
            Fit continuum to a single exposure
            """

            params = continuum_model.fit(wave, flux, flux_err, mask=mask, continuum_finder=continuum_finder)
            _, cont = continuum_model.eval(params)
            
            return params, cont
        
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
            if continuum_finder is None:
                all_mask = np.concatenate(mask)
                params = continuum_model.fit(all_wave, all_flux, all_flux_err, all_mask)
                
                # Evaluate model for each exposure
                for i in range(len(flux)):
                    _, cont[i] = continuum_model.eval(params, wave=wave[i])
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
                        _, cont[i] = continuum_model.eval(params, wave=wave[i])

                    # Find the continuum pixels for each exposure
                    more_iter = False
                    for i in range(len(flux)):
                        mask[i], mi = continuum_finder.find(iter, wave[i], flux[i], mask=mask[i], cont=cont[i])
                        more_iter |= mi

                    iter += 1

            return params, cont

        if self.cont_per_arm:
            coeffs = {}
            continua = {}
            for arm in spectra:
                if self.cont_per_exp:
                    # Fit each exposure separately
                    coeffs[arm] = []
                    continua[arm] = []
                    for ei, (spec, temp, model, finder) in enumerate(zip(spectra[arm], templates[arm],
                                                                         self.cont_model[arm], self.cont_finder[arm])):
                        if spec is not None:
                            wave, flux, flux_err, mask = get_pixels_per_exp(arm, spec, temp)
                            p, c = fit_continuum_per_exp(model, finder,
                                                         wave, flux, flux_err, mask)
                            coeffs[arm].append(p)
                            continua[arm].append(c)
                        else:
                            coeffs[arm].append(None)
                            continua[arm].append(None)
                else:
                    # Fit all exposures simultaneously
                    model = self.cont_model[arm]
                    finder = self.cont_finder[arm]
                    wave, flux, flux_err, mask = get_pixels_all_exp(arm, spectra[arm], templates[arm])
                    p, c = fit_continuum_all_exp(model, finder,
                                                 wave, flux, flux_err, mask)
                    coeffs[arm] = p
                    continua[arm] = c
        else:
            raise Exception('Continuum fitting using a single model for all arms is not supported yet.')

        return coeffs, continua
    
    def eval_continuum_fit(self, spectra, templates, coeffs):
        """
        Evaluate the continuum fit for each exposure in each arm. Templates
        are assumed to be Doppler shifted to a certain RV and resampled to
        the same grid as the spectra.
        """

        if self.cont_per_arm:
            continua = {}
            for arm in spectra:
                if self.cont_per_exp:
                    # Fit each exposure separately
                    continua[arm] = []
                    for ei, (spec, a, model) in enumerate(zip(spectra[arm], coeffs[arm], self.cont_model[arm],)):
                        if spec is not None:
                            _, cont = model.eval(a, wave=spec.wave)
                            continua[arm].append(cont)
                        else:
                            continua[arm].append(None)
                else:
                    # Fit all exposures simultaneously
                    continua[arm] = []
                    for ei, spec in enumerate(spectra[arm]):
                        if spec is not None:
                            model = self.cont_model[arm]
                            a = coeffs[arm]
                            _, cont = model.eval(a, wave=spec.wave)
                            continua[arm].append(cont)
                        else:
                            continua[arm].append(None)
        else:
            raise Exception('Continuum fitting using a single model for all arms is not supported yet.')
        
        return continua
    
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
            a, _ = self.fit_continuum(spectra, templates)
            
        # a = self.concat_coeffs(a)
        # if a.ndim > 1:
        #     a = a.squeeze(0)

        return a

    def apply_correction(self, spectra, templates, a=None):
        """
        Apply the continuum correction to pre-processed templates. Templates
        are assumed to be Doppler shifted to a certain RV and resampled to the
        same grid as the spectra.

        Parameters
        ----------
        spectra : dict of list
            Dictionary of spectra for each arm and exposure.
        templates : dict of list
            Dictionary of templates for each arm and exposure.
        a : array
            Continuum model parameters.
        """

        if self.use_cont_norm:
            if a is None:
                a, continua = self.fit_continuum(spectra, templates)
            else:
                continua = self.eval_continuum_fit(spectra, templates, a)

            for arm in spectra:
                for ei, (spec, temp, cont) in enumerate(zip(spectra[arm], templates[arm], continua[arm])):
                    if temp is not None and cont is not None:
                        temp.multiply(cont)