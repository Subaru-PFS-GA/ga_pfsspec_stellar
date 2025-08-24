import numpy as np

class CorrectionModel():
    """
    Base class for flux correction or continuum fitting to be
    applied to observed spectra before template fitting.
    """

    def __init__(self, trace=None, orig=None):
        if not isinstance(orig, CorrectionModel):
            self.trace = trace
            self.tempfit = None
        else:
            self.trace = trace if trace is not None else orig.trace
            self.tempfit = orig.tempfit

    def reset(self):
        pass

    def add_args(self, config, parser):
        pass

    def init_from_args(self, script, config, args):
        pass

    def init_models(self, spectra, rv_bounds=None, force=False):
        raise NotImplementedError()
        
    def calculate_coeffs(self, spectra, templates, a=None):
        raise NotImplementedError()
    
    def eval_correction(self, pp_specs, pp_temps, a=None):
        raise NotImplementedError()

    def append_model(self, spectra, corrections, correction_masks, normalization,
                     apply_mask=False, apply_normalization=False,
                     mask_bit=1):
        """
        Append the correction model to the spectra.

        This function only appends the model to the spectrum but does not modify
        the flux. It can, however, apply a mask bit to indicate where the model
        is invalid.

        Parameters:
        ----------
        spectra : dict of list
            Dictionary of spectra for each arm and exposure.
        corrections : dict of list
            Continuum model evaluated on the same wavelength grid as the spectra.
        correction_masks: dict of list
            Masks where the continuum model is valid.
        normalization: float
            Normalization factor to be applied to the correction model.
        apply_mask: bool
            Set the mask bits where the continuum model is valid.
        apply_normalization: bool
            Apply the normalization factor to the correction model.
        mask_bit: int
            Bit to set in the mask. The mask is a bit mask so the bit is set
            where the model is valid.
        """

        for arm in spectra:
            for ei, (spec, corr, mask) in enumerate(zip(spectra[arm], corrections[arm], correction_masks[arm])):
                if spec is not None and corr is not None:
                    self._append_model_impl(spec, corr, normalization, apply_normalization=apply_normalization)
                    
                    if mask is not None and apply_mask:
                        self._apply_correction_mask_impl(spec, mask, mask_bit=mask_bit)

    def apply_correction(self, spectra, template=False):

        """
        Apply the flux correction to the spectra. 
        
        Before calling this function, the correction model must have been
        appended to the spectra using `append_correction`. The function modifies
        the objects in-place.
        
        Parameters
        ----------
        spectra : dict of list
            Dictionary of spectra for each arm and exposure.
        template : bool
            If True, the `spectra` are templates and the correction model
            is applied by correcting the template to the observed flux instead
            of correcting the observed flux to the template.
        """

        # The flux correction model always multiplies the template so it's inverse is
        # a multiplier of the original measured flux.
        
        for arm in spectra:
            for ei, spec in enumerate(spectra[arm]):
                if spec is not None:
                    self._apply_correction_impl(spec, template=template)
                
    def _append_model_impl(self, spec, corr, normalization, apply_normalization=False):
        """
        Append the correction model to a spectrum. If normalization is provided and not None,
        the correction model is multiplied with the normalization factor but this behavior can
        depend on the particular implementation of the correction model.
        """

        pass

    def _apply_correction_impl(self, spec, template=False):
        """
        Apply the correction model to a spectrum. If `template` is True, the spectrum
        is a template and the correction model is applied by correcting the template.

        By default, the spectrum is multiplied with the correction model. If the spectrum
        is a template, the flux is divided by the correction model.

        Parameters
        ----------
        spec : PfsSpectrum
            Spectrum to which the correction model is to be applied.
        template : bool
            If True, the `spec` is a template and the correction model
        """

        pass

    def _apply_correction_mask_impl(self, spec, mask, mask_bit=1):
        if spec.mask is None:
            spec.mask = mask
        elif spec.mask.dtype == bool:
            spec.mask &= mask
        else:                           
            # spec.mask is a bit mask
            # mask is True where the continuum is valid so set the bit
            # where where the mask is false
            spec.mask = np.where(mask, spec.mask, spec.mask | mask_bit)

    def get_wave_include(self):
        return None
    
    def get_wave_exclude(self):
        return None