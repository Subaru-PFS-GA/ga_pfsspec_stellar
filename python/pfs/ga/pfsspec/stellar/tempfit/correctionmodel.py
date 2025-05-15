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

    def apply_correction(self, spectra, corrections, correction_masks, normalization,
                         apply_flux=False, apply_mask=False, apply_normalization=False,
                         mask_bit=1, template=False):

        """
        Apply the flux correction to the spectra. This function appends the correction
        model to the spectrum and optionally applies it to the flux and sets the
        mask bits. The function modifies the objects in-place.
        
        Parameters
        ----------
        spectra : dict of list
            Dictionary of spectra for each arm and exposure.
        templates: dict of list
            Dictionary of templates for each arm and exposure.
        corrections : dict of list
            Continuum model evaluated on the wavelength grid.
        correction_masks: dict of list
            Masks where the continuum model is valid.
        normalization: dict of list
            Normalization factor
        apply_flux: bool
            Apply the continuum model to the flux. When true, the function multiplies the template
            and divides the measured flux with the continuum.
        apply_mask: bool
            Set the mask bits where the continuum model is valid. True where the
            model is valid.
        mask_bit: int
            Bit to set in the mask. The mask is a bit mask so the bit is set
            where the model is valid.
        """

        # The flux correction model always multiplies the template so it's inverse is
        # a multiplier of the original measured flux.
        
        if spectra is not None:
            for arm in spectra:
                for ei, (spec, corr, mask) in enumerate(zip(spectra[arm], corrections[arm], correction_masks[arm])):
                    if spec is not None and corr is not None:
                        self._apply_correction_impl(spec, corr,
                                                    template=template,
                                                    apply_flux=apply_flux)
                        
                        if mask is not None and apply_mask:
                            self._apply_correction_mask_impl(spec, mask,
                                                             mask_bit=mask_bit,
                                                             template=template)
                        
                        if normalization is not None and apply_normalization:
                            self._apply_normalization_impl(spec, normalization,
                                                           template=template)

    def _apply_correction_impl(self, spec, corr, template=False, apply_flux=False):
        if not template:
            if apply_flux:
                spec.multiply(1.0 / corr)
        else:
            if apply_flux:
                spec.multiply(corr)

    def _apply_correction_mask_impl(self, spec, mask, mask_bit=1, template=False):
        if spec.mask is None:
            spec.mask = mask
        elif spec.mask.dtype == bool:
            spec.mask &= mask
        else:                           
            # spec.mask is a bit mask
            # mask is True where the continuum is valid so set the bit
            # where where the mask is false
            spec.mask = np.where(mask, spec.mask, spec.mask | mask_bit)

    def _apply_normalization_impl(self, spec, norm, template=False):
        raise NotImplementedError()

    def get_wave_include(self):
        return None
    
    def get_wave_exclude(self):
        return None