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

    def append_correction(self, spectra, corrections, apply=False):
        raise NotImplementedError()
    
    def append_correction_mask(self, spectra, correction_masks):
        if self.use_flux_corr:
            for arm in spectra:
                for ei, (spec, mask) in enumerate(zip(spectra[arm], correction_masks[arm])):
                    if spec is not None and spec.mask is not None and mask is not None:
                        spec.mask &= mask

    def get_wave_include(self):
        return None
    
    def get_wave_exclude(self):
        return None