from pfs.ga.pfsspec.core.util.copy import safe_deep_copy

from .tempfitstate import TempFitState

class ModelGridTempFitState(TempFitState):
    """
    State object for temperature fitting using a model grid.
    """

    def __init__(self, spectra=None, templates=None, fluxes=None, orig=None):
        super().__init__(spectra=spectra, templates=templates, fluxes=fluxes, orig=orig)

        if not isinstance(orig, ModelGridTempFitState):
            self.params_0 = None
            self.params_fixed = None
            self.params_guess = None
            self.params_bounds = None
            self.params_priors = None
            self.params_steps = None
            self.params_free = None
            self.params_fit = None
            self.params_err = None
            self.params_flags = None

            self.cov = None
            self.cov_params = None
        else:
            self.params_0 = safe_deep_copy(orig.params_0)
            self.params_fixed = safe_deep_copy(orig.params_fixed)
            self.params_guess = safe_deep_copy(orig.params_guess)
            self.params_bounds = safe_deep_copy(orig.params_bounds)
            self.params_priors = safe_deep_copy(orig.params_priors)
            self.params_steps = safe_deep_copy(orig.params_steps)
            self.params_free = safe_deep_copy(orig.params_free)
            self.params_fit = safe_deep_copy(orig.params_fit)
            self.params_err = safe_deep_copy(orig.params_err)
            self.params_flags = safe_deep_copy(orig.params_flags)

            self.cov = safe_deep_copy(orig.cov)
            self.cov_params = safe_deep_copy(orig.cov_params)