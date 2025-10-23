from pfs.ga.pfsspec.core.util.copy import safe_deep_copy

class TempFitState():
    """
    State object for temperature fitting.
    """

    def __init__(self, spectra, templates, fluxes=None, orig=None):
        if not isinstance(orig, TempFitState):
            self.spectra = spectra
            self.templates = templates
            self.fluxes = fluxes

            self.rv_0 = None
            self.rv_fixed = None
            self.rv_guess = None
            self.rv_bounds = None
            self.rv_prior = None
            self.rv_step = None
            self.rv_fit = None
            self.rv_err = None
            self.rv_flags = None

            self.a_fit = None
            self.a_err = None

            self.spec_norm = None
            self.temp_norm = None
            self.template_wlim = None

            self.pp_spec = None
            self.log_L_fun = None
            self.pack_params = None
            self.unpack_params = None
            self.pack_bounds = None

            self.x_0 = None
            self.steps = None
            self.bounds = None
            self.flags = None

            self.log_L_0 = None
            self.log_L_guess = None
        else:
            self.spectra = spectra if spectra is not None else orig.spectra
            self.templates = templates if templates is not None else orig.templates
            self.fluxes = fluxes if fluxes is not None else orig.fluxes

            self.rv_0 = safe_deep_copy(orig.rv_0)
            self.rv_fixed = safe_deep_copy(orig.rv_fixed)
            self.rv_guess = safe_deep_copy(orig.rv_guess)
            self.rv_bounds = safe_deep_copy(orig.rv_bounds)
            self.rv_prior = safe_deep_copy(orig.rv_prior)
            self.rv_step = safe_deep_copy(orig.rv_step)
            self.rv_fit = safe_deep_copy(orig.rv_fit)
            self.rv_err = safe_deep_copy(orig.rv_err)
            self.rv_flags = safe_deep_copy(orig.rv_flags)

            self.a_fit = safe_deep_copy(orig.a_fit)
            self.a_err = safe_deep_copy(orig.a_err)

            self.spec_norm = orig.spec_norm
            self.temp_norm = orig.temp_norm
            self.template_wlim = safe_deep_copy(orig.template_wlim)

            self.pp_spec = None
            self.log_L_fun = orig.log_L_fun
            self.pack_params = orig.pack_params
            self.unpack_params = orig.unpack_params
            self.pack_bounds = orig.pack_bounds

            self.x_0 = safe_deep_copy(orig.x_0)
            self.steps = safe_deep_copy(orig.steps)
            self.bounds = safe_deep_copy(orig.bounds)
            self.flags = safe_deep_copy(orig.flags)

            self.log_L_0 = safe_deep_copy(orig.log_L_0)
            self.log_L_guess = safe_deep_copy(orig.log_L_guess)