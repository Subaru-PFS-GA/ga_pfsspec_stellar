from .tempfitresults import TempFitResults

class ModelGridTempFitResults(TempFitResults):
    def __init__(self, /,
                 params_free=None, params_fit=None, params_err=None, params_mcmc=None, params_flags=None,
                 orig=None,
                 **kwargs):

        super().__init__(orig=orig, **kwargs)

        if not isinstance(orig, ModelGridTempFitResults):
            self.params_free = params_free                  # Lis of free model parameters
            self.params_fit = params_fit                    # Best fit model parameters
            self.params_err = params_err                    # Best fit model parameter uncertainties
            self.params_mcmc = params_mcmc                  # Model parameter MC samples
            self.params_flags = params_flags                # Model parameter flags 
        else:
            self.params_free = params_free if params_free is not None else orig.params_free
            self.params_fit = params_fit if params_fit is not None else orig.params_fit
            self.params_err = params_err if params_err is not None else orig.params_err
            self.params_mcmc = params_mcmc if params_mcmc is not None else orig.params_mcmc
            self.params_flags = params_flags if params_flags is not None else orig.params_flags

    @staticmethod
    def from_state(state):
        return ModelGridTempFitResults(
            rv_fit=state.rv_fit, rv_err=state.rv_err, rv_flags=state.rv_flags,
            params_free=state.params_free, params_fit=state.params_fit, params_err=state.params_err, params_flags=state.params_flags,
            a_fit=state.a_fit, a_err=state.a_err,
            cov=state.cov, cov_params=state.cov_params,
            flags=state.flags)