from .tempfitresults import TempFitResults

class ModelGridTempFitResults(TempFitResults):
    def __init__(self, /,
                 rv_fit=None, rv_err=None, rv_mcmc=None,
                 a_fit=None, a_err=None, a_mcmc=None,
                 params_free=None, params_fit=None, params_err=None, params_mcmc=None,
                 log_L_fit=None, log_L_mcmc=None,
                 accept_rate=None,
                 cov=None,
                 orig=None):

        super().__init__(rv_fit=rv_fit, rv_err=rv_err, rv_mcmc=rv_mcmc,
                         a_fit=a_fit, a_err=a_err, a_mcmc=a_mcmc,
                         log_L_fit=log_L_fit, log_L_mcmc=log_L_mcmc,
                         accept_rate=accept_rate,
                         cov=cov,
                         orig=orig)

        if not isinstance(orig, ModelGridTempFitResults):
            self.params_free = params_free                  # Lis of free model parameters
            self.params_fit = params_fit                    # Best fit model parameters
            self.params_err = params_err                    # Best fit model parameter uncertainties
            self.params_mcmc = params_mcmc                  # Model parameter MC samples
        else:
            self.params_free = params_free if params_free is not None else orig.params_free
            self.params_fit = params_fit if params_fit is not None else orig.params_fit
            self.params_err = params_err if params_err is not None else orig.params_err
            self.params_mcmc = params_mcmc if params_mcmc is not None else orig.params_mcmc