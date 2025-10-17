class TempFitResults():
    def __init__(self, /,
                 rv_fit=None, rv_err=None, rv_mcmc=None, rv_flags=None,
                 a_fit=None, a_err=None, a_mcmc=None,
                 log_L_fit=None, log_L_mcmc=None,
                 accept_rate=None,
                 cov=None, cov_params=None,
                 flags=None,
                 orig=None):
        
        if not isinstance(orig, TempFitResults):
            self.rv_fit = rv_fit                        # Best fit RV
            self.rv_err = rv_err                        # Best fit RV uncertainty
            self.rv_mcmc = rv_mcmc                      # RV MC samples
            self.rv_flags = rv_flags                    # RV flags
            self.a_fit = a_fit                          # Best fit flux corr / continuum parameters
            self.a_err = a_err
            self.a_mcmc = a_mcmc
            self.log_L_fit = log_L_fit                  # log likelihood at best fit
            self.log_L_mcmc = log_L_mcmc                # log likelihood at MC samples
            self.accept_rate = accept_rate              # MC acceptance rate
            self.cov = cov                              # Covariance matrix
            self.cov_params = cov_params          # Indexes of parameters in the covariance matrix
            self.flags = flags                          # Flags for the fit
        else:
            self.rv_fit = rv_fit if rv_fit is not None else orig.rv_fit
            self.rv_err = rv_err if rv_err is not None else orig.rv_err
            self.rv_mcmc = rv_mcmc if rv_mcmc is not None else orig.rv_mcmc
            self.rv_flags = rv_flags if rv_flags is not None else orig.rv_flags
            self.a_fit = a_fit if a_fit is not None else orig.a_fit
            self.a_err = a_err if a_err is not None else orig.a_err
            self.a_mcmc = a_mcmc if a_mcmc is not None else orig.a_mcmc
            self.log_L_fit = log_L_fit if log_L_fit is not None else orig.log_L_fit
            self.log_L_mcmc = log_L_mcmc if log_L_mcmc is not None else orig.log_L_mcmc
            self.accept_rate = accept_rate if accept_rate is not None else orig.accept_rate
            self.cov = cov if cov is not None else orig.cov
            self.cov_params = cov_params if cov_params is not None else orig.cov_params
            self.flags = flags if flags is not None else orig.flags

    @staticmethod
    def from_state(state):
        return TempFitResults(
            rv_fit=state.rv_fit, rv_err=state.rv_err, rv_flags=state.rv_flags,
            a_fit=state.a_fit, a_err=state.a_err,
            log_L_fit=state.log_L_fit,
            cov=state.cov, cov_params=state.cov_params,
            flags=state.flags)