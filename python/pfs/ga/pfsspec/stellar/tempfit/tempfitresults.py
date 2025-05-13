class TempFitResults():
    def __init__(self, /,
                 rv_fit=None, rv_err=None, rv_mcmc=None,
                 a_fit=None, a_err=None, a_mcmc=None,
                 log_L_fit=None, log_L_mcmc=None,
                 accept_rate=None,
                 cov=None,
                 flags=None,
                 orig=None):
        
        if not isinstance(orig, TempFitResults):
            self.rv_fit = rv_fit                        # Best fit RV
            self.rv_err = rv_err                        # Best fit RV uncertainty
            self.rv_mcmc = rv_mcmc                      # RV MC samples
            self.a_fit = a_fit                          # Best fit flux corr / continuum parameters
            self.a_err = a_err
            self.a_mcmc = a_mcmc
            self.log_L_fit = log_L_fit                  # log likelihood at best fit
            self.log_L_mcmc = log_L_mcmc                # log likelihood at MC samples
            self.accept_rate = accept_rate              # MC acceptance rate
            self.cov = cov                              # Covariance matrix
            self.flags = flags                          # Flags for the fit
        else:
            self.rv_fit = rv_fit if rv_fit is not None else orig.rv_fit
            self.rv_err = rv_err if rv_err is not None else orig.rv_err
            self.rv_mcmc = rv_mcmc if rv_mcmc is not None else orig.rv_mcmc
            self.a_fit = a_fit if a_fit is not None else orig.a_fit
            self.a_err = a_err if a_err is not None else orig.a_err
            self.a_mcmc = a_mcmc if a_mcmc is not None else orig.a_mcmc
            self.log_L_fit = log_L_fit if log_L_fit is not None else orig.log_L_fit
            self.log_L_mcmc = log_L_mcmc if log_L_mcmc is not None else orig.log_L_mcmc
            self.accept_rate = accept_rate if accept_rate is not None else orig.accept_rate
            self.cov = cov if cov is not None else orig.cov
            self.flags = flags if flags is not None else orig.flags