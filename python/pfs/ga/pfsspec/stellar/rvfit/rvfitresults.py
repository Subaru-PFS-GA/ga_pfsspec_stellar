class RVFitResults():
    def __init__(self, orig=None,
                 rv_fit=None, rv_err=None, rv_mcmc=None,
                 a_fit=None, a_err=None, a_mcmc=None,
                 params_fit=None, params_err=None, params_mcmc=None,
                 log_L_fit=None, log_L_mcmc=None, accept_rate=None, cov=None):
        
        if not isinstance(orig, RVFitResults):
            self.rv_fit = rv_fit
            self.rv_err = rv_err
            self.rv_mcmc = rv_mcmc
            self.a_fit = a_fit
            self.a_err = a_err
            self.a_mcmc = a_mcmc
            self.params_fit = params_fit
            self.params_err = params_err
            self.params_mcmc = params_mcmc
            self.log_L_fit = log_L_fit
            self.log_L_mcmc = log_L_mcmc
            self.accept_rate = accept_rate
            self.cov = cov
        else:
            self.rv_fit = rv_fit if rv_fit is not None else orig.rv_fit
            self.rv_err = rv_err if rv_err is not None else orig.rv_err
            self.rv_mcmc = rv_mcmc if rv_mcmc is not None else orig.rv_mcmc
            self.a_fit = a_fit if a_fit is not None else orig.a_fit
            self.a_err = a_err if a_err is not None else orig.a_err
            self.a_mcmc = a_mcmc if a_mcmc is not None else orig.a_mcmc
            self.params_fit = params_fit if params_fit is not None else orig.params_fit
            self.params_err = params_err if params_err is not None else orig.params_err
            self.params_mcmc = params_mcmc if params_mcmc is not None else orig.params_mcmc
            self.log_L_fit = log_L_fit if log_L_fit is not None else orig.log_L_fit
            self.log_L_mcmc = log_L_mcmc if log_L_mcmc is not None else orig.log_L_mcmc
            self.accept_rate = accept_rate if accept_rate is not None else orig.accept_rate
            self.cov = cov if cov is not None else orig.cov