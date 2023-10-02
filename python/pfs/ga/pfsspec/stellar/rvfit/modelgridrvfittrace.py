import os

from .rvfittrace import RVFitTrace

class ModelGridRVFitTrace(RVFitTrace):
    
    def reset(self):
        super().reset()

        self.params_iter = None

    def on_prepare_fit(self, rv_0, rv_bounds, rv_step,
                       params_0, params_fixed, params_free, params_bounds, params_steps):
        super().on_prepare_fit(rv_0, rv_bounds, rv_step)

        # if self.params_iter is None:
        #     self.params_iter = {}
        # for p in params_0:
        #     if p not in self.params_iter:
        #         self.params_iter[p] = []
        #     self.params_iter[p].append(params_0[p])

    def on_fit_rv_iter(self, rv, params):
        super().on_fit_rv_iter(rv)

        if self.params_iter is None:
            self.params_iter = {}
        for p in params:
            if p not in self.params_iter:
                self.params_iter[p] = []
            self.params_iter[p].append(params[p])

    def on_fit_rv(self, spectra, templates, rv, params):
        super().on_fit_rv(spectra, templates, rv)

        if self.params_iter is None:
            self.params_iter = {}
        for p in params:
            if p not in self.params_iter:
                self.params_iter[p] = []
            self.params_iter[p].append(params[p])

    def on_calculate_log_L(self, spectra, templates, rv, params, a):
        pass