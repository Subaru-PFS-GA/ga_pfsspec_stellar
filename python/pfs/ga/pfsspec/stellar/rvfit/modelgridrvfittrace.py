import os
import numpy as np

from pfs.ga.pfsspec.core import Trace
from pfs.ga.pfsspec.core.plotting import styles
from pfs.ga.pfsspec.core.util.args import *
from .rvfittrace import RVFitTrace

class ModelGridRVFitTrace(RVFitTrace):
    def __init__(self, outdir='.',
                 plot_inline=False,
                 plot_level=Trace.PLOT_LEVEL_NONE,
                 log_level=Trace.LOG_LEVEL_NONE):

        self.plot_params_priors = False
        self.plot_params_cov = False

        super().__init__(outdir=outdir, plot_inline=plot_inline, plot_level=plot_level, log_level=log_level)
    
    def reset(self):
        super().reset()

        self.params_iter = None

    def add_args(self, config, parser):
        super().add_args(config, parser)

    def init_from_args(self, script, config, args):
        super().init_from_args(script, config, args)

        self.plot_params_priors = get_arg('plot_params_priors', self.plot_params_priors, args)
        self.plot_params_cov = get_arg('plot_params_cov', self.plot_params_cov, args)

    def on_fit_rv_start(self, spectra, templates, 
                        rv_0, rv_bounds, rv_prior, rv_step,
                        params_0, params_bounds, params_priors, params_steps,
                        log_L_fun):
        
        super().on_fit_rv_start(spectra, templates,
                                rv_0, rv_bounds, rv_prior, rv_step,
                                log_L_fun)
        
        self.params_iter = { p: [ params_0[p] ] for p in params_0 }

        # Plot priors etc.

    def on_fit_rv_iter(self, rv, params):
        super().on_fit_rv_iter(rv)

        for p in params:
            self.params_iter[p].append(params[p])

    def on_fit_rv_finish(self, spectra, templates, processed_templates,
                         rv_0, rv_fit, rv_err, rv_bounds, rv_prior, rv_step,
                         params_0, params_fit, params_err, params_bounds, params_priors, params_steps,
                         log_L_fun):
        
        for p in params_fit:
            self.params_iter[p].append(params_fit[p])

        super().on_fit_rv_finish(spectra, templates, processed_templates,
                            rv_0, rv_fit, rv_err, rv_bounds, rv_prior, rv_step,
                            log_L_fun)

    def on_calculate_log_L(self, spectra, templates, rv, params, a):
        pass