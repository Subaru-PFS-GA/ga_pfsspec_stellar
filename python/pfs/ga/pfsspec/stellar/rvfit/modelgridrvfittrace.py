import os
import numpy as np

from pfs.ga.pfsspec.core import Trace
from pfs.ga.pfsspec.core.plotting import styles
from pfs.ga.pfsspec.core.util.args import *
from pfs.ga.pfsspec.core.plotting import DiagramPage, DiagramAxis, CornerPlot
from .rvfittrace import RVFitTrace

class ModelGridRVFitTrace(RVFitTrace):
    def __init__(self, figdir='.', logdir='.',
                 plot_inline=False,
                 plot_level=Trace.PLOT_LEVEL_NONE,
                 log_level=Trace.LOG_LEVEL_NONE):

        self.plot_params_priors = False
        self.plot_params_cov = False

        super().__init__(figdir=figdir, logdir=logdir, plot_inline=plot_inline, plot_level=plot_level, log_level=log_level)
    
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
                         params_free, cov,
                         log_L_fun):
        
        for p in params_fit:
            self.params_iter[p].append(params_fit[p])

        super().on_fit_rv_finish(spectra, templates, processed_templates,
                            rv_0, rv_fit, rv_err, rv_bounds, rv_prior, rv_step,
                            log_L_fun)
        
        # TODO: move it to a function and remove duplicate lines
        # Plot corner plot of parameters
        nparam = len(params_fit) + 1        # including RV
        f = self.get_diagram_page('rvfit_cov', npages=1, nrows=nparam, ncols=nparam)

        # Collect the axes from the free parameters and RV
        axes = []
        priors = []
        for p in params_free:
            # limits = params_bounds[p]
            limits = (params_fit[p] - 10 * params_err[p], params_fit[p] + 10 * params_err[p])
            axes.append(DiagramAxis(limits, label=p))
            priors.append((params_priors[p], limits, None, None))

        # limits = rv_bounds
        limits = (rv_fit - 10 * rv_err, rv_fit + 10 * rv_err)
        axes.append(DiagramAxis(limits, label='RV'))
        priors.append((rv_prior, limits, None, None))

        cc = CornerPlot(f, axes)

        # Plot the covariance contours
        mu = np.array([ params_fit[p] for p in params_free ] + [ rv_fit ])
        if cov is not None and not np.any(np.isnan(cov)):
            cc.plot_covariance(mu, cov, sigma=[1, 2, 3])
        
        # Plot the best fit values with error bars
        args = [ (params_fit[p], params_err[p]) for p in params_free ] + [ (rv_fit, rv_err) ]
        cc.errorbar(*args, sigma=[1, 2, 3])

        # Plot the priors            
        cc.plot_priors(*priors, normalize=True)

        self.flush_figures()

    def on_calculate_log_L(self, spectra, templates, rv, params, a):
        pass