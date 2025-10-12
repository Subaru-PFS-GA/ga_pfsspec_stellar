import os
import numpy as np

from pfs.ga.pfsspec.core import Trace
from pfs.ga.pfsspec.core.plotting import styles
from pfs.ga.pfsspec.core.util.args import *
from pfs.ga.pfsspec.core.plotting import DiagramPage, DiagramAxis, CornerPlot
from .tempfittrace import TempFitTrace

class ModelGridTempFitTrace(TempFitTrace):
    def __init__(self,
                 id=None,
                 figdir='.', logdir='.',
                 plot_inline=False,
                 plot_level=Trace.PLOT_LEVEL_NONE,
                 log_level=Trace.LOG_LEVEL_NONE):

        self.plot_params_priors = False
        self.plot_params_cov = False
        self.plot_params_convergence = False
        self.plot_log_L_map = False

        super().__init__(id=id, figdir=figdir, logdir=logdir, plot_inline=plot_inline, plot_level=plot_level, log_level=log_level)
    
    def reset(self):
        super().reset()

        self.params_iter = None

    def add_args(self, config, parser):
        super().add_args(config, parser)

    def init_from_args(self, script, config, args):
        super().init_from_args(script, config, args)

        self.plot_params_priors = get_arg('plot_params_priors', self.plot_params_priors, args)
        self.plot_params_cov = get_arg('plot_params_cov', self.plot_params_cov, args)
        self.plot_params_convergence = get_arg('plot_params_convergence', self.plot_params_convergence, args)
        self.plot_log_L_map = get_arg('plot_log_L_map', self.plot_log_L_map, args)

    def on_fit_rv_start(self, spectra, templates, 
                        rv_0, rv_bounds, rv_prior, rv_step,
                        params_0, params_bounds, params_priors, params_steps,
                        log_L_0, log_L_fun,
                        wave_include=None, wave_exclude=None):
        
        super().on_fit_rv_start(spectra, templates,
                                rv_0, rv_bounds, rv_prior, rv_step,
                                log_L_0, log_L_fun,
                                wave_include=wave_include, wave_exclude=wave_exclude)
        
        self.params_iter = { p: [ params_0[p] ] for p in params_0 }

        # Plot priors etc.

    def on_fit_rv_iter(self, rv, params, log_L, log_L_fun):
        super().on_fit_rv_iter(rv, log_L, log_L_fun)

        for p in params:
            if p in self.params_iter:
                self.params_iter[p].append(params[p])

    def on_fit_rv_finish(self, spectra, templates,
                         rv_0, rv_fit, rv_err, rv_bounds, rv_prior, rv_step, rv_fixed,
                         params_0, params_fit, params_err, params_bounds, params_priors, params_steps, params_free,
                         cov,
                         log_L_0, log_L_fit, log_L_fun):
        
        for p in params_fit:
            if p in self.params_iter:
                self.params_iter[p].append(params_fit[p])

        super().on_fit_rv_finish(spectra, templates,
                            rv_0, rv_fit, rv_err, rv_bounds, rv_prior, rv_step, rv_fixed,
                            log_L_0, log_L_fit, log_L_fun)
        
        if self.plot_rv_fit is None and self.plot_level >= Trace.PLOT_LEVEL_INFO \
            or self.plot_rv_fit:
            
            # Corner plot of parameters
            self._plot_fit_results(rv_0, rv_fit, rv_err, rv_bounds, rv_prior, rv_step, rv_fixed,
                                      params_0, params_fit, params_err, params_bounds, params_priors, params_steps, params_free,
                                      cov)

        # Plot the convergence of template params, if available
        if (self.plot_params_convergence is None and self.plot_level >= Trace.PLOT_LEVEL_TRACE \
            or self.plot_params_convergence) and self.params_iter is not None:
            
            params = { p: np.array(v) for p, v in self.params_iter.items() }
            log_L = np.array(self.log_L_iter)

            self._plot_params_convergence('pfsGA-tempfit-params-converge-{id}',
                                      params=params, log_L=log_L,
                                      title='Parameter convergence - {id}')

        self.flush_figures()
            
    def on_map_log_L(self, spectra, templates, params, log_L):

        if self.plot_log_L_map:
            # Cannot plot a map for less than 1 or more than 2 parameters
            if len(params) == 2:
                pnames = '_'.join(list(params.keys()))
                self._plot_log_L_map(f'pfsGA-tempfit-logL-map-{pnames}-{{id}}',
                                    params=params,
                                    log_L=log_L)
        
        self.flush_figures()
        
    def _plot_fit_results(self,
                          rv_0, rv_fit, rv_err, rv_bounds, rv_prior, rv_step, rv_fixed,
                          params_0, params_fit, params_err, params_bounds, params_priors, params_steps, params_free,
                          cov):

        # TODO: move it to a function and remove duplicate lines
        # Plot corner plot of parameters
        nparam = len(params_fit)
        if not rv_fixed:
            nparam += 1
        f = self.get_diagram_page('pfsGA-tempfit-params-{id}',
                                  title='TempFit parameters - {id}',
                                  npages=1, nrows=nparam, ncols=nparam,
                                  gutters=(0.25, 0.25))

        # Collect the axes from the free parameters and RV
        axes = []
        priors = []
        for p in params_free:
            # limits = params_bounds[p]
            if not (np.isnan(params_fit[p]) or np.isnan(params_err[p])):
                limits = (params_fit[p] - 10 * params_err[p], params_fit[p] + 10 * params_err[p])
            else:
                limits = None
            axes.append(DiagramAxis(limits, label=p))
            priors.append((params_priors[p] if params_priors is not None and p in params_priors else None,
                           limits, None, None))

        # limits = rv_bounds
        if not rv_fixed:
            limits = (rv_fit - 10 * rv_err, rv_fit + 10 * rv_err)
            axes.append(DiagramAxis(limits, label='RV'))
            priors.append((rv_prior, limits, None, None))

        cc = CornerPlot(f, axes)

        # Plot the covariance contours
        all_params = [ params_fit[p] for p in params_free ]
        if not rv_fixed:
            all_params.append(rv_fit)
        mu = np.array(all_params)
        if cov is not None and not np.any(np.isnan(cov)):
            cc.plot_covariance(mu, cov, sigma=[1, 2, 3])
        
        # Plot the best fit values with error bars
        all_params = [ (params_fit[p], params_err[p]) for p in params_free ]
        if not rv_fixed:
            all_params.append((rv_fit, rv_err))
        cc.errorbar(*all_params, sigma=[1, 2, 3])

        # Plot the priors            
        cc.plot_priors(*priors, normalize=True)

        # Print the final best fit parameters and their errors
        cc.print_parameters(*all_params)

    def _plot_params_convergence(self, key, /, params=None, log_L=None, title=None):
        
        f = self.get_diagram_page(key, npages=1, nrows=len(params) + 1, ncols=1,
                                  title=title)

        ax = f.add_axes((0, 0, 0))
        ax.plot(log_L, '-k')
        ax.set_xlabel('iteration')
        ax.set_ylabel('log-posterior')
        
        for i, p in enumerate(params):
            ax = f.add_axes((0, i + 1, 0))
            ax.plot(params[p], '-k')
            ax.set_xlabel('iteration')
            ax.set_ylabel(p)

    def _plot_log_L_map(self, key, /, params=None, log_L=None, title=None):

        f = self.get_diagram_page(key, npages=1, nrows=1, ncols=1,
                                  title=title,
                                  page_size=(4, 4))

        labels = list(params.keys())
        axes = list(params.values())

        ax = f.add_axes((0, 0, 0))

        ax.imshow(log_L, aspect='auto', origin='lower',
                  extent=[axes[0][0], axes[0][-1], axes[1][0], axes[1][-1]],)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])