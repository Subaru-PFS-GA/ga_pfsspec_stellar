import os
import numpy as np

from pfs.ga.pfsspec.core.plotting import SpectrumPlot, DistributionPlot, styles
from pfs.ga.pfsspec.core import Trace, SpectrumTrace
from pfs.ga.pfsspec.core.util.args import *

class TempFitTrace(Trace, SpectrumTrace):
    """
    Implements call-back function to profile and debug RV fitting. Allows for
    generating plots of intermediate steps.
    """

    #region Initializers

    def __init__(self,
                 id=None,
                 figdir='.', logdir='.',
                 plot_inline=False, 
                 plot_level=Trace.PLOT_LEVEL_NONE, 
                 log_level=Trace.LOG_LEVEL_NONE):
        
        Trace.__init__(self, id=id,
                       figdir=figdir, logdir=logdir,
                       plot_inline=plot_inline, 
                       plot_level=plot_level,
                       log_level=log_level)
        
        SpectrumTrace.__init__(self)
        
        self.plot_priors = False
        self.plot_rv_guess = False
        self.plot_rv_fit = False
        self.plot_input_spec = False
        self.plot_fit_spec = {}

        self.reset()

    def reset(self):
        self.process_spectrum_count = 0
        self.process_template_count = 0
        self.resample_template_count = 0
        self.template_cache_hit_count = 0
        self.template_cache_miss_count = 0
        self.eval_phi_chi_count = 0
        self.eval_log_L_phi_chi_count = 0
        self.eval_log_L_a_count = 0

        self.rv_iter = None                    # Keep track of convergence

        self.rv_guess = None

    def add_args(self, config, parser):
        Trace.add_args(self, config, parser)
        SpectrumTrace.add_args(self, config, parser)
    
    def init_from_args(self, script, config, args):
        Trace.init_from_args(self, script, config, args)
        SpectrumTrace.init_from_args(self, script, config, args)

        self.plot_priors = get_arg('plot_priors', self.plot_priors, args)
        self.plot_rv_guess = get_arg('plot_rv_guess', self.plot_rv_guess, args)
        self.plot_rv_fit = get_arg('plot_rv_fit', self.plot_rv_fit, args)
        self.plot_input_spec = get_arg('plot_input_spec', self.plot_input_spec, args)
        self.plot_fit_spec = get_arg('plot_fit_spec', self.plot_fit_spec, args)

    #endregion
    #region Trace hooks

    def on_fit_rv_start(self, spectra, templates, 
                        rv_0, rv_bounds, rv_prior, rv_step,
                        log_L_fun):
        
        self.rv_iter = [ rv_0 ]

        if self.plot_input_spec:
            self._plot_spectra('pfsGA-RVfit-input-{id}', spectra,
                               title='RVFit input spectra - {id}')
            self.flush_figures()
        
        if self.plot_priors:
            self._plot_prior('pfsGA-RVFit-RV-prior-{id}', rv_prior, rv_bounds, rv_0, rv_step,
                             title='Prior on RV - {id}')
            self.flush_figures()

    def on_guess_rv(self, rv, log_L, rv_guess, log_L_fit, function, pp, pcov):
        # Called when the RV guess is made

        # Store the results for plotting when fitting is done
        self.guess_rv_results = (rv, log_L, rv_guess, log_L_fit, function, pp, pcov)

        if self.plot_rv_guess or self.plot_level >= Trace.PLOT_LEVEL_INFO:
            self._plot_rv_guess('pfsGA-RVFit-RV-guess-{id}',
                                title='RV guess - {id}')
    
    def on_fit_rv_iter(self, rv):
        self.rv_iter.append(rv)
    
    def on_fit_rv_finish(self, spectra, templates, processed_templates, 
                         rv_0, rv_fit, rv_err, rv_bounds, rv_prior, rv_step, rv_fixed,
                         log_L_fun):
        
        self.rv_iter.append(rv_fit)
        
        # Plot the final results based on the configuration settings
        for key, config in self.plot_fit_spec.items():
            self._plot_spectra(key, spectra, templates=templates, processed_templates=processed_templates,
                               **config)

        if self.plot_rv_fit:
            self._plot_rv_guess('pfsGA-RVFit-RV-fit-{id}',
                                rv_0, rv_bounds, rv_prior, rv_step,
                                rv_fit, rv_err,
                                title='RVFit results - {id}')

        self.flush_figures()

    def on_process_spectrum(self, arm, i, spectrum, processed_spectrum):
        if self.plot_level >= Trace.PLOT_LEVEL_TRACE:

            self._plot_spectrum(f'pfsGA-RVFit-spectrum-{arm}-{i}-{{id}}', arm,
                                spectrum=spectrum,
                                plot_spectrum=True,
                                title='Observed spectrum - {id}')
            
            self._plot_spectrum(f'pfsGA-RVFit-spectrum-processed-{arm}-{i}-{{id}}', arm,
                                spectrum=spectrum, processed_spectrum=processed_spectrum,
                                plot_spectrum=False, plot_processed_spectrum=True,
                                title='Processed spectrum - {id}')

            self.flush_figures()

        if self.log_level >= Trace.LOG_LEVEL_NONE:
            self._save_spectrum_history(f'pfsGA-RVFit-spectrum-{arm}-{i}-{{id}}.log', spectrum)

        self.process_spectrum_count += 1

    def on_process_template(self, arm, rv, template, processed_template):
        if self.plot_level >= Trace.PLOT_LEVEL_TRACE:

            self._plot_spectrum(f'pfsGA-RVFit-template-{arm}-{{id}}', arm,
                                template=template,
                                plot_template=True,
                                title='Original template - {id}')
            
            self._plot_spectrum(f'pfsGA-RVFit-template-processed-{arm}-{{id}}', arm,
                                template=template, processed_template=processed_template,
                                plot_template=False, plot_processed_template=True,
                                title='Processed template - {id}')

            self.flush_figures()

        self.process_template_count += 1

    def on_resample_template(self, arm, rv, spectrum, template, resampled_template):
        if self.resample_template_count == 0:
            if self.plot_level:

                self._plot_spectrum(f'pfsGA-RVFit-template-resampled-{arm}-{{id}}', arm,
                                    template=template, processed_template=resampled_template,
                                    plot_template=False, plot_processed_template=True,
                                    title='Resampled template - {id}')

                self.flush_figures()

            if self.log_level >= Trace.LOG_LEVEL_NONE:
                self._save_spectrum_history(f'pfsGA-RVFit-resampled-template-{arm}-{{id}}.log', resampled_template)

        self.resample_template_count += 1

    def on_template_cache_hit(self, template, rv_q, rv):
        self.template_cache_hit_count += 1
    
    def on_template_cache_miss(self, template, rv_q, rv):
        self.template_cache_miss_count += 1

    def on_calculate_log_L(self, spectra, templates, rv, log_L):
        pass

    # Callbacks related to flux correction

    def on_eval_flux_corr_basis(self, spectra, basis):
        pass

    def on_eval_phi_chi(self, spectra, templates, bases, log_L, phi, chi):
        self.eval_phi_chi_count += 1

    def on_eval_log_L_phi_chi(self, phi, chi, log_L):
        self.eval_log_L_phi_chi_count += 1

    def on_eval_log_L_a(self, phi, chi, a, log_L):
        self.eval_log_L_a_count += 1

    def on_eval_F_mcmc(self, x, log_L):
        pass

    #endregion
                
    def _plot_prior(self, key, param_prior, param_bounds, param_0, param_step,
                    title=None):

        f = self.get_diagram_page(key, npages=1, nrows=1, ncols=1,
                                  title=title,
                                  page_size=(5.5, 3.5))
        p = DistributionPlot()
        ax = f.add_diagram((0, 0, 0), p)

        p.plot_prior(param_prior, param_bounds, param_0, param_step)

    def _plot_rv_guess(self, key,
                       rv_0=None, rv_bounds=None, rv_prior=None, rv_step=None,
                       rv_fit=None, rv_err=None,
                       title=None):
        
        # Plot the RV prior, initial value, guess and final fitted value

        f = self.get_diagram_page(key, npages=1, nrows=1, ncols=1,
                                  title=title,
                                  page_size=(5.5, 3.5))
        ax = f.add_axes((0, 0, 0))

        text = ''

        # Plot RV guess results
        if self.guess_rv_results is not None and self.plot_rv_guess:
            (rv, log_L, rv_guess, fit, function, pp, pcov) = self.guess_rv_results

            m = ~np.isnan(log_L) & ~np.isinf(log_L)
            a = np.nanmax(log_L[m])
            b = np.nanmin(log_L[m])
            ax.plot(rv, (log_L - b) / (a - b), '.')
            if fit is not None:
                ax.plot(rv, (fit - b) / (a - b), **styles.solid_line())
            ax.axvline(rv_guess, **styles.blue_line(**styles.solid_line()))

            text += f'$v_\\mathrm{{los, guess}} = {rv_guess:0.2f}$ km s$^{-1}$\n'

        # Plot the prior on RV
        if self.plot_priors:
            p = DistributionPlot(ax)
            ax = f.add_diagram((0, 0, 0), p)
            p.plot_prior(rv_prior, rv_bounds, rv_0, rv_step, normalize=True)

        # Plot fit results
        if self.plot_rv_fit and rv_fit is not None:
            ax.axvline(rv_fit, **styles.red_line(**styles.solid_line()))

            if rv_err is not None:
                ax.axvline(rv_fit - 3 * rv_err, **styles.red_line(**styles.dashed_line()))
                ax.axvline(rv_fit + 3 * rv_err, **styles.red_line(**styles.dashed_line()))

                text += f'$v_\\mathrm{{los, fit}} = {rv_fit:0.2f} \\pm {rv_err:0.2f}$ km s$^{-1}$\n'
            else:
                text += f'$v_\\mathrm{{los, fit}} = {rv_fit:0.2f}$ km s$^{-1}$\n'

        ax.text(0.95, 0.95, text, transform=ax.transAxes, ha='right', va='top')

        self.flush_figures()