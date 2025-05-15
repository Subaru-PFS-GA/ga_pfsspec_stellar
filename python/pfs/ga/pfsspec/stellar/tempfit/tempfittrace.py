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
        self.plot_rv_convergence = False
        self.plot_input_spec = False
        self.plot_fit_spec = {}

        self.plot_continuum_fit_start = False
        self.plot_continuum_fit_iter = False
        self.plot_continuum_fit_end = False

        self.reset()

    def reset(self):
        super().reset()

        self.rv_iter = None                    # Keep track of convergence
        self.log_L_iter = None
        self.guess_rv_results = None

        self.continuum_fit_iter = 0

    def add_args(self, config, parser):
        Trace.add_args(self, config, parser)
        SpectrumTrace.add_args(self, config, parser)
    
    def init_from_args(self, script, config, args):
        Trace.init_from_args(self, script, config, args)
        SpectrumTrace.init_from_args(self, script, config, args)

        self.plot_priors = get_arg('plot_priors', self.plot_priors, args)
        self.plot_rv_guess = get_arg('plot_rv_guess', self.plot_rv_guess, args)
        self.plot_rv_fit = get_arg('plot_rv_fit', self.plot_rv_fit, args)
        self.plot_rv_convergence = get_arg('plot_rv_convergence', self.plot_rv_convergence, args)
        self.plot_input_spec = get_arg('plot_input_spec', self.plot_input_spec, args)
        self.plot_fit_spec = get_arg('plot_fit_spec', self.plot_fit_spec, args)

        self.plot_continuum_fit_start = get_arg('plot_continuum_fit_start', self.plot_continuum_fit_start, args)
        self.plot_continuum_fit_iter = get_arg('plot_continuum_fit_iter', self.plot_continuum_fit_iter, args)
        self.plot_continuum_fit_end = get_arg('plot_continuum_fit_end', self.plot_continuum_fit_end, args)

    #endregion
    #region Trace hooks

    def on_fit_rv_start(self, spectra, templates, 
                        rv_0, rv_bounds, rv_prior, rv_step,
                        log_L_0, log_L_fun,
                        wave_include=None, wave_exclude=None):
        
        self.rv_iter = [ rv_0 ]
        self.log_L_iter = [ log_L_0 ]

        if self.plot_input_spec:
            self._plot_spectra('pfsGA-tempfit-input-{id}',
                               spectra=spectra,
                               plot_flux=True, plot_flux_err=True,
                               title='TempFit input spectra - {id}',
                               wave_include=wave_include, wave_exclude=wave_exclude)
            self.flush_figures()
        
        if self.plot_priors:
            self._plot_prior('pfsGA-tempfit-rv-prior-{id}', rv_prior, rv_bounds, rv_0, rv_step,
                             title='Prior on RV - {id}')
            self.flush_figures()

    def on_guess_rv(self, rv, log_L, rv_guess, log_L_guess, log_L_fit, function, pp, pcov):
        # Called when the RV guess is made

        # Save results for later plotting
        self.guess_rv_results = (rv, log_L, rv_guess, log_L_guess, log_L_fit, function, pp, pcov)

        if self.plot_rv_guess or self.plot_level >= Trace.PLOT_LEVEL_INFO:
            self._plot_rv_fit('pfsGA-tempfit-rv-guess-{id}',
                              rv=rv, log_L=log_L, log_L_fit=log_L_fit,
                              rv_guess=rv_guess,
                              title='RV guess - {id}')
    
    def on_fit_rv_iter(self, rv, log_L, log_L_fun):
        self.rv_iter.append(rv)
        self.log_L_iter.append(log_L)
    
    def on_fit_rv_finish(self, spectra, templates, processed_templates, 
                         rv_0, rv_fit, rv_err, rv_bounds, rv_prior, rv_step, rv_fixed,
                         log_L_0, log_L_fit, log_L_fun):
        
        self.rv_iter.append(rv_fit)
        self.log_L_iter.append(log_L_fit)
        
        # Plot the final results based on the configuration settings
        for key, config in self.plot_fit_spec.items():
            self._plot_spectra(key, spectra, templates=processed_templates, **config)

        # Plot rv_fit and rv_guess and the likelihood function
        if self.plot_rv_fit:
            if self.guess_rv_results is not None:
                rv, log_L, rv_guess, log_L_guess, log_L_fit, function, pp, pcov = self.guess_rv_results
            else:
                rv, log_L, rv_guess, log_L_guess, log_L_fit, function, pp, pcov = None, None, None, None, None, None, None

            self._plot_rv_fit(
                'pfsGA-tempfit-rv-fit-{id}',
                rv=rv, log_L=log_L, log_L_fit=log_L_fit,
                rv_0=rv_0, rv_bounds=rv_bounds, rv_prior=rv_prior, rv_step=rv_step,
                rv_guess=rv_guess, rv_fit=rv_fit, rv_err=rv_err,
                title='TempFit results - {id}')
            
        # Plot a zoom-in of the likelihood function
        if self.plot_rv_fit and log_L_fun is not None and rv_fit is not None and rv_err is not None:

            rv = np.linspace(rv_fit - 3 * rv_err, rv_fit + 3 * rv_err, 100)
            log_L = log_L_fun(rv)

            self._plot_rv_fit(
                'pfsGA-tempfit-rv-fit-zoom-{id}',
                rv=rv, log_L=log_L, log_L_fit=log_L_fit,
                rv_guess=rv_guess, rv_0=rv_0,
                rv_fit=rv_fit, rv_err=rv_err,
                title='TempFit results zoom-in - {id}')

        # Plot the convergence of RV, if available
        if self.plot_rv_convergence and self.rv_iter is not None and self.log_L_iter is not None:
            rv = np.array(self.rv_iter)
            log_L = np.array(self.log_L_iter)

            self._plot_rv_convergence('pfsGA-tempfit-rv-converge-{id}',
                                      rv=rv, log_L=log_L,
                                      title='RV convergence - {id}',)
            
        self.flush_figures()

    def on_process_spectrum(self, arm, i, spectrum, processed_spectrum):
        if self.plot_level >= Trace.PLOT_LEVEL_TRACE:

            self._plot_spectrum(f'pfsGA-tempfit-spectrum-{arm}-{i}-{{id}}', arm,
                                spectrum=spectrum,
                                plot_spectrum=True,
                                title='Observed spectrum - {id}')
            
            self._plot_spectrum(f'pfsGA-tempfit-spectrum-processed-{arm}-{i}-{{id}}', arm,
                                spectrum=spectrum, processed_spectrum=processed_spectrum,
                                plot_spectrum=False, plot_processed_spectrum=True,
                                title='Processed spectrum - {id}')

            self.flush_figures()

        if self.log_level >= Trace.LOG_LEVEL_NONE:
            self._save_spectrum_history(f'pfsGA-tempfit-spectrum-{arm}-{i}-{{id}}.log', spectrum)

        self.inc_counter('process_spectrum')

    def on_process_template(self, arm, rv, template, processed_template):
        if self.plot_level >= Trace.PLOT_LEVEL_TRACE:

            self._plot_spectrum(f'pfsGA-tempfit-template-{arm}-{{id}}', arm,
                                template=template,
                                plot_template=True,
                                title='Original template - {id}')
            
            self._plot_spectrum(f'pfsGA-tempfit-template-processed-{arm}-{{id}}', arm,
                                template=template, processed_template=processed_template,
                                plot_template=False, plot_processed_template=True,
                                title='Convolved template - {id}')

            self.flush_figures()

        self.inc_counter('process_template')

    def on_resample_template(self, arm, rv, spectrum, template, resampled_template):
        if self.get_counter('resample_template') is None:
            if self.plot_level:

                self._plot_spectrum(f'pfsGA-tempfit-template-resampled-{arm}-{{id}}', arm,
                                    template=resampled_template,
                                    title='Resampled template - {id}')

                self.flush_figures()

            if self.log_level >= Trace.LOG_LEVEL_NONE:
                self._save_spectrum_history(f'pfsGA-tempfit-resampled-template-{arm}-{{id}}.log', resampled_template)

        self.inc_counter('resample_template')

    def on_template_cache_hit(self, template, rv_q, rv):
        self.inc_counter('template_cache_hit')
    
    def on_template_cache_miss(self, template, rv_q, rv):
        self.inc_counter('template_cache_miss')

    def on_calculate_log_L(self, spectra, templates, rv, log_L):
        pass

    # Callbacks related to flux correction

    def on_eval_flux_corr_basis(self, spectra, basis):
        self.inc_counter('eval_flux_corr_basis')

    def on_eval_phi_chi(self, spectra, templates, bases, phi, chi):
        self.inc_counter('eval_phi_chi')

    def on_eval_log_L_phi_chi(self, phi, chi, log_L):
        self.inc_counter('eval_log_L_phi_chi')

    def on_eval_log_L_a(self, phi, chi, a, log_L):
        self.inc_counter('eval_log_L_a')

    def on_eval_nu2(self, phi, chi, nu2):
        self.inc_counter('eval_nu2')

    #endregion
    # Callbacks related to continuum fitting

    def on_continuum_fit_start(self, spec):
        if self.plot_continuum_fit_start or self.plot_level >= Trace.PLOT_LEVEL_TRACE:
            self._plot_spectrum('pfsGA-continuumfit-{id}', 
                                spectrum=spec,
                                plot_flux=True, plot_continuum=False,
                                title='Continuum fit input spectrum - {id}')

    def on_continuum_fit_finish(self, spec):
        if self.plot_continuum_fit_end or self.plot_level >= Trace.PLOT_LEVEL_TRACE:
            self._plot_spectrum('pfsGA-continuumfit-{id}', 
                                spectrum=spec,
                                plot_flux=True, plot_continuum=True,
                                title='Continuum fit results - {id}')
            self.flush_figures()

    def on_continuum_fit_function_iter(self, piece_id, iter, x, y, w, model, mask):
        self.continuum_fit_iter += 1

        if self.plot_continuum_fit_iter or self.plot_level >= Trace.PLOT_LEVEL_TRACE:
            raise NotImplementedError()

    #endregion Plotting
                
    def _plot_prior(self, key, param_prior, param_bounds, param_0, param_step,
                    title=None):

        f = self.get_diagram_page(key, npages=1, nrows=1, ncols=1,
                                  title=title,
                                  page_size=(5.5, 3.5))
        p = DistributionPlot()
        ax = f.add_diagram((0, 0, 0), p)

        p.plot_prior(param_prior, param_bounds, param_0, param_step)

    def _plot_rv_fit(self, key, /,
                     rv=None, log_L=None, log_L_fit=None,
                     rv_0=None, rv_bounds=None, rv_prior=None, rv_step=None,
                     rv_fit=None, rv_err=None,  
                     rv_guess=None,
                     title=None):
        
        """
        Plot the RV prior, initial value, guess and final fitted value

        Parameters
        ----------
        key : str
            Key for the plot
        rv : array-like
            RV values where log L is calculated
        log_L : array-like
            Log likelihood values at RV
        rv_0 : float
            Initial value of RV
        rv_bounds : array-like
            Bounds on RV
        rv_prior : Distribution
            Prior on RV
        rv_step : float
            Step size for RV
        rv_fit : float
            Fitted value of RV
        rv_err : float
            Error on RV
        rv_guess : float
            Guessed value of RV
        """

        f = self.get_diagram_page(key, npages=1, nrows=1, ncols=1,
                                  title=title,
                                  page_size=(5.5, 3.5))
        ax = f.add_axes((0, 0, 0))

        text = ''

        # Plot likelihood function, if provided
        if rv is not None and log_L is not None:
            m = ~np.isnan(log_L) & ~np.isinf(log_L)
            a = np.max(log_L[m])
            b = np.min(log_L[m])
            ax.plot(rv, (log_L - b) / (a - b), '.')

            if log_L_fit is not None:
                ax.plot(rv, (log_L_fit - b) / (a - b), **styles.solid_line())

        if rv_guess is not None:
            ax.axvline(rv_guess, **styles.blue_line(**styles.solid_line()))
            text += f'$v_\\mathrm{{los, guess}} = {rv_guess:0.2f}$ km s$^{{-1}}$\n'

        if rv_0 is not None:
            text += f'$v_\\mathrm{{los, 0}} = {rv_0:0.2f}$ km s$^{{-1}}$\n'

        # Plot the prior on RV
        if rv_prior is not None and rv_bounds is not None:
            p = DistributionPlot(ax)
            ax = f.add_diagram((0, 0, 0), p)
            p.plot_prior(rv_prior, rv_bounds, rv_0, rv_step, normalize=True)

        # Plot fit results
        if rv_fit is not None:
            ax.axvline(rv_fit, **styles.red_line(**styles.solid_line()))

            if rv_err is not None:
                ax.axvline(rv_fit - 3 * rv_err, **styles.red_line(**styles.dashed_line()))
                ax.axvline(rv_fit + 3 * rv_err, **styles.red_line(**styles.dashed_line()))

                text += f'$v_\\mathrm{{los, fit}} = {rv_fit:0.2f} \\pm {rv_err:0.2f}$ km s$^{{-1}}$\n'
            else:
                text += f'$v_\\mathrm{{los, fit}} = {rv_fit:0.2f}$ km s$^{{-1}}$\n'

        ax.text(0.95, 0.95, text, transform=ax.transAxes, ha='right', va='top')
        ax.set_xlabel(R'$v_\mathrm{los}$ [km s$^{-1}$]')
        ax.set_ylabel('normalized log-posterior')

        self.flush_figures()

    def _plot_rv_convergence(self, key, /, rv=None, log_L=None, title=None):
        
        f = self.get_diagram_page(key, npages=1, nrows=1, ncols=1,
                                  title=title,
                                  page_size=(5.5, 3.5))
        ax = f.add_axes((0, 0, 0))
        ax.plot(rv, '-k')
        ax2 = ax.twinx()
        ax2.plot(log_L, '-b')
        ax.set_xlabel('iteration')
        ax.set_ylabel('RV [km s$^{-1}$]')
        ax2.set_ylabel('log-posterior')