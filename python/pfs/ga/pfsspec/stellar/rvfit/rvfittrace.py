import os
import numpy as np

from pfs.ga.pfsspec.core.plotting import SpectrumPlot, DistributionPlot, styles
from pfs.ga.pfsspec.core import Trace
from pfs.ga.pfsspec.core.util.args import *

class RVFitTrace(Trace):
    """
    Implements call-back function to profile and debug RV fitting. Allows for
    generating plots of intermediate steps.
    """

    #region Initializers

    def __init__(self, figdir='.', logdir='.',
                 plot_inline=False, 
                 plot_level=Trace.PLOT_LEVEL_NONE, 
                 log_level=Trace.LOG_LEVEL_NONE):
        
        super().__init__(figdir=figdir, logdir=logdir,
                         plot_inline=plot_inline, 
                         plot_level=plot_level,
                         log_level=log_level)
        
        self.plot_priors = False
        self.plot_rv_guess = False
        self.plot_rv_fit = False
        self.plot_input_spec = False
        self.plot_fit_spec = {}
        self.plot_spec_flux_err = False
        self.plot_spec_mask = False
        self.plot_spec_cont = False

        self.reset()

    def reset(self):
        self.process_spectrum_count = 0
        self.process_template_count = 0
        self.resample_template_count = 0
        self.template_cache_hit_count = 0
        self.template_cache_miss_count = 0
        self.eval_phi_chi_count = 0
        self.eval_log_L_count = 0
        self.eval_log_L_a_count = 0

        self.rv_iter = None                    # Keep track of convergence

        self.rv_guess = None

    def add_args(self, config, parser):
        super().add_args(config, parser)
    
    def init_from_args(self, script, config, args):
        super().init_from_args(script, config, args)

        self.plot_priors = get_arg('plot_priors', self.plot_priors, args)
        self.plot_rv_guess = get_arg('plot_rv_guess', self.plot_rv_guess, args)
        self.plot_rv_fit = get_arg('plot_rv_fit', self.plot_rv_fit, args)
        self.plot_input_spec = get_arg('plot_input_spec', self.plot_input_spec, args)
        self.plot_fit_spec = get_arg('plot_fit_spec', self.plot_fit_spec, args)
        self.plot_spec_flux_err = get_arg('plot_spec_flux_err', self.plot_spec_flux_err, args)
        self.plot_spec_mask = get_arg('plot_spec_mask', self.plot_spec_mask, args)
        self.plot_spec_cont = get_arg('plot_spec_cont', self.plot_spec_cont, args)

    #endregion
    #region Trace hooks

    def on_fit_rv_start(self, spectra, templates, 
                        rv_0, rv_bounds, rv_prior, rv_step,
                        log_L_fun):
        
        self.rv_iter = [ rv_0 ]

        if self.plot_input_spec:
            self._plot_spectra('rvfit_input', spectra)
            self.flush_figures()
        
        if self.plot_priors:
            self._plot_prior('rvfit_rv_prior', rv_prior, rv_bounds, rv_0, rv_step)
            self.flush_figures()

    def on_guess_rv(self, rv, log_L, rv_guess, log_L_fit, function, pp, pcov):
        # Called when the RV guess is made

        # Store the results for plotting when fitting is done
        self.guess_rv_results = (rv, log_L, rv_guess, log_L_fit, function, pp, pcov)

        if self.plot_rv_guess or self.plot_level >= Trace.PLOT_LEVEL_INFO:
            self._plot_rv_guess('rvfit_rv_guess')
    
    def on_fit_rv_iter(self, rv):
        self.rv_iter.append(rv)
    
    def on_fit_rv_finish(self, spectra, templates, processed_templates, 
                         rv_0, rv_fit, rv_err, rv_bounds, rv_prior, rv_step,
                         log_L_fun):
        
        self.rv_iter.append(rv_fit)
        
        # TODO: plot whatever requested
        for key, config in self.plot_fit_spec.items():
            self._plot_spectra(key, spectra, templates=templates, processed_templates=processed_templates,
                               **config)

        if self.plot_rv_fit:
            self._plot_rv_guess('rvfit_rv_fit',
                                rv_0, rv_bounds, rv_prior, rv_step,
                                rv_fit, rv_err)

        self.flush_figures()

    def on_process_spectrum(self, arm, i, spectrum, processed_spectrum):
        if self.plot_level >= Trace.PLOT_LEVEL_TRACE:
            self.plot_spectrum(f'spectrum_{arm}_{i}', arm, spectrum.wave, spectrum.flux)
            self.plot_spectrum(f'processed_spectrum_{arm}_{i}', arm, processed_spectrum.wave, processed_spectrum.flux)
            self.flush_figures()

        if self.log_level >= Trace.LOG_LEVEL_NONE:
            self.save_history(f'spectrum_{arm}_{i}.log', spectrum)

        self.process_spectrum_count += 1

    def on_process_template(self, arm, rv, template, processed_template):
        if self.plot_level >= Trace.PLOT_LEVEL_TRACE:
            self.plot_spectrum(f'template_{arm}', arm, template.wave, template.flux)
            self.plot_spectrum(f'processed_template_{arm}', arm, processed_template.wave, processed_template.flux)
            self.flush_figures()

        self.process_template_count += 1

    def on_resample_template(self, arm, rv, spectrum, template, resampled_template):
        if self.resample_template_count == 0:
            if self.plot_level:
                self.plot_spectrum(f'resampled_template_{arm}', arm, resampled_template.wave, resampled_template.flux)
                self.flush_figures()

            if self.log_level >= Trace.LOG_LEVEL_NONE:
                self.save_history(f'resampled_template_{arm}.log', resampled_template)

        self.resample_template_count += 1

    def on_template_cache_hit(self, template, rv_q, rv):
        self.template_cache_hit_count += 1
    
    def on_template_cache_miss(self, template, rv_q, rv):
        self.template_cache_miss_count += 1

    def on_eval_flux_corr_basis(self, spectra, basis):
        pass

    def on_eval_phi_chi(self, rv, spectra, templates, bases, sigma2, weights, masks, log_L, phi, chi):
        self.eval_phi_chi_count += 1

    def on_eval_log_L(self, phi, chi, log_L):
        self.eval_log_L_count += 1

    def on_eval_log_L_a(self, phi, chi, a, log_L):
        self.eval_log_L_a_count += 1

    def on_eval_F_mcmc(self, x, log_L):
        pass

    #endregion

    def save_history(self, filename, spectrum):
        """
        Save the processing history of a spectrum
        """
        if spectrum.history is not None:
            fn = os.path.join(self.logdir, filename)
            self.make_outdir(fn)
            with open(fn, "w") as f:
                f.writelines([ s + '\n' for s in spectrum.history ])
                
    def _plot_spectra(self, key, spectra, templates=None, processed_templates=None,
                      wlim=None,
                      plot_spectra=True, plot_flux_err=True,
                      plot_templates=True,
                      plot_processed_templates=True,
                      plot_residuals=False):
        # Number of exposures
        nexp = np.max([ len(spectra[arm]) for arm in spectra.keys() ])

        ncols = 1
        nrows = 4
        npages = int(np.ceil(nexp / (ncols * nrows)))
        f = self.get_diagram_page(key, npages, nrows, ncols, diagram_size=(6.5, 2.0))
       
        for i, (j, k, l) in enumerate(np.ndindex((npages, nrows, ncols))):
            if i == nexp:
                break

            p = SpectrumPlot()
            ax = f.add_diagram((j, k, l), p)

            p.plot_mask = self.plot_spec_mask
            p.plot_flux_err = self.plot_spec_flux_err
            p.plot_cont = self.plot_spec_cont

            for arm, specs in spectra.items():
                spec = specs[i]

                # TODO: define arm color in styles
                if plot_spectra:
                    p.plot_spectrum(specs[i], plot_flux_err=plot_flux_err, wlim=wlim, auto_limits=True)

                if plot_templates and templates is not None:
                    raise NotImplementedError()

                if plot_processed_templates and processed_templates is not None:
                    temp = processed_templates[arm][i]
                    p.plot_template(temp, wlim=wlim)

                if plot_residuals and processed_templates is not None:
                    temp = processed_templates[arm][i]
                    p.plot_residual(spec, temp, wlim=wlim)

            # TODO: Add SNR, exp time, obs date
            p.title = spec.get_id_string()

            p.apply()

        f.match_limits()

    def _plot_prior(self, key, param_prior, param_bounds, param_0, param_step):
        f = self.get_diagram_page(key, npages=1, nrows=1, ncols=1, page_size=(5.5, 3.5))
        p = DistributionPlot()
        ax = f.add_diagram((0, 0, 0), p)

        p.plot_prior(param_prior, param_bounds, param_0, param_step)

    def _plot_rv_guess(self, key,
                       rv_0=None, rv_bounds=None, rv_prior=None, rv_step=None,
                       rv_fit=None, rv_err=None):
        
        # Plot the RV prior, initial value, guess and final fitted value

        f = self.get_diagram_page(key, npages=1, nrows=1, ncols=1, page_size=(5.5, 3.5))
        ax = f.add_axes((0, 0, 0))

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

        self.flush_figures()