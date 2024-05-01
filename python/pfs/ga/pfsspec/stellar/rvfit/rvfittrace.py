import os
import numpy as np

from pfs.ga.pfsspec.core.plotting import SpectrumPlot, styles
from pfs.ga.pfsspec.core import Trace
from pfs.ga.pfsspec.core.util.args import *

class RVFitTrace(Trace):
    """
    Implements call-back function to profile and debug RV fitting. Allows for
    generating plots of intermediate steps.
    """

    #region Initializers

    def __init__(self, outdir='.', 
                 plot_inline=False, 
                 plot_level=Trace.PLOT_LEVEL_NONE, 
                 log_level=Trace.LOG_LEVEL_NONE):
        
        super().__init__(outdir=os.path.join(outdir, 'rvfit'), 
                         plot_inline=plot_inline, 
                         plot_level=plot_level,
                         log_level=log_level)
        
        self.plot_rv_prior = False
        self.plot_rv_guess = False
        self.plot_input_spec = False
        self.plot_fit_spec = False
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

    def add_args(self, config, parser):
        super().add_args(config, parser)
    
    def init_from_args(self, script, config, args):
        super().init_from_args(script, config, args)

        self.plot_rv_prior = get_arg('plot_rv_prior', self.plot_rv_prior, args)
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
        
        # TODO: plot rv_0, prior and bounds here
    
    def on_fit_rv_iter(self, rv):
        self.rv_iter.append(rv)
    
    def on_fit_rv_finish(self, spectra, templates, processed_templates, 
                         rv_0, rv_fit, rv_err, rv_bounds, rv_prior, rv_step,
                         log_L_fun):
        
        self.rv_iter.append(rv_fit)
        
        # TODO: plot whatever requested
        if self.plot_fit_spec:
            self._plot_spectra('rvfit_best', spectra, templates=templates, processed_templates=processed_templates,
                               plot_spectra=True, plot_processed_templates=True,
                               plot_flux_err=True, plot_residuals=False)
            self._plot_spectra('rvfit_residuals', spectra, templates=templates, processed_templates=processed_templates,
                               plot_spectra=False, plot_processed_templates=False,
                               plot_flux_err=False, plot_residuals=True)
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

    def on_guess_rv(self, rv, log_L, rv_guess, fit, function, pp, pcov):
        if self.plot_rv_guess or self.plot_level >= Trace.PLOT_LEVEL_INFO:
            (f, ax) = self.get_page('guess_rv', 1, 1)
            ax.plot(rv, log_L, '.')
            if fit is not None:
                ax.plot(rv, fit, **styles.solid_line())
            ax.axvline(rv_guess, c='r')

            self.flush_figures()

    #endregion

    def save_history(self, filename, spectrum):
        """
        Save the processing history of a spectrum
        """
        if spectrum.history is not None:
            fn = os.path.join(self.outdir, filename)
            self.make_outdir(fn)
            with open(fn, "w") as f:
                f.writelines([ s + '\n' for s in spectrum.history ])
                
    def _plot_spectra(self, key, spectra, templates=None, processed_templates=None,
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
                    p.plot_spectrum(specs[i], plot_flux_err=plot_flux_err)

                if plot_templates and templates is not None:
                    raise NotImplementedError()

                if plot_processed_templates and processed_templates is not None:
                    temp = processed_templates[arm][i]
                    p.plot_template(temp)

                if plot_residuals and processed_templates is not None:
                    temp = processed_templates[arm][i]
                    p.plot_residual(spec, temp)

            # TODO: Add SNR, exp time, obs date
            p.title = spec.get_id_string()
            p.apply()

        f.match_limits()

        pass

    