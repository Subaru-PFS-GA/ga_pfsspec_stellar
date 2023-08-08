import os

from pfs.ga.pfsspec.core import Trace

class RVFitTrace(Trace):
    """
    Implements call-back function to profile and debug RV fitting. Allows for
    generating plots of intermediate steps.
    """

    def __init__(self, outdir='.', plot_inline=False, plot_level=Trace.PLOT_LEVEL_NONE, log_level=Trace.LOG_LEVEL_NONE):
        super().__init__(outdir=os.path.join(outdir, 'rvfit'), plot_inline=plot_inline, plot_level=plot_level, log_level=log_level)
        
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

    def save_history(self, filename, spectrum):
        if spectrum.history is not None:
            fn = os.path.join(self.outdir, filename)
            self.make_outdir(fn)
            with open(fn, "w") as f:
                f.writelines([ s + '\n' for s in spectrum.history ])

    def plot_spectrum(self, key, arm, wave, flux=None, error=None, cont=None, model=None, label=None):
        def plot(ax, mask=()):
            if flux is not None:
                ax.plot(wave[mask], flux[mask], label=label, **styles.solid_line())
            if error is not None:
                ax.plot(wave[mask], error[mask], **styles.solid_line())
            if cont is not None:
                ax.plot(wave[mask], cont[mask], **styles.solid_line())
            if model is not None:
                ax.plot(wave[mask], model[mask], **styles.solid_line())

        (f, axs) = self.get_figure(key, 2, 1)
        plot(axs[0])
        plot(axs[1], (np.abs(wave - wave.mean()) < 20.0))

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
        if self.plot_level >= Trace.PLOT_LEVEL_INFO:
            (f, ax) = self.get_figure('guess_rv', 1, 1)
            ax.plot(rv, log_L, '.')
            if fit is not None:
                ax.plot(rv, fit, **styles.solid_line())
            ax.axvline(rv_guess, c='r')

            self.flush_figures()

    def on_prepare_fit(self, rv_0, rv_bounds, rv_step):
        pass
    
        # if self.rv_iter is None:
        #     self.rv_iter = []
        # self.rv_iter.append(rv_0)

    def on_fit_rv_iter(self, rv):
        if self.rv_iter is None:
            self.rv_iter = []
        self.rv_iter.append(rv)

    def on_fit_rv(self, spectra, templates, rv):
        if self.rv_iter is None:
            self.rv_iter = []
        self.rv_iter.append(rv)