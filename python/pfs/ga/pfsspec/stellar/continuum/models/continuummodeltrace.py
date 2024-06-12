from pfs.ga.pfsspec.core import Trace
from pfs.ga.pfsspec.core import SpectrumTrace

class ContinuumModelTrace(Trace, SpectrumTrace):
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

        self.plot_fit_start = False
        self.plot_fit_iter = False
        self.plot_fit_end = False

        self.reset()

    def reset(self):
        self.fit_iter = 0

    def add_args(self, config, parser):
        Trace.add_args(self, config, parser)

    def init_from_args(self, script, config, args):
        Trace.init_from_args(self, script, config, args)

    def on_fit_start(self, spec):
        if self.plot_fit_start or self.plot_level >= Trace.PLOT_LEVEL_TRACE:
            self._plot_spectrum('pfsGA-continuumfit-{id}', 
                                spectrum=spec,
                                plot_flux=True, plot_continuum=False,
                                title='Continuum fit input spectrum - {id}')

    def on_fit_finish(self, spec):
        if self.plot_fit_end or self.plot_level >= Trace.PLOT_LEVEL_TRACE:
            self._plot_spectrum('pfsGA-continuumfit-{id}', 
                                spectrum=spec,
                                plot_flux=True, plot_continuum=True,
                                title='Continuum fit results - {id}')
            self.flush_figures()

    def on_fit_function_iter(self, piece_id, iter, x, y, w, model, mask):
        self.fit_iter += 1

        if self.plot_fit_iter or self.plot_level >= Trace.PLOT_LEVEL_TRACE:
            raise NotImplementedError()