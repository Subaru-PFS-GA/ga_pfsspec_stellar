import numpy as np

from pfs.ga.pfsspec.core.util.copy import safe_deep_copy
from pfs.ga.pfsspec.core import Physics
from .piecewise import Piecewise
from ..functions import ChebyshevFunction as ChebyshevFunction

class PiecewiseChebyshev(Piecewise):
    def __init__(self, deg=None, continuum_finder=None, trace=None, orig=None):
        super().__init__(continuum_finder=continuum_finder,
                         trace=trace, orig=orig)

        if not isinstance(orig, PiecewiseChebyshev):
            self.chebyshev_degrees = deg if deg is not None else 6
        else:
            self.chebyshev_degrees = deg if deg is not None else orig.chebyshev_degrees

    @property
    def name(self):
        return "chebyshev"

    def create_function(self, i):
        return ChebyshevFunction(self.chebyshev_degrees)

    def add_args(self, parser):
        super().add_args(parser)

    def init_from_args(self, args):
        super().init_from_args(args)

    def get_constants(self, wave=None):
        """
        Return the constants necessary to evaluate the continuum model
        """
        constants = super().get_constants()
        constants['chebyshev_degrees'] = self.chebyshev_degrees
        return constants

    def set_constants(self, constants, wave=None):
        """
        Load the constants necessary to evaluate the continuum model
        """

        if self.version == 1:
            self.chebyshev_degrees = int(constants['chebyshev_degrees'])
        else:
            raise NotImplementedError()    
