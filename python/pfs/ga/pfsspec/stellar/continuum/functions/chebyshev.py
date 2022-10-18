import numpy as np

from .polynomial import PolynomialFunction

class ChebyshevFunction(PolynomialFunction):
    def __init__(self, deg, domain=None):
        super().__init__(np.polynomial.Chebyshev, deg, domain=domain)

        self.name = "chebyshev"
