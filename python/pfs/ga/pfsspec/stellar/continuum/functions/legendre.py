import numpy as np

from .polynomial import PolynomialFunction

class Legendre(PolynomialFunction):
    def __init__(self, deg, domain=None):
        super().__init__(np.polynomial.Legendre, deg, domain=domain)
