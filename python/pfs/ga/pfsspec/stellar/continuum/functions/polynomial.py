import numpy as np

from ..modelfunction import ModelFunction

class PolynomialFunction(ModelFunction):
    def __init__(self, poly, deg, domain=None):
        super().__init__()

        self.poly = poly
        self.deg = deg
        self.domain = domain

    def get_min_point_count(self):
        return self.deg + 2

    def get_param_count(self):
        return self.deg + 1

    def fit(self, x, y, w=None, p0=None):
        ll = self.poly.fit(x, y, self.deg, w=w, domain=self.domain, full=False).convert()
        coef = np.zeros(self.deg + 1)
        coef[:ll.coef.shape[0]] = ll.coef
        return coef

    def shift(self, c, params):
        # Shift baseline
        params[0] += c

    def eval(self, x, params):
        ll = self.poly(params, domain=self.domain)
        return ll(x)

    def find_p0(self, x, y, w=None):
        # Not necessary to initialize p0 so just report success and return None
        return True, self.fit(x, y, w=w)