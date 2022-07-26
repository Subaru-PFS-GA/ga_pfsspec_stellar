import numpy as np

class Legendre():
    def __init__(self, deg, domain=None):
        self.deg = deg
        self.domain = domain

    def get_param_count(self):
        return self.deg + 1

    def fit(self, x, y, w=None, p0=None, max_deg=None):
        deg = min(self.deg, x.shape[0] - 2)
        if max_deg is not None:
            deg = min(deg, max_deg)

        ll = np.polynomial.Legendre.fit(x, y, deg, w=w, domain=self.domain, full=False)
        coef = np.zeros(self.deg + 1)
        coef[:ll.coef.shape[0]] = ll.coef
        return coef

    def shift(self, c, params):
        # Shift baseline
        params[0] += c

    def eval(self, x, params):
        ll = np.polynomial.Legendre(params, domain=self.domain)
        return ll(x)

    def find_p0(self, x, y, w=None):
        return True, None