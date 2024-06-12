class ModelFunction():
    def __init__(self):
        pass

    def get_min_point_count(self):
        """
        Return the minimum number of points necessary to fit the function
        """

        raise NotImplementedError

    def get_param_count(self):
        """
        Return the number of function parameters.
        """

        raise NotImplementedError()

    def fit(self, x, y, w=None, p0=None, **kwargs):
        """
        Fit the function.
        """

        raise NotImplementedError()

    def eval(self, x, params):
        """
        Evaluate the function.
        """

        raise NotImplementedError()

    def find_p0(self, x, y, w=None, mask=None):
        """
        Find some good initial values for fitting the function.
        """

        raise NotImplementedError()