class FluxCorrection():
    def __init__(self, orig=None):
        if not isinstance(orig, FluxCorrection):
            pass
        else:
            pass

    def get_coeff_count(self):
        raise NotImplementedError()

    def get_basis_callable(self):
        """
        Return a callable that evaluates the basis functions over a
        wavelength grid.
        """
        raise NotImplementedError()
