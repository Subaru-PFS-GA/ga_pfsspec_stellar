class ContinuumFinder():
    """
    When implemented in derived classes, finds the
    control points to be used for continuum fitting.
    """

    def __init__(self, orig=None):
        if not isinstance(orig, ContinuumFinder):
            pass
        else:
            pass
    
    def find(self, iter, wave, flux, flux_err=None, weight=None, mask=None, model=None):
        """
        When implemented in derived classes, return a boolean mask that selects
        the spectral pixels to be used for continuum fitting.
        """
        raise NotImplementedError()