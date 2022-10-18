class ContinuumFinder():
    """
    When implemented in derived classes, find the
    control points to be used for continuum fitting.
    """

    def __init__(self, max_iter=None, orig=None):
        
        if not isinstance(orig, ContinuumFinder):
            self.max_iter = max_iter if max_iter is not None else 5
        else:
            self.max_iter = max_iter if max_iter is not None else orig.max_iter

    def get_max_iter(self):
        return self.max_iter
    
    def find(self, iter, wave, flux, flux_err=None, weight=None, mask=None, model=None):
        raise NotImplementedError()