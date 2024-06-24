from ..continuumobject import ContinuumObject

class ContinuumFinder(ContinuumObject):
    """
    When implemented in derived classes, finds the
    control points to be used for continuum fitting.
    """

    def __init__(self, orig=None):
        super().__init__(orig=orig)

        if not isinstance(orig, ContinuumFinder):
            pass
        else:
            pass
    
    def find(self, iter, wave, flux, /, w=None, mask=None, cont=None):
        """
        When implemented in derived classes, return a boolean mask that selects
        the spectral pixels to be used for continuum fitting.
        """
        raise NotImplementedError()