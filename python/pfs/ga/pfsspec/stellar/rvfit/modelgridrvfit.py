from .rvfit import RVFit, RVFitTrace

class ModelGridRVFit(RVFit):

    # TODO: this doesn't do much right now, extend to incorporate
    #       fitting model parameters as well

    def __init__(self, trace=None, orig=None):
        super().__init__(trace=trace, orig=orig)

        if not isinstance(orig, ModelGridRVFit):
            self.grid = None
            self.psf = None
        else:
            self.grid = orig.grid
            self.psf = None

    def get_template(self, convolve=True, wlim=None, **kwargs):
        """
        Generate a noiseless template spectrum with same line spread function as the
        observation but keep the original, high-resolution binning.
        """

        # TODO: add template caching

        temp = self.grid.get_nearest_model(**kwargs)
        temp.cont = None        # Make sure it's not passed around for better performance
        temp.mask = None

        if convolve:
            temp.convolve_psf(self.psf, wlim=wlim)

        return temp