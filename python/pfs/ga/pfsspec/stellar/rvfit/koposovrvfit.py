import rvspecfit

class KoposovRVFit():
    """
    Implements a wrapper around Koposov's RVSPECFIT code.
    """

    def __init__(self, orig=None):

        if not isinstance(orig, KoposovRVFit):
            pass
        else:
            pass

    def fit_rv(self, spectra, templates, rv_bounds=(-500, 500)):
        # override = dict(ccf_continuum_normalize=ccf_continuum_normalize)
        # config = utils.read_config(config_fname, override)