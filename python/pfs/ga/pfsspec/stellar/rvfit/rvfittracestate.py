class RVFitTraceState():
    """
    Trace info on preprocessed spectra and templates
    """

    def __init__(self):
        self.spectra = None
        self.templates = None
        self.bases = None
        self.masks = None
        self.sigma2 = None
        self.weights = None

    def reset(self):
        self.spectra = {}
        self.templates = {}         # Collect preprocessed templates for tracing
        self.bases = {}         # Collect continuum basis functions for tracing
        self.masks = {}
        self.sigma2 = {}
        self.weights = {}

    def append(self, arm, spec, temp, sigma2, weight, mask, basis):
        def append_item(d, arm, item):
            if arm not in d:
                d[arm] = []

        append_item(self.spectra, arm, spec)
        append_item(self.templates, arm, temp)
        append_item(self.bases, arm, basis)
        append_item(self.masks, arm, mask)
        append_item(self.sigma2, arm, sigma2)
        append_item(self.weights, arm, weight)