class RVFitTraceState():
    """
    Trace info on preprocessed spectra and templates
    """

    def __init__(self):
        self.spectra = None
        self.templates = None

    def reset(self):
        self.spectra = {}
        self.templates = {}         # Collect preprocessed templates for tracing

    def append(self, arm, spec, temp):
        def append_item(d, arm, item):
            if arm not in d:
                d[arm] = []

        append_item(self.spectra, arm, spec)
        append_item(self.templates, arm, temp)