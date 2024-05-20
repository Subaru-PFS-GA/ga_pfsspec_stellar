import os
import numpy as numpy

from pfs.ga.pfsspec.stellar.grid.kurucz.io import KuruczAtmReader
from pfs.ga.pfsspec.stellar.grid.kurucz import KuruczAtm

class PhoenixAtmReader(KuruczAtmReader):
    def __init__(self, path=None):
        super(PhoenixAtmReader, self).__init__()
        self.path = path

