import os
import logging
import numpy as numpy

from pfsspec.stellar.grid.kurucz.io import KuruczAtmReader
from pfsspec.stellar.grid.kurucz import KuruczAtm

class BoszAtmReader(KuruczAtmReader):
    def __init__(self, path=None):
        super(BoszAtmReader, self).__init__()
        self.path = path

