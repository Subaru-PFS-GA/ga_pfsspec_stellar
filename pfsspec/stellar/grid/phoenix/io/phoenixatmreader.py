import os
import logging
import numpy as numpy

from pfsspec.stellar.grid.kurucz.io import KuruczAtmReader
from pfsspec.stellar.grid.kurucz import KuruczAtm

class PhoenixAtmReader(KuruczAtmReader):
    def __init__(self, path=None):
        super(PhoenixAtmReader, self).__init__()
        self.path = path

