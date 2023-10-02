import os
import logging
import multiprocessing
import time

from pfs.ga.pfsspec.stellar.grid.io import AtmGridReader
from .kuruczatmreader import KuruczAtmReader

class KuruczAtmGridReader(AtmGridReader):
    def __init__(self, grid, reader, max=None, parallel=True):
        super(KuruczAtmGridReader, self).__init__()
        self.reader = reader