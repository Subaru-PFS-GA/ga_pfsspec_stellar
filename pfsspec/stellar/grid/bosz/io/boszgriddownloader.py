import os
import glob
import logging

from pfsspec.core.grid import ArrayGrid
from pfsspec.stellar.grid import ModelGrid
from pfsspec.stellar.grid.io import ModelGridDownloader
from pfsspec.stellar.grid.bosz import Bosz
from .boszspectrumreader import BoszSpectrumReader

class BoszGridDownloader(ModelGridDownloader):
    def __init__(self, grid=None, orig=None):
        super(BoszGridDownloader, self).__init__(grid=grid, orig=orig)

        if isinstance(orig, BoszGridDownloader):
            pass
        else:
            pass

    def create_grid(self):
        return ModelGrid(Bosz(), ArrayGrid)

    def create_reader(self, input_path, output_path):
        reader = BoszSpectrumReader(input_path)
        if self.args is not None:
            reader.init_from_args(self.args)
        return reader