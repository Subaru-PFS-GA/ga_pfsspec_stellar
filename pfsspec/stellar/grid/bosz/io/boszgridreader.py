import os
import glob
import logging

from pfsspec.core.grid import ArrayGrid
from pfsspec.stellar.grid import ModelGrid
from pfsspec.stellar.grid.io import ModelGridReader
from pfsspec.stellar.grid.bosz import Bosz
from .boszspectrumreader import BoszSpectrumReader

class BoszGridReader(ModelGridReader):
    def __init__(self, grid=None, orig=None):
        super(BoszGridReader, self).__init__(grid=grid, orig=orig)

        if isinstance(orig, BoszGridReader):
            pass
        else:
            pass

    def create_grid(self):
        return ModelGrid(Bosz(), ArrayGrid)

    def create_reader(self, input_path, output_path, wave=None, resolution=None):
        return BoszSpectrumReader(input_path, wave, resolution)

    def get_example_filename(self):
        # Here we use constants because this particular model must exist in every grid.
        return self.reader.get_filename(Fe_H=0.0, T_eff=5000.0, log_g=1.0, O_M=0.0, C_M=0.0, R=self.resolution)