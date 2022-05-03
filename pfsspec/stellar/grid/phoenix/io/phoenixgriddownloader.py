import os
import glob
import logging

from pfsspec.core.grid import ArrayGrid
from pfsspec.stellar.grid import ModelGrid
from pfsspec.stellar.grid.io import ModelGridDownloader
from pfsspec.stellar.grid.phoenix import Phoenix
from .phoenixspectrumreader import PhoenixSpectrumReader

class PhoenixGridDownloader(ModelGridDownloader):
    def __init__(self, grid=None, orig=None):
        super(PhoenixGridDownloader, self).__init__(grid=grid, orig=orig)

        if isinstance(orig, PhoenixGridDownloader):
            pass
        else:
            pass

    def create_grid(self):
        return ModelGrid(Phoenix(), ArrayGrid)

    def create_reader(self, input_path, output_path):
        return PhoenixSpectrumReader(input_path)