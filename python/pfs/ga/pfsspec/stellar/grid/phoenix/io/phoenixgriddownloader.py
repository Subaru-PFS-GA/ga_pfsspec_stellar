import os
import glob
import logging

from pfs.ga.pfsspec.core.grid import ArrayGrid
from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid.io import ModelGridDownloader
from pfs.ga.pfsspec.stellar.grid.phoenix import Phoenix
from .phoenixspectrumreader import PhoenixSpectrumReader

class PhoenixGridDownloader(ModelGridDownloader):
    def __init__(self, grid=None, orig=None):
        super(PhoenixGridDownloader, self).__init__(grid=grid, orig=orig)

        if not isinstance(orig, PhoenixGridDownloader):
            self.version = "1.0"
        else:
            self.version = orig.version

    def add_args(self, parser, config):
        super().add_args(parser, config)

        parser.add_argument('--phoenix-version', type=str, default="1.0", help='PHOENIX grid version')

    def init_from_args(self, config, args):
        super().init_from_args(config, args)

        self.version = self.get_arg('phoenix_version', self.version, args)

    def create_grid(self):
        return ModelGrid(Phoenix(), ArrayGrid)

    def create_reader(self, input_path, output_path):
        return PhoenixSpectrumReader(input_path, version=self.version)