import os
import glob
import logging
import subprocess
import multiprocessing
import time
from tqdm import tqdm

from pfs.ga.pfsspec.core.util import SmartParallel
from pfs.ga.pfsspec.core.io import Downloader
from pfs.ga.pfsspec.core.grid import GridEnumerator
from pfs.ga.pfsspec.stellar.grid.io.modelgridreader import ModelGridReader

class ModelGridDownloader(Downloader):

    def __init__(self, grid=None, orig=None):
        super(ModelGridDownloader, self).__init__(orig=orig)
        
        if not isinstance(orig, ModelGridDownloader):
            self.grid = grid
            self.reader = None
        else:
            self.grid = grid if grid is not None else orig.grid
            self.reader = orig.reader

    def add_subparsers(self, configurations, parser):
        return None
            
    def add_args(self, parser, config):
        super().add_args(parser, config)

        # Add spectrum reader parameters
        reader = self.create_reader(None, None)
        reader.add_args(parser)

        # Add grid parameters to allow defining ranges
        grid = self.create_grid()
        grid.add_args(parser)

    def init_from_args(self, script, config, args):
        super().init_from_args(script, config, args)
        
        self.grid = self.create_grid()
        self.grid.init_from_args(args)

        self.reader = self.create_reader(None, None)
        self.reader.init_from_args(args)

    def process_item_error(self, ex, i):
        raise NotImplementedError()

    def process_item(self, i):
        # Called when processing the grid point by point
        index, params = i
        fn = self.reader.get_filename(**params)
        url = self.reader.get_url(**params)

        # Download file with wget
        outfile = os.path.join(self.outdir, fn)
        self.wget_download(url, outfile, resume=self.resume, create_dir=True)

    def create_grid(self):
        raise NotImplementedError()

    def create_reader(self, input_path, output_path, wave=None, resolution=None):
        raise NotImplementedError()

    def download_grid(self, resume=False):
        # Iterate over the grid points and call a function for each
        self.logger.info("Downloading grid {}.".format(type(self.grid).__name__))
        if self.top is not None:
            self.logger.info("Downloading grid will stop after {} items.".format(self.top))

        g = GridEnumerator(self.grid, s=self.grid.get_slice(), top=self.top)
        t = tqdm(total=len(g))
        with SmartParallel(verbose=False, parallel=self.parallel, threads=self.threads) as p:
            for res in p.map(self.process_item, self.process_item_error, g):
                # Nothing to do here, file is saved in process_item
                t.update(1)

        self.logger.info("Grid downloaded.")

    def run(self):
        self.download_grid(resume=self.resume)