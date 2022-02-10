import os
import glob
import logging
import subprocess
import multiprocessing
import time
from tqdm import tqdm

from pfsspec.core.util import SmartParallel
from pfsspec.core.io import Downloader
from pfsspec.core.grid import GridEnumerator
from pfsspec.stellar.grid.io.modelgridreader import ModelGridReader

class ModelGridDownloader(Downloader):

    def __init__(self, grid=None, orig=None):
        super(ModelGridDownloader, self).__init__(orig=orig)
        
        if not isinstance(orig, ModelGridDownloader):
            self.args = None
            self.parallel = True
            self.threads = multiprocessing.cpu_count() // 2

            self.top = None

            self.grid = grid
            self.reader = None
        else:
            self.args = orig.args
            self.parallel = orig.parallel
            self.threads = orig.threads

            self.top = orig.top

            self.grid = grid if grid is not None else orig.grid

    def add_subparsers(self, configurations, parser):
        return None
            
    def add_args(self, parser):
        super(ModelGridDownloader, self).add_args(parser)

        parser.add_argument('--top', type=int, default=None, help='Limit number of results')

        # Add spectrum reader parameters
        reader = self.create_reader(None, None)
        reader.add_args(parser)

        # Add grid parameters to allow defining ranges
        grid = self.create_grid()
        grid.add_args(parser)

    def init_from_args(self, config, args):
        super(ModelGridDownloader, self).init_from_args(config, args)
        
        self.top = self.get_arg('top', self.top, args)

        self.grid = self.create_grid()
        self.grid.init_from_args(args)

        self.reader = self.create_reader(None, None)
        self.reader.init_from_args(args)

    def process_item(self, i):
        # Called when processing the grid point by point

        logger = multiprocessing.get_logger()

        index, params = i
        fn = self.reader.get_filename(**params)
        url = self.reader.get_url(**params)

        # Download file with wget
        outfile = os.path.join(self.outdir, fn)
        outdir, _ = os.path.split(outfile)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        cmd = [ 'wget', url ]
        cmd.extend(['--tries=3', '--timeout=1', '--wait=1'])
        cmd.extend(['--no-verbose'])
        cmd.extend(['-O', outfile])
        if self.resume:
            cmd.append('--continue')
        subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if os.path.getsize(outfile) == 0:
            os.remove(outfile)

    def create_grid(self):
        raise NotImplementedError()

    def create_reader(self, input_path, output_path, wave=None, resolution=None):
        raise NotImplementedError()

    def download_grid(self, resume=False):
        # Iterate over the grid points and call a function for each
        self.logger.info("Downloading grid {}.".format(type(self.grid).__name__))
        if self.top is not None:
            self.logger.info("Downloading grid will stop after {} items.".format(self.top))

        g = GridEnumerator(self.grid, top=self.top)
        t = tqdm(total=len(g))
        with SmartParallel(verbose=False, parallel=self.parallel, threads=self.threads) as p:
            for res in p.map(self.process_item, g):
                # Nothing to do here, file is saved in process_item
                t.update(1)

        self.logger.info("Grid downloaded.")

    def run(self):
        self.download_grid(resume=self.resume)