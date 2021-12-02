import os
import glob
import logging
import multiprocessing
import time

from pfsspec.core.grid.io import GridReader

class ModelGridReader(GridReader):

    CONFIG_PIPELINE = "pipeline"

    def __init__(self, grid=None, orig=None):
        super(ModelGridReader, self).__init__(grid=grid, orig=orig)

        if not isinstance(orig, ModelGridReader):
            self.wave = None
            self.resolution = 5000          # Nominal resolution of input models

            self.reader = None
            self.path = None
            self.preload_arrays = False
            self.files = None

            self.is_wave_regular = None
            self.is_wave_lin = None
            self.is_wave_log = None
        else:
            self.wave = orig.wave
            self.resolution = orig.resolution

            self.reader = orig.reader
            self.path = orig.path
            self.preload_arrays = orig.preload_arrays
            self.files = None

            self.is_wave_regular = orig.is_wave_regular
            self.is_wave_lin = orig.is_wave_lin
            self.is_wave_log = orig.is_wave_log

    def add_subparsers(self, configurations, parser):
        # Register pipeline subparsers
        if 'pipelines' in configurations:
            pipeline_parsers = []
            pps = parser.add_subparsers(dest=self.CONFIG_PIPELINE)
            for p in configurations['pipelines']:
                pp = pps.add_parser(p)

                # Add grid reader args
                self.add_args(pp)

                # Instantiate pipeline and add pipeline args
                pipeline = configurations['pipelines'][p]()
                pipeline.add_args(pp)

                pipeline_parsers.append(pp)
            return pipeline_parsers
        else:
            return None

    def add_args(self, parser):
        super(ModelGridReader, self).add_args(parser)

    def init_from_args(self, args):
        super(ModelGridReader, self).init_from_args(args)

    def process_item(self, i):
        # Called when processing the grid point by point

        logger = multiprocessing.get_logger()

        index, params = i
        fn = self.reader.get_filename(R=self.reader.resolution, **params)
        fn = os.path.join(self.reader.path, fn)

        if os.path.isfile(fn):
            tries = 3
            while True:
                try:
                    spec = self.reader.read(fn)
                    return index, params, spec
                except Exception as e:
                    logger.error('Error parsing {}'.format(fn))
                    time.sleep(0.01)    # ugly hack
                    tries -= 1
                    if tries == 0:
                        raise e

        else:
            logger.debug('Cannot find file {}'.format(fn))
            return None

    def process_file(self, file):
        # Called when processing the grid file by file

        logger = multiprocessing.get_logger()

        params = self.reader.parse_filename(file)
        index = self.grid.get_index(**params)
        spec = self.reader.read(file)

        return index, params, spec

    def store_item(self, res):
        if res is not None:
            index, params, spec = res

            if self.grid.get_wave() is None:
                self.grid.set_wave(spec.wave)
                self.grid.is_wave_regular = self.is_wave_regular
                self.grid.is_wave_lin = self.is_wave_lin
                self.grid.is_wave_log = self.is_wave_log

            self.grid.set_flux_at(index, spec.flux, spec.cont)

    def create_grid(self):
        raise NotImplementedError()

    def get_array_grid(self):
        return self.grid.array_grid

    def create_reader(self, input_path, output_path, wave=None, resolution=None):
        raise NotImplementedError()

    def open_data(self, input_path, output_path):
        # Initialize input

        if self.reader is None:
            self.reader = self.create_reader(input_path, output_path)

        if os.path.isdir(input_path):
            self.logger.info('Running in grid mode')
            self.path = input_path

            # Load the first spectrum to get wavelength grid.
            fn = self.get_example_filename()
            fn = os.path.join(self.path, fn)
            spec = self.reader.read(fn)
        else:
            self.logger.info('Running in file list mode')
            self.files = glob.glob(os.path.expandvars(input_path))
            self.files.sort()
            self.logger.info('Found {} files.'.format(len(self.files)))

            # Load the first spectrum to get wavelength grid
            spec = self.reader.read(self.files[0])

        self.logger.info('Found spectrum with {} wavelength elements.'.format(spec.wave.shape))

        # Initialize output

        fn = os.path.join(output_path, 'spectra.h5')

        if self.grid is None:
            self.grid = self.create_grid()
            self.grid.preload_arrays = self.preload_arrays

        if self.resume:
            if self.grid.preload_arrays:
                raise NotImplementedError("Can only resume import when preload_arrays is False.")
            self.grid.load(fn, format='h5')
        else:
            # Initialize the wavelength grid based on the first spectrum read
            self.grid.set_wave(spec.wave)
            self.grid.build_axis_indexes()
           
            # Force creating output file for direct hdf5 writing
            self.grid.save(fn, format='h5')

    def save_data(self):
        self.grid.save(self.grid.filename, self.grid.fileformat)

    def run(self):
        if os.path.isdir(self.path):
            self.read_grid(resume=self.resume)
        else:
            self.read_files(self.files, resume=self.resume)