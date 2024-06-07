import os
import glob
import time

from pfs.ga.pfsspec.core.setup_logger import logger
from pfs.ga.pfsspec.core.grid.io import GridReader

class ModelGridReader(GridReader):

    CONFIG_PIPELINE = "pipeline"

    def __init__(self, grid=None, orig=None):
        super(ModelGridReader, self).__init__(grid=grid, orig=orig)

        if not isinstance(orig, ModelGridReader):
            self.reader = None
            self.pipeline = None
            self.path = None
            self.preload_arrays = False
            self.files = None
        else:
            self.reader = orig.reader
            self.pipeline = orig.pipeline
            self.path = orig.path
            self.preload_arrays = orig.preload_arrays
            self.files = None

    def add_subparsers(self, configurations, parser):
        # Register pipeline subparsers
        if 'pipelines' in configurations:
            pipeline_parsers = []
            pps = parser.add_subparsers(dest=self.CONFIG_PIPELINE)
            for p in configurations['pipelines']:
                pp = pps.add_parser(p)

                # Instantiate pipeline and add pipeline args
                pipeline = configurations['pipelines'][p]()
                pipeline.add_args(pp)

                pipeline_parsers.append(pp)
            return pipeline_parsers
        else:
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

        self.pipeline = config['pipelines'][args[self.CONFIG_PIPELINE]]()
        self.pipeline.init_from_args(config, args)

        if self.reader is None:
            self.reader = self.create_reader(None, None)
        self.reader.init_from_args(args)

        if self.grid is None:
            self.grid = self.create_grid()
            self.grid.preload_arrays = self.preload_arrays
        self.grid.init_from_args(config, args)

    def process_item(self, i):
        # Called when processing the grid point by point
        index, params = i
        fn = self.reader.get_filename(R=self.reader.resolution, **params)
        fn = os.path.join(self.reader.path, fn)

        if os.path.isfile(fn):
            tries = 3
            while True:
                try:
                    spec = self.reader.read(fn)
                    spec.set_params(params)
                    if self.pipeline is not None:
                        self.pipeline.run(spec, **params)
                    params = spec.get_params()
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
        
    def process_item_error(self, ex, i):
        raise NotImplementedError()

    def process_file(self, file):
        # Called when processing the grid file by file
        params = self.reader.parse_filename(file)
        index = self.grid.get_index(**params)
        spec = self.reader.read(file)

        return index, params, spec

    def store_item(self, res):
        if res is not None:
            index, params, spec = res
            self.grid.set_flux_at(index, spec.flux, spec.cont)

    def create_grid(self):
        raise NotImplementedError()

    def get_array_grid(self):
        return self.grid.array_grid

    def create_reader(self, input_path, output_path):
        raise NotImplementedError()

    def open_data(self, args, input_path, output_path):
        self.open_input_data(args, input_path)

        if not self.resume:
            axes, wave, wave_edges, is_wave_regular, is_wave_lin, is_wave_log = self.determine_grid_axes(input_path)
            
            # Update grid axes with new values
            if axes is not None:
                for i, p, ax in self.grid.enumerate_axes():
                    if p in axes:
                        ax.values = axes[p]

            
            # Initialize the wavelength grid based on the first spectrum read
            self.grid.set_wave(wave, wave_edges=wave_edges)

            # TODO: Here we assume that all spectra of the grid have the
            #       same binning
            self.grid.is_wave_regular = is_wave_regular
            self.grid.is_wave_lin = is_wave_lin
            self.grid.is_wave_log = is_wave_log

            self.grid.build_axis_indexes()
        
        self.open_output_data(args, output_path)    

    def open_input_data(self, args, input_path):
        self.reader.path = input_path
        
        if os.path.isdir(input_path):
            logger.info(f'{type(self).__name__} running in grid mode')
            self.path = input_path
        else:
            logger.info(f'{type(self).__name__} running in file list mode')
            self.files = glob.glob(os.path.expandvars(input_path))
            self.files.sort()
            logger.info('Found {} files.'.format(len(self.files)))

    def determine_grid_axes(self, input_path):
        # The default implementation takes the grid axes from the grid config
        # and tries to load a single file in order to determine the wavelengths.
        # This method should be overridden if the grid axes cannot be determined
        # this way.
    
        if os.path.isdir(input_path):
            # Load the first spectrum to get wavelength grid.
            fn = self.get_example_filename()
            fn = os.path.join(self.path, fn)
            spec = self.reader.read(fn)
        else:
            # Load the first spectrum to get wavelength grid
            spec = self.reader.read(self.files[0])

        # Run through the import pipeline to make sure the wavelength grid
        # matches the output
        if self.pipeline is not None:
            self.pipeline.run(spec)

        logger.info('Found spectrum with {} wavelength elements.'.format(spec.wave.shape))

        return None, spec.wave, spec.wave_edges, spec.is_wave_regular, spec.is_wave_lin, spec.is_wave_log

    def open_output_data(self, args, output_path):
        fn = os.path.join(output_path, 'spectra.h5')

        if self.resume:
            if self.grid.preload_arrays:
                raise NotImplementedError("Can only resume import when preload_arrays is False.")
            self.grid.load(fn, format='h5')
        else:          
            # Force creating output file for direct hdf5 writing
            self.grid.save(fn, format='h5')

    def save_data(self, args, output_path):
        self.grid.save(self.grid.filename, self.grid.fileformat)

    def run(self):
        if os.path.isdir(self.path):
            self.read_grid(resume=self.resume)
        else:
            self.read_files(self.files, resume=self.resume)