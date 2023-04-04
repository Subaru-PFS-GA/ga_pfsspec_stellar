import numpy as np
from tqdm import tqdm

from pfs.ga.pfsspec.core.util.args import *
from pfs.ga.pfsspec.core.util.smartparallel import SmartParallel
from pfs.ga.pfsspec.core.grid import GridBuilder
from pfs.ga.pfsspec.core.grid import ArrayGrid
from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid import ModelGridBuilder
from pfs.ga.pfsspec.stellar.grid import ModelGridConfig

class ModelGridConverter(GridBuilder, ModelGridBuilder):
    """
    Run a spectrum processing pipeline on a grid of synthetic spectra.
    """

    CONFIG_PIPELINE = "pipeline"

    def __init__(self, grid_config, orig=None):
        GridBuilder.__init__(self, orig=orig)
        ModelGridBuilder.__init__(self, grid_config, orig=orig)

        if isinstance(orig, ModelGridConverter):
            self.resolution = orig.resolution
            self.pipeline = orig.pipeline
            self.grid_config = grid_config if grid_config is not None else orig.grid_config
        else:
            self.resolution = None
            self.pipeline = None
            self.grid_config = grid_config

    def add_subparsers(self, configurations, parser):
        # Register pipeline subparsers
        if 'pipelines' in configurations:
            pipeline_parsers = []
            pps = parser.add_subparsers(dest=self.CONFIG_PIPELINE)
            for p in configurations['pipelines']:
                pp = pps.add_parser(p)

                # Instantiate pipeline and add pipeline args
                pipeline = self.create_pipeline(configurations['pipelines'][p])
                pipeline.add_args(pp)

                pipeline_parsers.append(pp)
            return pipeline_parsers
        else:
            return None

    def add_args(self, parser):
        GridBuilder.add_args(self, parser)
        ModelGridBuilder.add_args(self, parser)

        parser.add_argument('--resolution', type=float, help='Input model resolution.')

    def init_from_args(self, config, args):
        GridBuilder.init_from_args(self, config, args)
        ModelGridBuilder.init_from_args(self, config, args)

        self.pipeline = self.create_pipeline(config['pipelines'][args[self.CONFIG_PIPELINE]], args)
        self.resolution = get_arg('resolution', self.resolution, args)

    def create_pipeline(self, config, args=None):
        pipeline = config()
        if args is not None:
            pipeline.init_from_args(config, args)
        return pipeline

    def create_input_grid(self):
        grid =  ModelGridBuilder.create_input_grid(self)
        if self.resolution is not None:
            grid.resolution = self.resolution
        return grid

    def open_input_grid(self, input_path):
        return ModelGridBuilder.open_input_grid(self, input_path)

    def create_output_grid(self):
        return ModelGridBuilder.create_output_grid(self)

    def open_output_grid(self, output_path):
        return ModelGridBuilder.open_output_grid(self, output_path)

    def open_data(self, args, input_path, output_path, params_path=None):
        return ModelGridBuilder.open_data(self, args, input_path, output_path, params_path=params_path)

    def build_data_index(self):
        return ModelGridBuilder.build_data_index(self)

    def store_item(self, idx, spec):
        self.output_grid.grid.set_value_at('flux', idx, spec.flux)
        if spec.cont is not None:
            self.output_grid.grid.set_value_at('cont', idx, spec.cont)

    def process_item(self, i):
        input_idx, output_idx, spec = self.get_gridpoint_model(i)

        self.pipeline.run(spec)
        return i, input_idx, output_idx, spec

    def run(self):
        output_initialized = False
        input_count = self.get_input_count()

        # TODO: init pipeline?
        # self.pipeline.init()

        # Run every spectrum through the pipeline
        t = tqdm(total=input_count)
        with SmartParallel(initializer=self.init_process, verbose=False, parallel=self.parallel, threads=self.threads) as p:
            for i, input_idx, output_idx, spec in p.map(self.process_item, range(input_count)):
                if not output_initialized:
                    self.output_grid.set_wave(spec.wave)
                    self.output_grid.allocate_values()
                    self.output_grid.build_axis_indexes()
                    output_initialized = True
                self.store_item(output_idx, spec)
                t.update(1)

    