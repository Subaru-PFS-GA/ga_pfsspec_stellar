import os
import re
import numpy as np

from pfs.ga.pfsspec.core.grid import ArrayGrid
from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid.io import ModelGridReader
from ..grid7 import Grid7
from .grid7spectrumreader import Grid7SpectrumReader

class Grid7GridReader(ModelGridReader):
    def __init__(self, grid=None, orig=None):
        super(Grid7GridReader, self).__init__(grid=grid, orig=orig)

        if isinstance(orig, Grid7GridReader):
            pass
        else:
            pass

    def create_grid(self, grid_path=None):
        return ModelGrid(Grid7(), ArrayGrid)

    def create_reader(self, input_path, output_path):
        return Grid7SpectrumReader(input_path)

    def get_example_filename(self):
        # Here we use constants because this particular model must exist in every grid.
        return self.reader.get_filename(M_H=0.0, T_eff=5000.0, log_g=1.0, a_M=0.0)
    
    def get_grid_axes(self, grid_path):
        # Figure out grid steps and bounds

        grid = self.create_grid()
        axes = { p: [] for i, p, _ in grid.enumerate_axes()}

        # Recursively collect and parse the filenames of all *.bin.gz models
        for root, subdirs, files in os.walk(grid_path):
            for file in files:
                if file.lower().endswith('.bin.gz'):
                    params = Grid7SpectrumReader.parse_filename(file)
                        
                    for p in params:
                        axes[p] += [ params[p] ]

        axes = { axis: np.unique(axes[axis]) for axis in axes }

        return axes