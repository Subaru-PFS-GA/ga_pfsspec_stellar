from pfs.ga.pfsspec.core.grid import ArrayGrid
from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid.io import ModelGridReader
from ..gk2025 import GK2025
from .gk2025spectrumreader import GK2025SpectrumReader

class GK2025GridReader(ModelGridReader):
    def __init__(self, grid=None, orig=None):
        super(GK2025GridReader, self).__init__(grid=grid, orig=orig)

        if isinstance(orig, GK2025GridReader):
            pass
        else:
            pass

    def create_grid(self):
        return ModelGrid(GK2025(), ArrayGrid)

    def create_reader(self, input_path, output_path):
        return GK2025SpectrumReader(input_path)

    def get_example_filename(self):
        # Here we use constants because this particular model must exist in every grid.
        return self.reader.get_filename(M_H=0.0, T_eff=5000.0, log_g=1.0, a_M=0.0, C_M=0.0)