from pfs.ga.pfsspec.core.grid import ArrayGrid
from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid.io import ModelGridReader
from pfs.ga.pfsspec.stellar.grid.phoenix import Phoenix
from .phoenixspectrumreader import PhoenixSpectrumReader

class PhoenixGridReader(ModelGridReader):
    def __init__(self, grid=None, orig=None):
        super(PhoenixGridReader, self).__init__(grid=grid, orig=orig)

        if isinstance(orig, PhoenixGridReader):
            pass
        else:
            pass

    def create_grid(self):
        return ModelGrid(Phoenix(), ArrayGrid)

    def create_reader(self, input_path, output_path, wave=None, resolution=None):
        return PhoenixSpectrumReader(input_path, wave, resolution)

    def get_example_filename(self):
        # Here we use constants because this particular model must exist in every grid.
        return self.reader.get_filename(M_H=0.0, T_eff=5000.0, log_g=1.0, a_M=0.0)