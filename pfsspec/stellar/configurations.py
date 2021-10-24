from pfsspec.core.grid import ArrayGrid, RbfGrid
from pfsspec.stellar.grid.bosz import Bosz
from pfsspec.stellar.grid.bosz.io import BoszGridReader
#from pfsspec.stellar.grid.phoenix import Phoenix
#from pfsspec.stellar.grid.phoenix.io import PhoenixGridReader

IMPORT_CONFIGURATIONS = {
    'stellar-grid': {
        'bosz': BoszGridReader,
        #'phoenix': PhoenixGridReader
    }
}