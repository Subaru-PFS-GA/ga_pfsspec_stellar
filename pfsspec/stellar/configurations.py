from pfsspec.core import Pipeline
from pfsspec.core.grid import ArrayGrid, RbfGrid
from pfsspec.stellar.grid.bosz import Bosz
from pfsspec.stellar.grid.phoenix import Phoenix
from pfsspec.stellar.grid.bosz.io import BoszGridReader
from pfsspec.stellar.grid.bosz.io import BoszGridDownloader
from pfsspec.stellar.grid.phoenix.io import PhoenixGridReader
from pfsspec.stellar.grid.phoenix.io import PhoenixGridDownloader
from pfsspec.stellar.grid import ModelGridFit, ModelPcaGridBuilder, ModelRbfGridBuilder

DOWNLOAD_CONFIGURATIONS = {
    'stellar-grid': {
        'bosz': {
            'type': BoszGridDownloader
        },
        'phoenix': {
            'type': PhoenixGridDownloader
        }
    }
}

IMPORT_CONFIGURATIONS = {
    'stellar-grid': {
        'bosz': {
            'type': BoszGridReader,
            'pipelines': {
                'basic': Pipeline
            },
        },
        'phoenix': {
            'type': PhoenixGridReader,
            'pipelines': {
                'basic': Pipeline
            }
        }
    }
}

FIT_CONFIGURATIONS = {
    'stellar-grid': {
        'bosz': {
            'type': ModelGridFit,
            'config': Bosz()
        },
        'phoenix': {
            'type': ModelGridFit,
            'config': Phoenix()
        }
    }
}

RBF_CONFIGURATIONS = {
    'stellar-grid': {
        'bosz': {
            'type': ModelRbfGridBuilder,
            'config': Bosz()
        },
        'phoenix': {
            'type': ModelRbfGridBuilder,
            'config': Phoenix()
        }
    }
}

PCA_CONFIGURATIONS = {
    'stellar-grid': {
        'bosz': {
            'type': ModelPcaGridBuilder,
            'config': Bosz()
        },
        'phoenix': {
            'type': ModelPcaGridBuilder,
            'config': Phoenix()
        }
    }
}