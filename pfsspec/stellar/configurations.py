from pfsspec.core import Pipeline
from pfsspec.core.grid import ArrayGrid, RbfGrid
from pfsspec.stellar.grid.bosz.io import BoszGridReader
from pfsspec.stellar.grid.bosz.io import BoszGridDownloader
from pfsspec.stellar.grid.phoenix.io import PhoenixGridReader

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

# FIT_CONFIGURATIONS = {
#     'grid': {
#         'bosz': {
#             'class': ModelGridFit,
#             'config': Bosz()
#         },
#         'phoenix': {
#             'class': ModelGridFit,
#             'config': Phoenix()
#         }
#     }
# }

# PCA_CONFIGURATIONS = {
#     'grid': {
#         'bosz': {
#             'class': ModelPcaGridBuilder,
#             'config': Bosz()
#         },
#         'phoenix': {
#             'class': ModelPcaGridBuilder,
#             'config': Phoenix()
#         }
#     }
# }

# RBF_CONFIGURATIONS = {
#     'grid': {
#         'bosz': {
#             'class': ModelRbfGridBuilder,
#             'config': Bosz()
#         },
#         'phoenix': {
#             'class': ModelRbfGridBuilder,
#             'config': Phoenix()
#         }
#     }
# }