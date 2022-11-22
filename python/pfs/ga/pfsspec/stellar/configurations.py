from pfs.ga.pfsspec.core import Pipeline
from pfs.ga.pfsspec.core.grid import ArrayGrid, RbfGrid
from pfs.ga.pfsspec.stellar.grid.bosz import Bosz
from pfs.ga.pfsspec.stellar.grid.phoenix import Phoenix
from pfs.ga.pfsspec.stellar.grid.bosz.io import BoszGridReader
from pfs.ga.pfsspec.stellar.grid.bosz.io import BoszGridDownloader
from pfs.ga.pfsspec.stellar.grid.phoenix.io import PhoenixGridReader
from pfs.ga.pfsspec.stellar.grid.phoenix.io import PhoenixGridDownloader
from pfs.ga.pfsspec.stellar.grid import ModelGridFit, ModelPcaGridBuilder, ModelRbfGridBuilder
from pfs.ga.pfsspec.stellar.grid import ModelGridConverter
from pfs.ga.pfsspec.learn.stellar import *

from pfs.ga.pfsspec.learn.configurations import KERAS_DNN_MODEL_TYPES, TORCH_DNN_MODEL_TYPES
from pfs.ga.pfsspec.learn.dnn import SimpleModelTrainer, AutoencodingModelTrainer
from pfs.ga.pfsspec.learn.dnn import SimpleModelPredictor, AutoencodingModelPredictor


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

CONVERT_CONFIGURATIONS = {
    'stellar-grid': {
        'phoenix': {
            'type': ModelGridConverter,
            'config': Phoenix(),
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

TRAIN_CONFIGURATIONS = {
    'stellar-model': {
        'reg': {
            'type': SimpleModelTrainer,
            'augmenter': ModelSpectrumRegressionalAugmenter,
            'models': KERAS_DNN_MODEL_TYPES['reg']
        },
        'gen': {
            'type': SimpleModelTrainer,
            'augmenter': ModelSpectrumGenerativeAugmenter,
            'models': KERAS_DNN_MODEL_TYPES['gen']
        },
        'ae': {
            'type': AutoencodingModelTrainer,
            'augmenter': ModelSpectrumAutoencodingAugmenter,
            'models': KERAS_DNN_MODEL_TYPES['ae']
        },

        'ae-torch': {
            'type': AutoencodingModelTrainer,
            'augmenter': ModelSpectrumAutoencodingAugmenter,
            'models': TORCH_DNN_MODEL_TYPES['ae-torch']
        }
    }
}


PREDICT_CONFIGURATIONS = {
    'stellar-model': {
        'reg': {
            'type': SimpleModelPredictor,
            'augmenter': ModelSpectrumRegressionalAugmenter,
            'models': KERAS_DNN_MODEL_TYPES['reg']
        },
        'gen': {
            'type': SimpleModelPredictor,
            'augmenter': ModelSpectrumGenerativeAugmenter,
            'models': KERAS_DNN_MODEL_TYPES['gen']
        },
        'ae': {
            'type': AutoencodingModelPredictor,
            'augmenter': ModelSpectrumAutoencodingAugmenter,
            'models': KERAS_DNN_MODEL_TYPES['ae']
        }
    }
}