from pfs.ga.pfsspec.core import Pipeline
from pfs.ga.pfsspec.core.grid import ArrayGrid, RbfGrid
from pfs.ga.pfsspec.stellar.grid.bosz import Bosz
from pfs.ga.pfsspec.stellar.grid.bosz.io import BoszGridReader, BoszGridDownloader
from pfs.ga.pfsspec.stellar.grid.phoenix import Phoenix
from pfs.ga.pfsspec.stellar.grid.phoenix.io import PhoenixGridReader, PhoenixGridDownloader
from pfs.ga.pfsspec.stellar.grid.grid7.io import Grid7GridReader
from pfs.ga.pfsspec.stellar.grid.grid7 import Grid7
from pfs.ga.pfsspec.stellar.grid import ModelGridFit, ModelPcaGridBuilder, ModelRbfGridBuilder
from pfs.ga.pfsspec.stellar.grid import ModelGridConverter
from pfs.ga.pfsspec.stellar.dataset import ModelDatasetMerger
from pfs.ga.pfsspec.learn.stellar import *

from pfs.ga.pfsspec.learn.configurations import KERAS_DNN_MODEL_TYPES, TORCH_DNN_MODEL_TYPES
from pfs.ga.pfsspec.learn.dnn import SimpleModelTrainer, ClassificationModelTrainer, AutoencodingModelTrainer
from pfs.ga.pfsspec.learn.dnn import SimpleModelPredictor, ClassificationModelPredictor, AutoencodingModelPredictor


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

MERGE_CONFIGURATIONS = {
    'dataset': {    
        'model': {
            'type': ModelDatasetMerger
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
        },
        'grid7': {
            'type': Grid7GridReader,
            'pipelines': {
                'basic': Pipeline
            }
        },
    }
}

CONVERT_CONFIGURATIONS = {
    'stellar-grid': {
        'bosz': {
            'type': ModelGridConverter,
            'config': Bosz(),
            'pipelines': {
                'basic': Pipeline
            }
        },
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
        'class': {
            'type': ClassificationModelTrainer,
            'augmenter': ModelSpectrumClassificationAugmenter,
            'models': KERAS_DNN_MODEL_TYPES['class']
        },
        'ae': {
            'type': AutoencodingModelTrainer,
            'augmenter': ModelSpectrumAutoencodingAugmenter,
            'models': KERAS_DNN_MODEL_TYPES['ae']
        },
        # 'reg-torch': {
        #     'type': SimpleModelTrainer,
        #     'augmenter': ModelSpectrumRegressionalAugmenter,
        #     'models': TORCH_DNN_MODEL_TYPES['reg-torch']
        # },
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
        'class': {
            'type': ClassificationModelPredictor,
            'augmenter': ModelSpectrumClassificationAugmenter,
            'models': KERAS_DNN_MODEL_TYPES['class']
        },
        'ae': {
            'type': AutoencodingModelPredictor,
            'augmenter': ModelSpectrumAutoencodingAugmenter,
            'models': KERAS_DNN_MODEL_TYPES['ae']
        },
        # 'reg-torch': {
        #     'type': SimpleModelPredictor,
        #     'augmenter': ModelSpectrumRegressionalAugmenter,
        #     'models': TORCH_DNN_MODEL_TYPES['reg-torch']
        # },
        'ae-torch' : {
            'type': AutoencodingModelPredictor,
            'augmenter': ModelSpectrumAutoencodingAugmenter,
            'models': TORCH_DNN_MODEL_TYPES['ae-torch']
        },
    }
}