from .tempfit import TempFit
from .tempfitflag import TempFitFlag
from .tempfittrace import TempFitTrace
from .tempfitstate import TempFitState
from .modelgridtempfit import ModelGridTempFit
from .modelgridtempfittrace import ModelGridTempFitTrace
from .modelgridtempfitstate import ModelGridTempFitState

from .fluxcorr import FluxCorr
from .contnorm import ContNorm
from .extinctionmodel import ExtinctionModel

CORRECTION_MODELS = {
    'fluxcorr': FluxCorr,
    'contnorm': ContNorm,
}

EXTINCTION_MODELS = {
    'default': ExtinctionModel,
}