from .tempfit import TempFit
from .tempfitflag import TempFitFlag
from .tempfittrace import TempFitTrace
from .tempfittrace import TempFitTrace as TempFitTrace
from .modelgridtempfit import ModelGridTempFit
from .modelgridtempfittrace import ModelGridTempFitTrace

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