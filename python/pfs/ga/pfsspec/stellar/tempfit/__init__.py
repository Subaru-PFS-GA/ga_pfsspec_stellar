from .tempfit import TempFit
from .tempfittrace import TempFitTrace
from .tempfittrace import TempFitTrace as TempFitTrace
from .modelgridtempfit import ModelGridTempFit
from .modelgridtempfittrace import ModelGridTempFitTrace

from .fluxcorr import FluxCorr
from .contnorm import ContNorm

CORRECTION_MODELS = {
    'fluxcorr': FluxCorr,
    'contnorm': ContNorm,
}