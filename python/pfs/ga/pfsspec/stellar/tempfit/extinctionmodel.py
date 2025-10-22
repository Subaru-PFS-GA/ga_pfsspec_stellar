import numpy as np
import extinction

from pfs.ga.pfsspec.core.util.copy import safe_deep_copy
from pfs.ga.pfsspec.core.util.args import get_arg

from .setup_logger import logger

class ExtinctionModel():
    """
    Extinction model.
    """

    def __init__(self, trace=None, orig=None):
        if not isinstance(orig, ExtinctionModel):
            self.trace = trace
            self.tempfit = None

            self.ext_type = 'ccm89'
            self.R_V = 3.1
        else:
            self.trace = trace if trace is not None else orig.trace
            self.tempfit = orig.tempfit

            self.ext_type = orig.ext_type
            self.R_V = orig.R_V

        self.reset()

    def reset(self):
        self.curves = None

    def add_args(self, config, parser):
        parser.add_argument('--ext-type', type=str, help='Extinction model.\n')
        parser.add_argument('--R_V', type=float, help='Value of R_V.\n')

    def init_from_args(self, script, config, args):
        self.ext_type = get_arg('ext_type', self.ext_type, args)
        self.R_V = get_arg('R_V', self.R_V, args)

    def get_wave_include(self):
        # By default, include all wavelengths
        return None
    
    def get_wave_exclude(self):
        # By default, exclude no wavelengths
        return None

    def init_curves(self, spectra, force=False):
        if self.curves is None or force:
            self.curves = self.eval_extinction(spectra)
        elif self.curves is not None and not force:
            logger.info("Flux correction model is already initialized, skipping reinitialization.")

    def __eval(self, wave, ebv):
        """
        Evaluate the extinction curve at the given wavelengths for the given E(B-V).

        Parameters:
        ----------
        wave : array-like
            Wavelengths in Angstroms.
        ebv : float
            E(B-V) value in magnitudes.

        Returns:
        -------
        A_lambda : array-like
            Extinction at the given wavelengths in magnitudes.
        """

        if self.ext_type == 'ccm89':
            return extinction.ccm89(wave, a_v=ebv * self.R_V, r_v=self.R_V)
        elif self.ext_type == 'odonnell94':
            return extinction.odonnell94(wave, a_v=ebv * self.R_V, r_v=self.R_V)
        elif self.ext_type == 'fitzpatrick99':
            return extinction.fitzpatrick99(wave, a_v=ebv * self.R_V, r_v=self.R_V)
        elif self.ext_type == 'fm07':
            return extinction.fm07(wave, a_v=ebv * self.R_V)
        elif self.ext_type == 'calzetti00':
            return extinction.calzetti00(wave, a_v=ebv * self.R_V, r_v=self.R_V)
        else:
            raise NotImplementedError(f"Extinction curve '{self.curve}' is not implemented.")

    def eval_extinction(self, spectra):
        """
        Calculate the extinction curve for the given spectra
        """

        ext_curve = { arm: [] for arm in spectra }

        for arm in spectra:
            for i in range(len(spectra[arm])):
                if spectra[arm][i] is not None:
                    ext_curve[arm].append(self.__eval(spectra[arm][i].wave, 1.0))
                else:
                    ext_curve[arm].append(None)

        return ext_curve
    
    def apply_extinction(self, templates, ebv):
        for arm in templates:
            for i in range(len(templates[arm])):
                if templates[arm][i] is not None:
                    templates[arm][i].apply_extinction(self.curves[arm][i], ebv)

    def eval_extinction_single(self, spectrum):
        """
        Calculate the extinction curve for a single spectrum
        """
        return self.__eval(spectrum.wave, 1.0)

    def apply_extinction_single(self, template, ebv, ext_curve):
        template.apply_extinction(ext_curve, ebv)