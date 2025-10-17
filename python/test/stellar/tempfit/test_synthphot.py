import os
import numpy as np
import numpy.testing as npt

from test.pfs.ga.pfsspec.core import TestBase
from pfs.ga.pfsspec.core import Spectrum, Filter, Physics


from .tempfittestbase import TempFitTestBase

class TestSynthMag(TempFitTestBase):
    def test_synth_mag(self):
        fn = os.path.join(self.PFSSPEC_DATA_PATH, f'subaru/hsc/filters/HSC-g.txt')
        filter = Filter()
        filter.read(fn)

        grid, temp = self.get_test_spectrum(M_H=-2.0, T_eff=4500, log_g=1.5, C_M=0, a_M=0)

        import pysynphot
        import pysynphot.spectrum

        filt = pysynphot.spectrum.ArraySpectralElement(filter.wave, filter.thru, waveunits='angstrom')
        spec = pysynphot.spectrum.ArraySourceSpectrum(wave=temp.wave, flux=temp.flux, keepneg=True, fluxunits='flam')
        mask = (filt.wave.min() <= spec.wave) & (spec.wave <= filt.wave.max())
        filt.binset = spec.wave[mask]

        obs = pysynphot.observation.Observation(spec, filt)
        fnu = obs.effstim('Jy')
        mag = Physics.jy_to_abmag(fnu)

        fnu2, fnu2_err = filter.synth_flux(temp.wave, temp.flux)
        mag2 = Physics.fnu_to_abmag(fnu2)

        pass