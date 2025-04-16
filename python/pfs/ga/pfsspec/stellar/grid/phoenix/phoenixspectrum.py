import numpy as np
from scipy.integrate import simpson

from pfs.ga.pfsspec.stellar import ModelSpectrum

class PhoenixSpectrum(ModelSpectrum):
    def __init__(self, orig=None):
        super().__init__(orig=orig)

        self.is_flux_calibrated = True

        if not isinstance(orig, PhoenixSpectrum):
            pass
        else:
            pass
   
    def synthmag_carrie(self, filter, log_L):
        #remember - phoenix flux needs to be multiplied by *1e-8

        #normalising spectra
        #getting bounds of integral
        lam = self.wave[(self.wave <= filter.wave.max()) & (self.wave >= filter.wave.min())]
        T = np.interp(lam, filter.wave, filter.thru)
        T = np.where(T < .001, 0, T)

        R = self.get_radius(log_L, np.log10(self.T_eff))

        #1/(3.08567758128*10**(19))**2 is just 1/10pc^2 in cm! (1/(3.086e19)**2)
        
        s = 1e-8 * self.flux[(self.wave <= filter.wave.max()) & (self.wave >= filter.wave.min())]
        s = s * (R / 3.086e19) ** 2           #NOT multiplied by pi!

        # Interpolating to get filter data on same scale as spectral data
        # Doing classic integral to get flux in bandpass
        stzp = 3.631e-9
        i1 = simpson(s * T * lam, lam)
        i2 = simpson(T * lam, lam)
        i3 = simpson(T / lam, lam)
        a = -2.5 * np.log10(i1 / (stzp * i2))
        b = -2.5 * np.log10(i2 / i3)

        return a + b + 18.6921

    def normalize_to_mag(self, filt, mag):
        try:
            m = self.synthmag(filt)
            # if m <= -10:
            #     # Checking that not really negative number, which happens when flux is from
            #     # Phoenix but isn't properly re-scaled - i.e. flux is ~1e8 too big
            #     # this step probably isn't really catching everything - must look into better way
            # m = self.synthmag_carrie(filt)
        except Exception as ex:
            print('flux max', np.max(self.flux))
            print('mag', mag)
            raise ex
        DM = mag - m
        D = 10 ** (DM / 5)

        self.multiply(1 / D**2)
        self.mag = mag