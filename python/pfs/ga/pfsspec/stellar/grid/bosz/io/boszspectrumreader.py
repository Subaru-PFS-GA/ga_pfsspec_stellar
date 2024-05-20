import os
import math
import numpy as np
import pandas as pd
import re
import multiprocessing
import time
from astropy.io import fits

from pfs.ga.pfsspec.core.io import SpectrumReader
from pfs.ga.pfsspec.stellar.grid.bosz import BoszSpectrum

class BoszSpectrumReader(SpectrumReader):

    # NOTE: wavelength values in the files have serious round-off errors,
    #       `correct_wave_grid` is used to calculate bins from scratch at 
    #       R = 300000, then the grid is downsampled to the required resolution
    #       by taking each kth element. Not that BOSZ spectra have two bins
    #       for each resolution element.

    MAP_FE_H = {
        'm03': -0.25,
        'm05': -0.5,
        'm08': -0.75,
        'm10': -1.0,
        'm13': -1.25,
        'm15': -1.5,
        'm18': -1.75,
        'm20': -2.0,
        'm23': -2.25,
        'm25': -2.5,
        'm28': -2.75,
        'm30': -3.0,
        'm35': -3.5,
        'm40': -4.0,
        'm45': -4.5,
        'm50': -5.0,
        'p00': 0.0,
        'p03': 0.25,
        'p05': 0.5,
        'p08': 0.75,
        'p10': 1.00,
        'p15': 1.50
    }

    MAP_C_M = {
        'm03': -0.25,
        'm05': -0.5,
        'p00': 0.0,
        'p03': 0.25,
        'p05': 0.5,
    }

    MAP_O_M = {
        'm03': -0.25,
        'p00': 0.0,
        'p03': 0.25,
        'p05': 0.5,
    }

    def __init__(self, path=None, format=None, wave_lim=None, resolution=None, orig=None):
        super(BoszSpectrumReader, self).__init__(wave_lim=wave_lim, orig=orig)

        if not isinstance(orig, BoszSpectrumReader):
            self.path = path
            self.format = format
            self.resolution = resolution
        else:
            self.path = path or orig.path
            self.format = format or orig.format
            self.resolution = resolution or orig.resolution

    def add_args(self, parser):
        super(BoszSpectrumReader, self).add_args(parser)

        parser.add_argument("--format", type=str, default='ascii', choices=['ascii', 'fits'], help="Data format.\n")
        parser.add_argument("--resolution", type=int, default=None, help="Resolution.\n")

    def init_from_args(self, args):
        super(BoszSpectrumReader, self).init_from_args(args)

        self.format = self.get_arg('format', self.format, args)
        self.resolution = self.get_arg('resolution', self.resolution, args)
        
    def correct_wave_grid(self, wlim, resolution):
        # BOSZ spectra are written to the disk with 3 decimals which aren't
        # enough to represent wavelength at high resolutions. This code is
        # from the original Kurucz SYNTHE to recalculate the wavelength grid.

        RESOLU = resolution
        WLBEG = wlim[0]  # nm
        WLEND = wlim[1]  # nm
        RATIO = 1. + 1. / RESOLU
        RATIOLG = np.log10(RATIO)
        
        IXWLBEG = int(np.round(np.log10(WLBEG) / RATIOLG))
        WBEGIN = 10 ** (IXWLBEG * RATIOLG)
        if WBEGIN < WLBEG:
            IXWLBEG = IXWLBEG + 1
            WBEGIN = 10 ** (IXWLBEG * RATIOLG)
            
        IXWLEND = int(np.round(np.log10(WLEND) / RATIOLG))
        WLLAST = 10 ** (IXWLEND * RATIOLG)
        if WLLAST >= WLEND:
            IXWLEND = IXWLEND - 1
            WLLAST = 10 ** (IXWLEND * RATIOLG)
        LENGTH = IXWLEND - IXWLBEG + 1
        DWLBEG = WBEGIN * RATIO - WBEGIN
        DWLLAST = WLLAST - WLLAST / RATIO
        
        a = np.linspace(np.log10(WBEGIN), np.log10(WLLAST), LENGTH)
        cwave = 10 ** a
        
        return cwave

    def read(self, file=None):
        if self.format == 'ascii':
            spec = self.read_ascii(file)
        elif self.format == 'fits':
            spec = self.read_fits(file)
        else:
            raise NotImplementedError()

        log_wave = np.log(spec.wave)
        wave_edges = np.empty((2, spec.wave.shape[0]), dtype=spec.wave.dtype)        
        wave_edges[0, 1:] = wave_edges[1, :-1] = np.exp((log_wave[1:] + log_wave[:-1]) / 2)
        wave_edges[0, 0] = np.exp(log_wave[0] - (log_wave[1] - log_wave[0]) / 2)
        wave_edges[1, -1] = np.exp(log_wave[-1] + (log_wave[-1] - log_wave[-2]) / 2)
        spec.wave_edges = wave_edges

        spec.resolution = self.resolution
        spec.is_wave_regular = True
        spec.is_wave_lin = False
        spec.is_wave_log = True

        return spec

    def read_fits(self, file=None):

        if file is None:
            file = self.path

        hdus = fits.open(file, memmap=False)
        
        # Use original wavelengths from the file
        # wave = hdus[1].data['Wavelength']

        # Use corrected wavelengths
        d = int(300000 // self.resolution // 2)
        wave = 10 * self.correct_wave_grid(wlim=(100, 32000), resolution=300000)[::d]

        if self.wave_lim is not None:
            filt = (self.wave_lim[0] <= wave) & (wave <= self.wave_lim[1])
        else:
            filt = slice(None)

        spec = BoszSpectrum()
        spec.wave = np.array(wave[filt])
        spec.flux = np.array(hdus[1].data['SpecificIntensity'][filt])
        spec.cont = np.array(hdus[1].data['Continuum'][filt])

        return spec

    def read_ascii(self, file):
        compression = None
        
        if file is None:
            file = self.path

        if type(file) is str:
            fn, ext = os.path.splitext(file)
            if ext == '.bz2':
                compression = 'bz2'

        # for some reason the C implementation of read_csv throws intermittent errors
        # when forking using multiprocessing
        # engine='python',
        try:
            df = pd.read_csv(file, delimiter=r'\s+', header=None, compression=compression)
        except Exception as ex:
            with open(file + '.error', 'w') as f:
                f.write(str(ex))
            raise Exception("Unable to read file {}".format(file)) from ex
        df.columns = ['wave', 'flux', 'cont']

        # Use original wavelengths from the file
        # wave = np.array(df['wave'])

        # Use corrected wavelengths
        d = int(300000 // self.resolution // 2)
        wave = 10 * self.correct_wave_grid(wlim=(100, 32000), resolution=300000)[::d]

        if self.wave_lim is not None:
            filt = (self.wave_lim[0] <= wave) & (wave <= self.wave_lim[1])
        else:
            filt = slice(None)

        spec = BoszSpectrum()
        spec.wave = np.array(wave[filt])
        spec.cont = np.array(df['cont'][filt])
        spec.flux = np.array(df['flux'][filt])

        return spec

    def get_filename(self, **kwargs):
        # amm03cm03om03t3500g25v20modrt0b5000rs.asc.bz2
        # amm15cm03op03t4250g25v20modrt0b2000rs.fits

        M_H = kwargs.pop('M_H')
        C_M = kwargs.pop('C_M')
        a_M = kwargs.pop('a_M')
        T_eff = kwargs.pop('T_eff')
        log_g = kwargs.pop('log_g')
        v_turb = kwargs.pop('v_turb', 0.2)
        v_rot = kwargs.pop('v_rop', 0)
        R = self.resolution

        fn = 'a'

        fn += 'm'
        fn += 'm' if M_H < 0 else 'p'
        fn += '%02d' % (int(abs(M_H) * 10 + 0.5))

        fn += 'c'
        fn += 'm' if C_M < 0 else 'p'
        fn += '%02d' % (int(abs(C_M) * 10 + 0.5))

        fn += 'o'
        fn += 'm' if a_M < 0 else 'p'
        fn += '%02d' % (int(abs(a_M) * 10 + 0.5))

        fn += 't'
        fn += '%d' % (int(T_eff))

        fn += 'g'
        fn += '%02d' % (int(log_g * 10))

        fn += 'v'
        fn += '%02d' % (int(v_turb * 100))

        fn += 'mod'

        fn += 'rt'
        fn += '%d' % (int(v_rot))

        fn += 'b'
        fn += '%d' % (R)

        if self.format == 'ascii':
            fn += 'rs.asc.bz2'
        elif self.format == 'fits':
            fn += 'rs.fits'
        else:
            raise NotImplementedError()

        return fn

    def get_url(self, **kwargs):
        # http://archive.stsci.edu/missions/hlsp/bosz/ascii/insbroad_050000/metal_+0.00/carbon_+0.00/alpha_+0.00/amp00cp00op00t3500g00v20modrt0b50000rs.asc.bz2
        # http://archive.stsci.edu/missions/hlsp/bosz/fits/insbroad_002000/metal_-1.50/carbon_-0.25/alpha_+0.25/amm15cm03op03t4250g25v20modrt0b2000rs.fits

        if self.format == 'ascii':
            url = "https://archive.stsci.edu/missions/hlsp/bosz/ascii"
        elif self.format == 'fits':
            url = "https://archive.stsci.edu/missions/hlsp/bosz/fits"

        url += "/insbroad_{:0>6}".format(self.resolution)
        url += "/metal_{:+.2f}".format(kwargs['M_H'])
        url += "/carbon_{:+.2f}".format(kwargs['C_M'])
        url += "/alpha_{:+.2f}".format(kwargs['a_M'])
        url += "/" + self.get_filename(**kwargs)
        
        return url

    @staticmethod
    def parse_filename(filename):

        # amm03cm03om03t3500g25v20modrt0b5000rs.asc.bz2
        # amm15cm03op03t4250g25v20modrt0b2000rs.fits

        p = r'am([pm]\d{2})c([pm]\d{2})o([pm]\d{2})t(\d{4,5})g(\d{2})v20'
        m = re.search(p, filename)

        return{
            'M_H': BoszSpectrumReader.MAP_FE_H[m.group(1)],
            'C_M': BoszSpectrumReader.MAP_C_M[m.group(2)],
            'a_M': BoszSpectrumReader.MAP_O_M[m.group(3)],
            'T_eff': float(m.group(4)),
            'log_g': float(m.group(5)) / 10
        }
