import os
import zipfile
import pickle
import numpy as np
import pandas as pd
import re
from astropy.io import fits

from pfs.ga.pfsspec.core.io import SpectrumReader
from pfs.ga.pfsspec.core.obsmod.resampling import Binning
from ..gk2025 import GK2025
from ..gk2025spectrum import GK2025Spectrum

class GK2025SpectrumReader(SpectrumReader):

    def __init__(self, path=None, format=None, wave_lim=None, resolution=None, carbon=None, orig=None):
        super(GK2025SpectrumReader, self).__init__(wave_lim=wave_lim, orig=orig)

        if not isinstance(orig, GK2025SpectrumReader):
            self.path = path
            self.format = format
            self.resolution = resolution
        else:
            self.path = orig.path
            self.format = orig.format
            self.resolution = orig.resolution

        self.index = None
        self.wave = None
        self.wave_edges = None

    def add_args(self, parser):
        super().add_args(parser)

        parser.add_argument("--format", type=str, default=None, help="File format (e.g. 'pkl', 'ascii').\n")
        parser.add_argument("--resolution", type=int, default=None, help="Resolution.\n")

    def init_from_args(self, args):
        super(GK2025SpectrumReader, self).init_from_args(args)

        self.format = self.get_arg('format', self.format, args)
        self.resolution = self.get_arg('resolution', self.resolution, args)

    def read(self, file=None):
        # TODO: handle formats here?
        
        if self.format is None or self.format == 'zip,pkl':
            zip_name, file_name = file

            # Open the zip file and read the specified file inside it
            with zipfile.ZipFile(zip_name, 'r') as z:
                with z.open(file_name) as f:
                    gkspec = pickle.load(f)
        else:
            raise NotImplementedError(f"Format '{self.format}' not implemented.")

        # Calculate the wavelength grid
        if self.wave is None:
            lgr = np.log(1.0 + 1.0 / gkspec['meta']['res'])
            wl_start = np.ceil(np.log(gkspec['meta']['wl_start']) / lgr)
            wl_end = np.floor(np.log(gkspec['meta']['wl_end']) / lgr) + 1
            self.wave = np.exp(np.arange(wl_start, wl_end) * lgr) * 10
            self.wave_edges = np.exp((np.arange(wl_start, wl_end + 1) - 0.5) * lgr) * 10

            if 'binning' in gkspec['meta'] and gkspec['meta']['binning'] > 1:
                binning = gkspec['meta']['binning']
                self.wave_edges = np.concatenate([
                    self.wave_edges[:(len(self.wave) // binning) * binning].reshape(-1, binning)[:, 0],
                    self.wave_edges[1:(len(self.wave) // binning) * binning + 1].reshape(-1, binning)[-1:, -1]
                ])
                self.wave = self.wave[:(len(self.wave) // binning) * binning].reshape(-1, binning).mean(axis = 1)

        if self.wave_lim is not None:
            filt = (self.wave_lim[0] <= self.wave) & (self.wave <= self.wave_lim[1])
            filt_edges = np.zeros_like(filt, dtype=bool)
            filt_edges[:-1] |= filt
            filt_edges[1:] |= filt     
        else:
            filt = slice(None)
            filt_edges = slice(None)

        spec = GK2025Spectrum()
        spec.wave = self.wave[filt]
        spec.wave_edges = self.wave_edges[filt_edges]
        spec.flux = gkspec['null']['cont'][filt] * gkspec['null']['line'][filt]
        spec.cont = gkspec['null']['cont'][filt]
        spec.line = gkspec['null']['line'][filt]

        spec.T_eff = gkspec['meta']['teff']
        spec.log_g = gkspec['meta']['logg']

        # TODO: figure out the rest of the parameters from the file name

        spec.is_wave_regular = True
        spec.is_wave_lin = False
        spec.is_wave_log = True
        spec.is_wave_vacuum = False

        spec.is_flux_calibrated = True

        return spec

    def load_index(self):
        type = os.path.basename(self.path)
        index_file = os.path.join(self.path, '..', f'index_{type}.pkl')
        with open(index_file, 'br') as f:
            self.index = pickle.load(f)

    def get_filename(self, **kwargs):        
        # File names are looked up in the index
        if self.index is None:
            self.load_index()

        M_H = kwargs.get('M_H', 0.0)
        T_eff = int(kwargs.get('T_eff', 5000.0))
        log_g = kwargs.get('log_g', 1.0)
        a_M = kwargs.get('a_M', 0.0)
        C = kwargs.get('C', 0.0)

        key = f't{T_eff:04.3f}l{log_g:01.3f}z{M_H:0.3f}a{a_M:0.3f}c{C:0.3f}'
        parts = self.index[1][key].split('/')
        
        zip_name = parts[0]
        file_name = '/'.join(parts[1:])

        return zip_name, file_name

    def prefix_filename(self, fn):
        zip_path, file_name = fn
        return os.path.join(self.path, zip_path), file_name

    def is_file(self, fn):
        zip_path, file_name = fn
        if os.path.isfile(zip_path) and zipfile.is_zipfile(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as z:
                return file_name in z.namelist()
    
        return False

    def get_url(self, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def parse_filename(filename):
        raise NotImplementedError()