import os
import math
import numpy as np
import pandas as pd
import re
import gzip
import multiprocessing
import time
from astropy.io import fits

from pfs.ga.pfsspec.core.io import SpectrumReader
from pfs.ga.pfsspec.stellar.grid.grid7 import Grid7Spectrum

class Grid7SpectrumReader(SpectrumReader):
    def __init__(self, path=None, format=None, wave_lim=None, resolution=None, orig=None):
        super().__init__(wave_lim=wave_lim, orig=orig)

        if not isinstance(orig, Grid7SpectrumReader):
            self.path = path
            self.format = format
            self.resolution = resolution
        else:
            self.path = path or orig.path
            self.format = format or orig.format
            self.resolution = resolution or orig.resolution

    def add_args(self, parser):
        super().add_args(parser)

        parser.add_argument("--format", type=str, default='bin', choices=['bin'], help="Data format.\n")
        parser.add_argument("--resolution", type=int, default=None, help="Resolution.\n")

    def init_from_args(self, args):
        super().init_from_args(args)

        self.format = self.get_arg('format', self.format, args)
        self.resolution = self.get_arg('resolution', self.resolution, args)

    def read(self, file=None):
        if self.format == 'bin':
            flux = self.read_bin(file)
        else:
            raise NotImplementedError()
        
        start = 6300
        stop = 9100
        step = 0.14
        wave = np.arange(start, stop + 0.01 * step, step)
        wave_edges = np.stack([wave - step / 2, wave + step / 2])

        if self.wave_lim is not None:
            mask = (self.wave_lim[0] <= wave) & (wave <= self.wave_lim[1])
        else:
            mask = slice(None)

        spec = Grid7Spectrum()
        spec.flux = flux
        spec.wave = wave
        spec.wave_edges = wave_edges
        
        spec.is_wave_regular = True
        spec.is_wave_lin = True
        spec.is_wave_log = False

        spec.resolution = self.resolution
        spec.continuum_normalized = True

        return spec

    def read_bin(self, file):
        with gzip.open(file, 'rb') as f:
            flux = 1.0 - np.frombuffer(f.read(), dtype = np.float32)
            return flux

    def get_filename(self, **kwargs):

        # TODO
        if self.format == 'bin':
            pass
        else:
            raise NotImplementedError()

        # bin/t5000/g_10/t5000g_10f_00a_00.bin.gz

        M_H = kwargs.pop('M_H')
        a_M = kwargs.pop('a_M')
        T_eff = kwargs.pop('T_eff')
        log_g = kwargs.pop('log_g')

        dir = 'bin'
        fn = ''

        part = f't{int(T_eff):04d}'
        dir = os.path.join(dir, part)
        fn += part

        sign = '_' if log_g >= 0.0 else '-'
        part = f'g{sign}{int(np.abs(log_g * 10)):02d}'
        dir = os.path.join(dir, part)
        fn += part

        sign = '_' if M_H >= 0.0 else '-'
        part = f'f{sign}{int(np.abs(M_H * 10)):02d}'
        fn += part

        sign = '_' if a_M >= 0.0 else '-'
        part = f'a{sign}{int(np.abs(a_M * 10)):02d}'
        fn += part

        fn += '.bin.gz'

        return os.path.join(dir, fn)

    @staticmethod
    def parse_filename(filename):
        if filename.lower().endswith('.bin.gz'):
            # bin/t5000/g_10/t5000g_10f_00a_00.bin.gz

            parts = list(re.findall('t([0-9]{4})g([_-][0-9]{2})f([_-][0-9]{2})a([_-][0-9]{2})\.', filename.replace('a_00a', 'a'))[0])
            parts[0] = float(parts[0])
            for i in range(1, len(parts)):
                parts[i] = np.round(float(parts[i].replace('_', '')) / 10.0, 1)

            return {
                'T_eff': parts[0],
                'log_g': parts[1],
                'M_H': parts[2],
                'a_M': parts[3],
            }
        else:
            raise NotImplementedError()
