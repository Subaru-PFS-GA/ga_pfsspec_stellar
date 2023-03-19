#!/usr/bin/env python

PROJECT_PATH = '/home/dobos/project/ga_isochrones/python:' + \
    '/home/dobos/project/ga_pfsspec_all/python:' + \
    '/home/dobos/project/pysynphot'

GRID_PATH = '/datascope/subaru/data/pfsspec/models/stellar/grid/phoenix/phoenix_HiRes'
FILTER_PATH = '/datascope/subaru/data/pfsspec/subaru/hsc/filters/fHSC-g.txt'

ARMS = [ 'b', 'r', 'mr', 'n' ]
FIT_ARMS = ['b', 'mr', 'n']

DETECTOR_PATH = '/datascope/subaru/data/pfsspec/subaru/pfs/arms/{}.json'
PSF_PATH = '/datascope/subaru/data/pfsspec/subaru/pfs/psf/import/{}.2'
SKY_PATH = '/datascope/subaru/data/pfsspec/subaru/pfs/noise/import/sky.see/{}/sky.h5'
MOON_PATH = '/datascope/subaru/data/pfsspec/subaru/pfs/noise/import/moon/{}/moon.h5'

##

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.interpolate import interp1d
import h5py as h5
from tqdm.notebook import trange, tqdm

## Allow load project as module
for p in reversed(PROJECT_PATH.split(':')):
    sys.path.insert(0, p)

## Matplotlib setup

plt.rc('font', size=7)

# Project imports

from pfs.ga.pfsspec.core.grid import ArrayGrid
from pfs.ga.pfsspec.stellar.grid import ModelGrid
from pfs.ga.pfsspec.stellar.grid.bosz import Bosz
from pfs.ga.pfsspec.stellar.grid.phoenix import Phoenix
from pfs.ga.pfsspec.core import Filter
from pfs.ga.pfsspec.sim.obsmod.pipelines import StellarModelPipeline
from pfs.ga.pfsspec.core import Physics
from pfs.ga.pfsspec.core.obsmod.psf import GaussPsf, PcaPsf
from pfs.ga.pfsspec.sim.obsmod import Detector
from pfs.ga.pfsspec.sim.obsmod.background import Sky
from pfs.ga.pfsspec.sim.obsmod.background import Moon
from pfs.ga.pfsspec.core.obsmod.snr import QuantileSnr
from pfs.ga.pfsspec.stellar import StellarSpectrum
from pfs.ga.pfsspec.sim.obsmod.observations import PfsObservation
from pfs.ga.pfsspec.sim.obsmod.noise import NormalNoise
from pfs.ga.pfsspec.sim.obsmod.calibration import FluxCalibrationBias
from pfs.ga.pfsspec.core.obsmod.resampling import FluxConservingResampler
from pfs.ga.pfsspec.stellar.rvfit import RVFit, RVFitTrace
from pfs.ga.pfsspec.core.util import SmartParallel
from collections.abc import Iterable

def load_spectrum_grid():
    fn = os.path.join(GRID_PATH, 'spectra.h5')
    grid = ModelGrid(Phoenix(), ArrayGrid)
    grid.preload_arrays = False
    grid.load(fn, format='h5')
    return grid

def load_psf(arm, grid):
    d = Detector()
    d.load_json(DETECTOR_PATH.format(arm))

    gauss_psf = GaussPsf()
    gauss_psf.load(os.path.join(PSF_PATH.format(arm), 'gauss.h5'))

    print(f'mean pixel size for arm {arm}', np.diff(d.get_wave()[0]).mean())
    print(f'mean sigma and FWHM for arm {arm}', gauss_psf.sigma.mean(), 2.355 * gauss_psf.sigma.mean())

    s = gauss_psf.get_optimal_size(grid.wave)
    print(f'optimal kernel size for arm {arm}:', s)

    pca_psf = PcaPsf()
    pca_psf.load(os.path.join(PSF_PATH.format(arm), 'pca.h5'))

    template_psf = PcaPsf.from_psf(gauss_psf, grid.wave, size=s, truncate=5)
    print(grid.wave.shape, 
        template_psf.wave.shape, template_psf.dwave.shape, template_psf.pc.shape)

    return gauss_psf, pca_psf, template_psf

def load_filter():
    filt = Filter()
    filt.read(FILTER_PATH)
    return filt

def create_observation(arm, gauss_psf):
    detector = Detector()
    detector.load_json(DETECTOR_PATH.format(arm))
    detector_wave, _ = detector.get_wave()
    detector_s = gauss_psf.get_optimal_size(detector_wave)
    print(f'Optimal size of PSF kernel for arm {arm}', detector_s)
    detector.psf = PcaPsf.from_psf(gauss_psf, detector_wave, size=detector_s, truncate=5)

    sky = Sky()
    sky.load(SKY_PATH.format(arm), format='h5')

    moon = Moon()
    moon.load(MOON_PATH.format(arm), format='h5')

    obs = PfsObservation()
    obs.detector = detector
    obs.sky = sky
    obs.moon = moon
    obs.noise_model = NormalNoise()

def create_pipeline(arm, grid, filter, obs, noise_level=1.0, noise_freeze=False, calib_bias=False):
    pp = StellarModelPipeline()
    pp.model_res = grid.resolution or 150000
    pp.mag_filter = filter
    pp.observation = obs
    pp.snr = QuantileSnr(binning=4.0)
    pp.resampler = FluxConservingResampler()
    pp.noise_level = noise_level
    pp.noise_freeze = noise_freeze
    if calib_bias:
        bias = FluxCalibrationBias(reuse_bias=False)
        bias.amplitude = 0.25
        pp.calibration = bias
    return pp

def create_rvfit():
    rvfit = RVFit(trace=RVFitTrace())        
    rvfit.template_resampler = FluxConservingResampler()
    return rvfit

def get_observation(arm, grid, obs, rv=0.0, noise_level=1.0, noise_freeze=True, calib_bias=True, mag=19, **kwargs):
    """
    Generate a spectrum and calculate the variance (sigma) of realistic observational error.
    """

    args = {
        'mag': mag,
        'seeing': 0.5,
        'exp_time': 15 * 60,
        'exp_count': 4 * 3,
        'target_zenith_angle': 0,
        'target_field_angle': 0.0,
        'moon_zenith_angle': 45,
        'moon_target_angle': 60,
        'moon_phase': 0.,
        'z': Physics.vel_to_z(rv) 
    }

    idx = grid.get_nearest_index(**kwargs)
    spec = grid.get_model_at(idx)
    pp = create_pipeline(arm, grid, filter, obs, noise_level=noise_level, noise_freeze=noise_freeze, calib_bias=calib_bias)
    pp.run(spec, **args)

    return idx, spec, pp

def get_template(arm, grid, template_psf, convolve=True, **kwargs):
    """
    Generate a noiseless template spectrum with same line spread function as the
    observation but keep the original, high-resolution binning.
    """

    # TODO: add template caching

    idx = grid.get_nearest_index(**kwargs)
    temp = grid.get_model_at(idx)
    temp.cont = None        # Make sure it's not passed around for better performance
    temp.mask = None

    if convolve:
        temp.convolve_psf(template_psf[arm])

    return idx, temp

def plot_psf_sigma(gauss_psf):
    f, ax = plt.subplots(1, 1, figsize=(3.4, 2.5), dpi=240)

    for arm in ARMS:
        ax.plot(gauss_psf[arm].wave, gauss_psf[arm].sigma)

    ax.set_xlabel(r'$\lambda$ [AA]')
    ax.set_ylabel(r'LSF sigma [AA]')

    ax.grid()

    return f

def plot_psf(arm, grid, gauss_psf, pca_psf, template_psf):
    f, ax = plt.subplots(1, 1, figsize=(3.4, 2.5), dpi=240)

    ### Gauss ###

    s = gauss_psf[arm].get_optimal_size(grid.wave)
    idx = np.digitize(5000, grid.wave)

    w = grid.wave[idx - s // 2:idx + s // 2 + 1]
    dw = w - grid.wave[idx]
    k = gauss_psf[arm].eval_kernel_at(grid.wave[idx], dw, normalize=True)
    print('k', k.shape)

    ax.plot(dw, k[0], '.-', ms=0.5, lw=0.5)

    ###########

    idx = np.digitize(5000, template_psf[arm].wave)

    w, dw, kk, _, _ = template_psf[arm].eval_kernel(template_psf[arm].wave)
    print('dw, kk', dw.shape, kk.shape)

    ax.plot(dw[idx], kk[idx], '.--', ms=0.5, lw=0.5)

    ax.set_xlim(-2, 2)
    ax.set_xlabel(r'$\Delta\lambda$')
    ax.set_title(f'Comparison of PCA and Gauss PSF for arm {arm}')

    ax.grid()

    return f

def plot_rvfit_mc(spec, rvfit, rv_gt, rv_fit):
    f, axs = plt.subplots(2, 2, figsize=(3.4, 2.5), dpi=240)

    # Noiseless spectrum

    axs[0, 0].plot(spec.wave, spec.flux_model, '-', lw=0.3)
    axs[0, 0].plot(spec.wave, spec.flux_err, '-', lw=0.3)
    axs[0, 0].set_ylim(0, None)

    # Noisy spectrum

    axs[0, 1].plot(spec.wave, spec.flux, '-', lw=0.3)
    axs[0, 1].plot(spec.wave, spec.flux_model, '-', lw=0.3)

    axs[0, 1].set_xlabel(r'$\lambda$')
    axs[0, 1].set_ylabel(r'$F_\lambda$')

    axs[0, 1].set_xlim(0.99 * spec.wave.min(), 1.01 * spec.wave.max())
    axs[0, 1].set_ylim(0, np.quantile(spec.flux, 0.99) * 1.2)

    # Calibration bias

    if spec.flux_calibration is not None:
        axs[1, 0].plot(spec.wave, spec.flux_calibration, '-', lw=0.3)

    # RV fit

    axs[1, 1].plot(rvfit.trace.guess_rv, rvfit.trace.guess_log_L)
    axs[1, 1].plot(rvfit.trace.guess_rv, rvfit.trace.guess_fit)
    axs[1, 1].axvline(rv, c='k')
    axs[1, 1].axvline(best_rv, c='r')

    f.suptitle(f'mag = {spec.mag:.2f}, SNR = {spec.snr:.2f}')

    f.tight_layout()

# Global declarations required by parallel execution

idx, spectra, pipelines = None, None, None
grid = None
observations = None
rvfit = None

def rvfit_mc_worker(i, rv=180, rv_bounds=(100, 300), calib_bias=False, mag=22, noise_level=1.0,
        M_H=-0.5, T_eff=5500, log_g=1.0, a_M=0.0):

    global idx, spectra, pipelines
    global grid, observations, rvfit
    
    # Sample RV randomly or use a constant value
    if isinstance(rv, Iterable):
        rv_gt = np.random.uniform(*rv)
        print(rv_gt)
        spec = None
    else:
        rv_gt = rv

    if grid is None:
        grid = load_spectrum_grid()

    if observations is None:
        observations = {}
        for arm in FIT_ARMS:
            gauss_psf, _, _ = load_psf(arm, grid)
            obs = create_observation(arm, gauss_psf)
        
    if rvfit is None:
        rvfit = create_rvfit()
    
    if spectra is None:
        spectra = {}
        pipelines = {}

        for arm in FIT_ARMS:
            idx, spectra[arm], pipelines[arm] = get_observation(arm, grid, obs, rv=rv_gt, mag=mag, M_H=M_H, T_eff=T_eff, log_g=log_g, a_M=a_M,
                calib_bias=calib_bias, noise_level=noise_level, noise_freeze=False)

        # Reset initial value for optimization so that it will try to guess it from scratch
        rvfit.rv_0 = None

    # Copy spectrum before generating the noise to keep original to be reused
    # when rv_gt is constant
    nspectra = {}
    for arm in FIT_ARMS:
        nspec = type(spectra[arm])(orig=spectra[arm])
        nspec.flux_model = nspec.flux
        nspec.apply_noise(pipelines[arm].observation.noise_model, noise_level=noise_level)
        nspectra[arm] = nspec

    rv_fit, rv_err = rvfit.fit_rv([nspectra[arm] for arm in FIT_ARMS], [templates[arm] for arm in FIT_ARMS],
        rv_bounds=rv_bounds)

    return i, rv_gt, rv_fit, rv_err

def rvfit_mc(mc_count=100, mag=22, **kwargs):
    
    """
    Fit RV to many realizations of the same observation. Sample in a range of
    magnitudes.
    """

    global idx, spectra, pipelines

    t = tqdm(total=mc_count)

    idx, spec, pp = None, None, None

    rv_gt = np.empty((mc_count,))
    rv_fit = np.empty((mc_count,))
    rv_err = np.empty((mc_count,))
    with SmartParallel(verbose=False, parallel=True, threads=12) as p:
        for i, res_rv_gt, res_rv_fit, res_rv_err in p.map(rvfit_mc_worker, list(range(mc_count)), mag=mag, **kwargs):
            rv_gt[i] = res_rv_gt
            rv_fit[i] = res_rv_fit
            rv_err[i] = res_rv_err
            t.update(1)

    return rv_gt, rv_fit, rv_err

def __main__():
    # grid = load_spectrum_grid()

    # filt_hsc_g = load_filter()

    # gauss_psf = {}
    # pca_psf = {}
    # template_psf = {}
    # for arm in ARMS:
    #     gauss_psf[arm], pca_psf[arm], template_psf[arm] = load_psf(arm, grid)

    # plot_psf_sigma(gauss_psf).savefig('psf_sigma.png')
    # for i, arm in enumerate(ARMS):
    #     plot_psf(arm, grid, gauss_psf, pca_psf, template_psf).savefig(f'psf_{arm}.png')

    # obs = {}
    # for arm in ARMS:
    #     obs[arm] = create_observation(arm, gauss_psf[arm])

    rv = 180.0
    M_H = -1.5
    T_eff = 4000
    log_g = 1.0
    a_M = 0.0

    
    N = 1000

    rv_gt = {}
    rv_fit = {}
    rv_err = {}

    for m in [19, 20, 21, 22, 23]:
        rv_gt[m] = {}
        rv_fit[m]= {}
        rv_err[m]= {}
        for cb in [False]:    
            rv_gt[m][cb], rv_fit[m][cb], rv_err[m][cb] = rvfit_mc(mc_count=N, rv=180, 
                mag=m, 
                calib_bias=cb)

__main__()