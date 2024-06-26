import copy
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d

try:
    import alphashape
except:
    alphashape = None

from pfs.ga.pfsspec.core.setup_logger import logger
from pfs.ga.pfsspec.core import Physics
from pfs.ga.pfsspec.core.util.array_filters import *
from .continuummodel import ContinuumModel
from .continuummodeltrace import ContinuumModelTrace
from .modelparameter import ModelParameter
from ..functions import AlexSigmoid, Legendre
from ..finders import SigmaClipping

class AlexContinuumModelTrace(ContinuumModelTrace):
    def __init__(self):
        super().__init__()

        self.model_cont = None
        self.model_blended = None
        self.norm_flux = None
        self.norm_cont = None

        self.legendre_control_points = {}
        self.blended_control_points = {}
        self.blended_p0 = {}
        self.blended_params = {}
        self.blended_chi2 = {}
        self.blended_fit = {}
        self.x1 = {}

class Alex(ContinuumModel):
    # Fit the upper envelope of a stellar spectrum model. The theoretical continuum
    # is first fitted with Lengendre polinomials between the Hzdrogen photoionization
    # limits, then the model is normalized and the remaining blended line regions
    # are fitted with a modified sigmoid function to remove all non-linearities
    # from the continuum.

    def __init__(self, orig=None, trace=None):
        super().__init__(orig=orig)

        # Trace certain variables for debugging purposes
        self.trace = trace

        if isinstance(orig, Alex):
            raise NotImplementedError()
        else:
            # Global wavelength limits that we can fit
            self.wave_min = 3000
            self.wave_max = 14000

            # The wave vector is assumed to be constant for all spectra and cached
            # TODO: this is not always the case, sometimes the spectra that need to be
            #       denormalized are sliced along the wavelength axis.
            self.log_wave = None

            # Masks of continuum intervals
            self.cont_fit_rate_multiplier = np.array([1, 3, 2])
            self.cont_fit_rate = None       # How many points to skip when fitting Legendre to continuum            
            self.cont_fit_masks = None      # Continuum intervals for fitting
            self.cont_eval_masks = None     # Intervals to evaluate continuum on
            self.cont_models = None

            # Parameters of continuum Legendre fits
            self.legendre_deg = 6
            self.legendre_slope_cutoff = 25  # Cutoff to filter out very steep intial part of continuum regions
            self.legendre_dx_multiplier = np.array([1, 1, 1])
            self.legendre_dx = None

            # Bounds and masks of blended regions near photoionization limits

            self.limit_wave = None                      # Limit wavelengths, including model lower and upper bounds
            self.limit_map = None                       # Map to continuum intervals the blended regions are associated with
            
            # Blended region upper limits
            # self.blended_bounds = np.array([3200.0, 5000, 12000])
            # self.blended_bounds = np.array([3400.0, 8000, 13000])
            self.blended_bounds = np.array([3200.0, 4500, 9200])
            self.blended_count = self.blended_bounds.size
            
            self.blended_fit_masks = None                 # Masks where limits are fitted
            self.blended_eval_masks = None                # Masks where limits are evaluated
            self.blended_models = None

            self.blended_dx_multiplier = np.array([1, 2, 1])
            self.blended_dx = None

            self.blended_slope_cutoff = 25  # Cutoff to filter out very steep intial part of blended regions

            # Smoothing parameters of blended region upper envelope fits
            # TODO: these should go to a superclass
            self.smoothing_iter = 5
            self.smoothing_option = 1
            self.smoothing_kappa = 50
            self.smoothing_gamma = 0.1

    @property
    def name(self):
        return "alex"

    def add_args(self, parser):
        super(Alex, self).add_args(parser)

        # TODO: these should go to a superclass
        parser.add_argument('--smoothing-iter', type=int, help='Smoothing iterations.\n')
        parser.add_argument('--smoothing-option', type=int, help='Smoothing kernel function.\n')
        parser.add_argument('--smoothing-kappa', type=float, help='Smoothing kappa.\n')
        parser.add_argument('--smoothing-gamma', type=float, help='Smoothing gamma.\n')

        parser.add_argument('--alex-legendre-deg', type=int, help='Max degree of Legendre polynomials.\n')

    def init_from_args(self, args):
        super(Alex, self).init_from_args(args)

        # TODO: these should go to a superclass
        if 'smoothing_iter' in args and args['smoothing_iter'] is not None:
            self.smoothing_iter = args['smoothing_iter']
        if 'smoothing_option' in args and args['smoothing_option'] is not None:
            self.smoothing_option = args['smoothing_option']
        if 'smoothing_kappa' in args and args['smoothing_kappa'] is not None:
            self.smoothing_kappa = args['smoothing_kappa']
        if 'smoothing_gamma' in args and args['smoothing_gamma'] is not None:
            self.smoothing_gamma = args['smoothing_gamma']

        self.legendre_deg = self.get_arg('alex_legendre_deg', self.legendre_deg, args)

    def get_constants(self, wave=None):
        self.find_limits(wave)

        constants = {}

        constants['legendre_deg'] = self.legendre_deg
        constants['blended_bounds'] = self.blended_bounds
        constants['limit_wave'] = self.limit_wave

        return constants

    def set_constants(self, constants, wave=None):
        self.find_limits(wave)

        self.legendre_deg = self.set_constant('legendre_deg', constants, self.legendre_deg)
        self.blended_bounds = self.set_constant('blended_bounds', constants, self.blended_bounds)
        self.limit_wave = self.set_constant('limit_wave', constants, self.limit_wave)

    def save_items(self):
        self.save_item('/'.join((self.PREFIX_CONTINUUM_MODEL, 'legendre_deg')), self.legendre_deg)
        self.save_item('/'.join((self.PREFIX_CONTINUUM_MODEL, 'blended_bounds')), self.blended_bounds)
        self.save_item('/'.join((self.PREFIX_CONTINUUM_MODEL, 'limit_wave')), self.limit_wave)

    def load_items(self):
        self.legendre_deg = self.load_item('/'.join((self.PREFIX_CONTINUUM_MODEL, 'legendre_deg')), int, default=self.legendre_deg)
        self.blended_bounds = self.load_item('/'.join((self.PREFIX_CONTINUUM_MODEL, 'blended_bounds')), np.ndarray, default=self.blended_bounds)
        self.limit_wave = self.load_item('/'.join((self.PREFIX_CONTINUUM_MODEL, 'limit_wave')), np.ndarray, default=self.limit_wave)

    def get_interpolated_params(self):
        """
        Return parameters that will be interpolated across gridpoints
        """

        params = super(Alex, self).get_interpolated_params()
        params.append(ModelParameter(name='legendre',
                rbf_method='solve',
                rbf_function='multiquadric',
                rbf_epsilon=1.0))
        for i, _ in enumerate(self.blended_models):
            params.append(ModelParameter(
                name='blended_' + str(i), 
                rbf_method='solve',
                rbf_function='multiquadric',
                rbf_epsilon=1.0))
        return params

    def init_wave(self, wave):
        self.find_limits(wave)

    def allocate_values(self, grid):
        k = 0
        for i, m in enumerate(self.cont_models):
            k += m.get_param_count()
        grid.allocate_value('legendre', (k,))

        for i, m in enumerate(self.blended_models):
            k = m.get_param_count()
            grid.allocate_value('blended_' + str(i), (k,))

#region Main entrypoints: fit, eval and normalize

    def fit_spectrum(self, spec):
        params = {}

        # Fit the spectrum and return the parameters
        log_flux = self.safe_log(spec.flux[self.wave_mask])
        if spec.cont is not None:
            log_cont = self.safe_log(spec.cont[self.wave_mask])
        else:
            log_cont = None
        
        # Fit continuum and normalize spectrum to fit blended lines as a next step
        try:
            cont_params = self.fit_continuum_all(log_flux, log_cont)
            params.update(cont_params)
            model_cont = self.eval_continuum_all(cont_params)
        except Exception as e:
            raise e
        norm_flux = log_flux - model_cont

        if self.trace is not None:
            self.trace.norm_flux = norm_flux
            
        # Fit blended lines of the photoionization limits
        try:
            limit_params = self.fit_blended_all(norm_flux)
            params.update(limit_params)
        except Exception as e:
            raise e

        return params

    def eval(self, params):
        # Evaluate the continuum model over the wave grid

        model_cont = self.eval_continuum_all(params)
        model_cont += self.eval_blended_all(params)
        model_cont = np.exp(model_cont)

        return self.wave, model_cont

    def normalize(self, spec, params):
        self.normalize_use_flux(spec, params)
       
    def normalize_use_log_flux(self, spec, params):
        # Takes log of the flux at the end

        # Evaluate the model
        model_cont = self.eval_continuum_all(params=params)
        model_blended = self.eval_blended_all(params)
        
        # Theoretical continuum, if available
        if spec.cont is not None:
            norm_cont = self.safe_log(spec.cont[self.wave_mask]) - model_cont
        else:
            norm_cont = None

        # Flux normalized with continuum and blended regions
        cont_norm_flux = self.safe_log(spec.flux[self.wave_mask]) - model_cont
        norm_flux = cont_norm_flux - model_blended

        if spec.flux_err is not None:
            # TODO: verify and test
            raise NotImplementedError()
            norm_flux_err = self.safe_log_error(spec.flux[self.wave_mask], spec.flux_err[self.wave_mask])
        else:
            norm_flux_err = None

        if self.trace is not None:
            self.trace.cont_norm_flux = cont_norm_flux
            self.trace.model_cont = model_cont
            self.trace.model_blended = model_blended
            self.trace.norm_flux = norm_flux
            self.trace.norm_cont = norm_cont
            
        spec.wave = self.wave
        spec.cont = norm_cont
        spec.flux = norm_flux
        spec.flux = norm_flux_err

        spec.append_history(f'Spectrum is normalized using model `{type(self).__name__}`.')

    def normalize_use_flux(self, spec, params):
        # Do not take the log of flux at the end
        # Models are always fitted to the log flux so take exp

        # Evaluate the model
        model_cont = self.safe_exp(self.eval_continuum_all(params=params))
        model_blended = self.safe_exp(self.eval_blended_all(params))

        # Theoretical continuum, if available
        if spec.cont is not None:
            norm_cont = spec.cont[self.wave_mask] / model_cont
        else:
            norm_cont = None

        # Flux normalized with continuum and blended regions
        cont_norm_flux = spec.flux[self.wave_mask] / model_cont
        norm_flux = cont_norm_flux / model_blended
        if spec.flux_err is not None:
            norm_flux_err = spec.flux_err / model_cont
        else:
            norm_flux_err = None

        if self.trace is not None:
            self.trace.cont_norm_flux = cont_norm_flux
            self.trace.model_cont = model_cont
            self.trace.model_blended = model_blended
            self.trace.norm_flux = norm_flux
            self.trace.norm_cont = norm_cont

        spec.wave = self.wave
        spec.cont = norm_cont
        spec.flux = norm_flux
        spec.flux_err = norm_flux_err

        spec.append_history(f'Spectrum is normalized using model `{type(self).__name__}`.')

    def denormalize(self, spec, params, s=None):
        self.denormalize_use_flux(spec, params, s=s)

    def denormalize_use_flux(self, spec, params, s=None):
        # Denormalize the spectrum given the fit params
        # Expects normalized flux (no log)
        model_cont = self.eval_continuum_all(params=params)
        model_cont = model_cont[s or ()]
        
        if spec.cont is not None:
            cont = spec.cont * self.safe_exp(model_cont)
        else:
            cont = self.safe_exp(model_cont)

        model_blended = self.eval_blended_all(params)
        model_blended = model_blended[s or ()]

        if self.trace is not None:
            self.trace.model_cont = model_cont
            self.trace.model_blended = model_blended
            self.trace.norm_flux = spec.flux
            self.trace.norm_cont = spec.cont

        # TODO: pass the model to the spectrum class instead because
        #       we don't know the vectors that need to be multiplied here
        model_full = self.safe_exp(model_cont + model_blended)
        spec.flux *= model_full
        if spec.flux_err is not None:
            spec.flux_err *= model_full

        # TODO: what if we know the theoretical continuum?
        spec.cont = cont

        spec.append_history(f'Spectrum is denormalized using model `{type(self).__name__}`.')

    def denormalize_use_log_flux(self, spec, params, s=None):
        # Denormalize the spectrum given the fit params
        # Expects normalized log flux

        model_cont = self.eval_continuum_all(params=params)
        model_cont = model_cont[s or ()]
        
        if spec.cont is not None:
            cont = self.safe_exp(spec.cont + model_cont)
        else:
            cont = self.safe_exp(model_cont)

        model_blended = self.eval_blended_all(params)
        model_blended = model_blended[s or ()]

        if self.trace is not None:
            self.trace.model_cont = model_cont
            self.trace.model_blended = model_blended
            self.trace.norm_flux = spec.flux
            self.trace.norm_cont = spec.cont

        # TODO: pass the model to the spectrum class instead because
        #       we don't know the vectors that need to be multiplied here
        flux = self.safe_exp(spec.flux + model_cont + model_blended)

        if spec.flux_err is not None:
            # TODO: verify and test
            raise NotImplementedError()
            flux_err = self.safe_exp_error(spec.flux, spec.flux_err)
        else:
            flux_err = None

        spec.flux = flux
        spec.flux_err = flux_err
        spec.cont = cont

        spec.append_history(f'Spectrum is denormalized using model `{type(self).__name__}`.')

    def fill_model_params_grid(self, name, params):
        # Fill in the holes in a parameter grid

        if name == 'legendre':
            return params
        else:
            return super().fill_model_params_grid(name, params)

    def smooth_model_params_grid(self, name, params):
        # Smooth the parameters.
        if name == 'legendre':
            return params
        else:
            return super().smooth_model_params_grid(name, params)

#endregion            
#region Limits and mask

    def get_limit_wavelengths(self):
        # Returns the wavelength associated with the Hydrogen ionization limits
        # and the global limits of what we can fit
        limits = np.array(Physics.HYDROGEN_LIMITS)
        limits = limits[(self.wave_min < limits) & (limits < self.wave_max)]
        limits = np.array([self.wave_min, *Physics.air_to_vac(limits), self.wave_max])
        return limits

    def find_limits(self, wave):
        # Exact wavelengths of the limits and the lower and upper bounds of what we can fit
        self.limit_wave = self.get_limit_wavelengths()

        # Mask that defines the entire range we can fit. Precalculate masked wavelength
        # grid and log lambda grid for convenience.
        [self.wave_mask], _, _ = self.limits_to_masks(wave, [self.limit_wave[0], self.limit_wave[-1]], dlambda=0)
        self.wave = wave[self.wave_mask]
        self.log_wave = np.log(self.wave)

        # Every mask below will refer to the grid defined by wave_mask

        # Masks that define the regions where we fit the continuum
        self.cont_fit_masks, _, _ = self.limits_to_masks(self.wave, self.limit_wave, dlambda=0.5)
        # Disjoint masks that define where we evaluate the continuum, no gaps here
        self.cont_eval_masks, _, _ = self.limits_to_masks(self.wave, self.limit_wave, dlambda=0.0)
        # Continuum models
        self.cont_models = []
        for i in range(len(self.cont_fit_masks)):
            w0 = np.log(self.limit_wave[i])
            w1 = np.log(self.limit_wave[i + 1])
            m = Legendre(self.legendre_deg, domain=[w0, w1])
            self.cont_models.append(m)

        # Masks where we will fit the blended lines' upper envelope. These are a
        # little bit redward from the photoionization limit.
        self.blended_fit_masks, self.limit_map = self.find_blended_masks(self.wave, self.cont_fit_masks)
        # Masks where we should evaluate the blended lines's upper envelope.
        self.blended_eval_masks, _ = self.find_blended_masks(self.wave, self.cont_eval_masks)
        # Blended region models
        self.blended_models = []
        for i in range(len(self.blended_fit_masks)):
            # amplitude, slope, midpoint, inflexion points s0, s1
            bounds = ([0.001, 0., np.log(self.limit_wave[self.limit_map[i]]), 0., 0.], \
                      [10., 1000, np.log(self.blended_bounds[i]), 1., 1.])
            m = AlexSigmoid(bounds=bounds)
            self.blended_models.append(m)

        # Determine the step size quantum for certain operations
        # TODO: what is this exactly?
        wl = max(3000.0, self.wave.min())
        dwl = 6.0
        mask = (self.wave > wl) & (self.wave < wl + dwl) 
        dx = int(mask.sum())
        self.blended_dx = self.blended_dx_multiplier * dx
        self.legendre_dx = self.legendre_dx_multiplier * dx
        
        # Downsampling of the wavelength grid for fitting the continuum
        self.cont_fit_rate = self.cont_fit_rate_multiplier * dx

    def find_blended_masks(self, wave, cont_masks):
        blended_masks = []
        limit_map = []
        for i in range(len(self.blended_bounds)):
            # Find the continuum region the limit is associated with
            for j in range(len(self.limit_wave)):
                if j == len(self.limit_wave) - 1:
                    limit_map.append(None)
                    break
                if self.limit_wave[j] < self.blended_bounds[i] and self.blended_bounds[i] < self.limit_wave[j + 1]:
                    limit_map.append(j)
                    break

            if limit_map[i] is not None:
                mask = cont_masks[limit_map[i]]
                blended_masks.append(mask & (wave < self.blended_bounds[i]))
            else:
                blended_masks.append(None)

        return blended_masks, limit_map

#endregion
#region Blended region fitting

    def fit_blended_all(self, norm_flux):
        params = {}
        for i in range(len(self.limit_map)):
            pp = self.fit_blended(norm_flux, i)
            params['blended_' + str(i)] = pp
        return params
        
    def eval_blended_all(self, params):
        # Evaluate model around the limits
        model = np.zeros_like(self.wave)
        for i in range(len(self.limit_map)):
            p = params['blended_' + str(i)]
            flux, mask = self.eval_blended(p, i)
            model[mask] += flux
        return model

    def eval_blended(self, params, i):
        mask = self.blended_eval_masks[i]
        model = self.blended_models[i]
        if np.any(np.isnan(params) | np.isinf(params)) or abs(params).sum() == 0:
            return np.zeros_like(self.log_wave[mask]), mask
        else:
            flux = model.eval(self.log_wave[mask], params)
            return flux, mask

    def fit_blended(self, norm_flux, i):
        # Fit a blended line region

        model = self.blended_models[i]

        # Try to fit and handle gracefully if fails
        try:
            # Get control points using the maximum hull method
            x, y = self.get_blended_control_points(norm_flux, i)

            # Check if control points are good enough for a fit
            if y.size <= 5:
                raise Exception('No valid control points.')

            if self.trace is not None:
                self.trace.blended_control_points[i] = (x, y)

            # Estimate the initial value of the parameters
            good, p0 = model.find_p0(x, y)
            if self.trace is not None:
                self.trace.blended_p0[i] = p0
        
            if good:
                pp = model.fit(x, y, w=None, p0=p0)
            else:
                pp = p0

            if self.trace is not None:
                self.trace.blended_fit[i] = True
                self.trace.blended_params[i] = pp
                self.trace.blended_chi2[i] = np.sum((y - model.eval(x, pp))**2)

            return pp
        except Exception as ex:
            # logger.warning(ex)
            if self.trace is not None:
                self.trace.blended_fit[i] = False
            return np.array(model.get_param_count() * [np.nan])

    def get_blended_control_points(self, norm_flux, i):
        # Find control points for fitting a modified sigmoid function
        # to a blended line region redward of the photoionization limits.

        # Make sure number of fitted parameters is correct and in the right range.
        def validate_control_points(y):
            return len(y) > 3 and y[0] < -0.001

        mask = self.blended_fit_masks[i]
        dx = self.blended_dx[i]

        x, y = self.log_wave[mask], norm_flux[mask]

        # Find the maximum in intervals of dx and determine the maximum hull
        #x, y = self.get_max_interval(x, y, dx=dx)
        x, y = self.get_max_hull(x, y)
          
        # Calculate the differential and drop the very steep part at the
        # beginning of the interval, as it may be a narrow line instead of a
        # blended region
        x, y = self.get_slope_filtered(x, y, cutoff=self.blended_slope_cutoff)

        return x, y

    def get_slope_filtered(self, x, y, cutoff=0):
        """
        Filter the points of the (x, y) curve so that the numerical differential
        is never larger than cutoff.
        """

        def get_min_max_norm(v):
            vmin, vmax = np.min(v), np.max(v)
            return (v - vmin) / (vmax - vmin)

        # Normalize both vectors to be between 0 and 1
        xx = get_min_max_norm(x)
        yy = get_min_max_norm(y)

        dd = np.diff(yy) / np.diff(xx)
        dd = np.abs(np.append(dd, dd[-1]))
        dd_median, dd_std = np.median(dd), dd.std()
        dd_high = dd_median + dd_std * 3.0
        slope_cut = np.min([dd_high, cutoff])
        mask = (dd < slope_cut)
        return x[mask], y[mask]
    
    def get_max_hull(self, x, y):
        # Get the maximum hull
        y_accumulated = np.maximum.accumulate(y)
        mask = (y >= y_accumulated)
        return x[mask], y[mask]

    def get_convex_hull(self, x, y):
        # Determine the upper envelope from the convex hull of points

        # Normalize data and make sure there's enough curvature so that
        # the uppen envelope of the spectrum is convex
        x_min, x_max = x.min(), x.max()
        cx = (x - x_min) / (x_max - x_min)
        
        cy = y * np.sqrt(1 - (0.75 * cx)**2)      # make convex
        y_min, y_max = cy.min(), cy.max()
        cy = (cy - y_min) / (y_max - y_min)

        # Append two extreme low points at the beginning and the end
        # to be able to determine the points of the upper envelope
        points = np.stack([cx, cy], axis=-1)
        points = np.concatenate([np.array([[cx[0], -0.1]]), points, np.array([[cx[-1], -0.1]])])
        
        h = ConvexHull(points)

        # Filter out extra points        
        mask = (h.vertices > 0) & (h.vertices < points.shape[0] - 1)
        ix = h.vertices[mask] - 1

        # Sort by wavelength
        sorting = np.argsort(x[ix])
        ix = ix[sorting]

        return x[ix], y[ix]

    def get_alpha_shape(self, x, y):
        if alphashape is None:
            raise Exception("Optional package alphashape is not available.")
        
        # Determine the upper envelope from the alpha shape of points
        
        # Append two extreme low points at the beginning and the end
        # to be able to determine the points of the upper envelope
        y_min = np.floor(y.min() - 1)
        points = np.stack([x, y], axis=-1)
        points = np.concatenate([np.array([[x[0], y_min]]), points, np.array([[x[-1], y_min]])])

        a = alphashape.alphashape(points, 0.2)

        x = np.array(a.boundary.coords.xy[0])
        y = np.array(a.boundary.coords.xy[1])
        mask = y > y_min

        return x[mask], y[mask]

    def get_max_interval(self, x, y, dx=500):
        # Get the maximum in every interval of dx

        N = x.shape[0]
        pad_row = int(np.floor(N / dx)) + 1 
        pad_num = pad_row * dx - N
        pad_val = np.min(y) - 1

        x_reshaped = np.pad(x, (0, pad_num), constant_values=pad_val).reshape(pad_row, dx)
        y_reshaped = np.pad(y, (0, pad_num), constant_values=pad_val).reshape(pad_row, dx)

        max_idx = np.argmax(y_reshaped, axis = 1)
        max_x = np.take_along_axis(x_reshaped, max_idx[..., np.newaxis], axis=1)[:, 0]
        max_y = np.take_along_axis(y_reshaped, max_idx[..., np.newaxis], axis=1)[:, 0]

        return max_x, max_y

#endregion
#region Modified sigmoid fitting to blended regions


#endregion
#region Continuum fitting with Legendre polynomials

    def get_cont_params(self, params, i):
        # Count the number of parameters before i
        l, u = 0, 0
        for k in range(i + 1):
            c = self.cont_models[i].get_param_count()
            if k < i:
                l += c
            u += c
        p = params['legendre'][l:u]
        return p

    def fit_continuum_all(self, log_flux, log_cont):
        params = []
        for i in range(len(self.cont_models)):
            p = self.fit_continuum(log_flux, log_cont, i)
            params.append(p)
        return { 'legendre': np.concatenate(params)}

    def get_continuum_contol_points(self):
        pass

    def fit_continuum(self, log_flux, log_cont, i):
        # Fit the smooth part (continuum) of the upper envelope. Use the
        # theoretical model, if available, or fit Legendre polynomials to
        # the uppen part of the convex hull of the log-flux.

        mask = self.cont_fit_masks[i]
        model = self.cont_models[i]

        # TODO: use get_continuum_contol_points

        if log_cont is not None:
            x = self.log_wave[mask]
            y = log_cont[mask]
            w = None
            max_deg = None
        else:
            x = self.log_wave[mask]
            y = log_flux[mask]

            # Find peaks
            ix, _ = find_peaks(y, distance=100)
            x, y = x[ix], y[ix]

            # Find convex hull of peaks
            x, y = self.get_convex_hull(x, y)

            # - find maximum in intervals,
            # - filter for extreme values of the slope to avoid steep edges
            # - determine convex hull of upper envelope

            #x, y = self.get_max_interval(x, y, dx=self.legendre_dx[i])
            
            x, y = self.get_slope_filtered(x, y, cutoff=self.legendre_slope_cutoff)
            
            # x, y = self.get_alpha_shape(x, y)
            # x, y = self.get_slope_filtered(x, y, cutoff=self.legendre_slope_cutoff)

            # Continuum control points must be within the interval of the Hydrogen ionization limits
            x_min = np.log(self.blended_bounds[i])
            # x_min = np.log(self.limit_wave[i])
            x_max = np.log(self.limit_wave[i + 1])

            cpmask = (x >= x_min) & (x <= x_max)
            cpx = x[cpmask]
            cpy = y[cpmask]

            ip = interp1d(cpx, cpy, fill_value='extrapolate')

            max_deg = cpx.shape[0]
            x, y, w = x, ip(x), None

            if False:
                # Add endpoints interpolated to the convex hull of the upper envelope
                cpx = np.concatenate([[x_min], cpx, [x_max]])
                cpy = np.concatenate([[ip(x_min)], cpy, [ip(x_max)]])
                
            if True:
                # The convex hull has too few points so generate more by interpolating
                # the straight lines between vertices
                x_min = np.log(self.limit_wave[i])

                mask &= (x_min <= self.log_wave) & (self.log_wave <= x_max)

                cpx = self.log_wave[mask][::int(self.legendre_dx[i])]
                cpy = ip(cpx)

                # Assign lower weights to in-between points
                w = np.full_like(cpy, 0.1)
                w[np.digitize(x, cpx) - 1] = 1.0

                x, y = cpx, cpy

        if self.trace is not None:
            self.trace.legendre_control_points[i] = (x, y, w)

        continuum_finder = SigmaClipping(sigma=[2, 2], max_iter=self.max_iter)
        
        success, params = self.fit_function(0, model, x, y, continuum_finder=continuum_finder)
        
        # Find the minimum difference between the model fitted to the continuum
        # and the actual flux and shift the model to avoid big jumps.
        v = model.eval(x, params)
        if np.any(v > y):
            offset = np.min((v - y)[v > y])
            if offset > 1e-2:
                model.shift(-offset, params)

        return params
        
    def eval_continuum_all(self, params):
        # Evaluate the fitted continuum model (Legendre polynomials) over the
        # wavelength grid.
        model_cont = np.zeros_like(self.log_wave)
        for i in range(len(self.cont_models)):
            cont, mask = self.eval_continuum(params, i)
            model_cont[mask] = cont
        return model_cont

    def eval_continuum(self, params, i):
        pp = self.get_cont_params(params, i)
        mask = self.cont_eval_masks[i]
        wave = self.log_wave[mask]
        cont = self.cont_models[i].eval(wave, pp)
        return cont, mask

#endregion