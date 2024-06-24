import numpy as np
from collections.abc import Iterable

from pfs.ga.pfsspec.core import PfsObject
from pfs.ga.pfsspec.core import Physics

class ContinuumObject(PfsObject):

    def __init__(self, orig=None):

        if not isinstance(orig, ContinuumObject):
            self.included_ranges = None
            self.included_mask = None
            self.included_overflow = None
            self.excluded_ranges = None
            self.excluded_mask = None
            self.excluded_overflow = None

            self.wave = None                                                # Cache for wave, when fitting an entire grid
        else:
            self.included_ranges = orig.included_ranges
            self.included_mask = orig.included_mask
            self.included_overflow = orig.included_overflow
            self.excluded_ranges = orig.excluded_ranges
            self.excluded_mask = orig.excluded_mask
            self.excluded_overflow = orig.excluded_overflow

            self.wave = orig.wave

    #region Functions for handling the wavelength grid, wavelength limits and masks
    
    def init_wave(self, wave, force=True, omit_overflow=False):
        """
        Initialize the wave vector cache and masks in derived classes.
        """

        if force or self.wave is None:
            self.wave = wave

        if self.included_ranges is not None and (force or self.included_mask is None):
            self.included_mask, self.included_overflow = self.ranges_to_mask(
                wave, self.included_ranges,
                omit_overflow=omit_overflow)

        if self.excluded_ranges is not None and (force or self.excluded_mask is None):
            self.excluded_mask, self.excluded_overflow = self.ranges_to_mask(
                wave, self.excluded_ranges,
                omit_overflow=omit_overflow)

    def get_hydrogen_limits(self):
        limits = [2530,] + Physics.HYDROGEN_LIMITS + [17500,]
        return Physics.air_to_vac(np.array(limits))

    def limits_to_ranges(self, limits):
        """
        Convert a list of limits to a list of ranges.
        """

        ranges = []
        for i in range(len(limits) - 1):
            ranges.append([limits[i], limits[i + 1]])
        return ranges
    
    @staticmethod
    def lessthan(a, b, strict=False):
        if strict:
            return a < b
        else:
            return a <= b
    
    def ranges_to_mask(self, wave, ranges, mask=None, dlambda=0, strict=False, omit_overflow=False):
        """
        Convert a list of wavelength ranges to a mask.

        Assume that the wave vector is sorted.

        Parameters
        ----------
        wave : array
            Wavelength vector.
        ranges : list of tuples
            List of wavelength ranges to convert to a mask.
        mask : array
            Optional input mask.
        dlambda : float or tuple
            Buffer around the limits of the ranges.
        strict : bool
            Use strict comparison.
        omit_overflow : bool
            Exclude ranges that overflow the wavelength vector.

        Returns
        -------
        mask : array
            Mask for the wavelength vector.
        overflow : list of bool
            List of indices of ranges that overflow the wavelength vector.
        """

        if not isinstance(ranges, Iterable):
            ranges = [ ranges ]

        if not isinstance(dlambda, Iterable):
            dlambda = ( dlambda, dlambda )

        # Construct a mask by merging all limits with a buffer of `dlambda`
        m = np.full(wave.shape, False)
        overflow = []
        for i, r in enumerate(ranges):
            # Handle overflow
            if r[0] + dlambda[0] < wave[0] or r[1] - dlambda[1] > wave[-1]:
                overflow.append(i)
                if omit_overflow:
                    continue

            # Generate the mask
            m |= self.lessthan(r[0] + dlambda[0], wave, strict) & \
                 self.lessthan(wave, r[1] - dlambda[1], strict)
            
        # Combine with optional input mask
        if mask is not None:
            m &= mask

        return m, overflow
    
    def limits_to_masks(self, wave, limits, mask=None, dlambda=0, strict=False, omit_overflow=False):
        """
        Convert a list of wavelengths into a list of masks in-between the limits.

        Assume that the wavelength vector is sorted.
        """

        if not isinstance(dlambda, Iterable):
            dlambda = ( dlambda, dlambda )

        # Find intervals between the limits
        masks = []
        ranges = []
        overflow = []
        for i in range(len(limits) - 1):
            l0, l1 = limits[i], limits[i + 1]
            
            l0 = l0 if l0 is not None else wave[0]
            l1 = l1 if l1 is not None else wave[-1]
            
            # Handle overflow
            if l0 + dlambda[0] < wave[0] or l1 - dlambda[1] > wave[-1]:
                overflow.append(i)
                if omit_overflow:
                    continue

            m = self.lessthan(l0 + dlambda[0], wave, strict) & \
                self.lessthan(wave, l1 - dlambda[1], strict)

            # Combine with optional input mask
            if mask is not None:
                m &= mask

            masks.append(m)

            # Find the actual range after applying the mask
            w = wave[m]
            if w.size > 0:
                ranges.append([w[0], w[-1]])
            else:
                ranges.append([np.nan, np.nan])

        return masks, ranges, overflow

    #endregion

    def get_full_mask(self, mask):
        mask = mask.copy() if mask is not None else np.full(self.wave.shape, True)
        
        if self.included_mask is not None:
            mask &= self.included_mask
        
        if self.excluded_mask is not None:
            mask &= ~self.excluded_mask

        return mask