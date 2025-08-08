#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
from logging import getLogger

import fabio
import numpy as np
from fabio.brukerimage import BrukerImage
from numpy import sin, cos, tan, arccos
from scipy.interpolate import RegularGridInterpolator

logger = getLogger(__name__)

DETTYPE = {
    # DETECTORNAME: (PIXPERCM, CMTOGRID)
    'MULTIWIRE': (47.5, 2.0),
    'CCD-PXL-2K': (56.02, 0.8),
    'CCD-PXL-ARR': (32.0, 0.4),
    'CCD-PXL-KAF1500': (51.2, 0.8),
    'CCD-PXL-L6000N': (56.02, 0.8),
    'CCD-PXL-L6000F': (56.02, 0.8),
    'CCD-PXL-L6500': (32.00, 1.5),
    'CCD-PXL-KAF2': (81.92, 0.8),
    'CCD-PXL-KAF': (81.92, 0.8),
    'CCD-PXL-MSPD': (81.92, 0.8),
    'CCD-PXL': (81.92, 0.8),
    'CCD-PXL-LDI': (83.333, 0.0),
}

if (b'FORMAT :100', 'bruker100') not in fabio.openimage.MAGIC_NUMBERS:
    fabio.openimage.MAGIC_NUMBERS.append((b'FORMAT :100', 'bruker100'))


class AreaDetectorImage(object):
    def __init__(self, image=None):
        """
        :param image: filename, file-like, or fabio image object
        """
        if image is None:
            image = fabio.fabioimage.FabioImage()
        elif not isinstance(image, fabio.fabioimage.FabioImage):
            image = fabio.open(image)
        if image.data is not None:
            if np.issubdtype(image.data.dtype, np.integer) and np.min(image.data) >= 0:
                image.data = image.data.astype(np.min_scalar_type(np.max(image.data)))
        self.image = image
        self.alpha = np.nan  # 2-theta value at the center of the detector (unit: radian)
        self.chi = np.nan  # Chi angle (unit: radian)
        self.distance = np.nan  # Distance from sample to detector plane (unit: cm)
        self.densityXY = (np.nan, np.nan)  # Number of pixels per unit length (cm)
        self.centerXY = (np.nan, np.nan)  # x, y coordinates of the detector center (unit: pixels).
        self.scale = 1  # linear scale factor
        self.offset = 0  # linear offset
        self.limits = (np.nan, np.nan, np.nan, np.nan)  # min2θ, max2θ, minγ, maxγ
        self.data_converted = np.ndarray((0, 0), dtype=int)
        self.indexes = (np.arange(0), np.arange(0))
        self.goniometer_pos = (np.nan, np.nan, np.nan, np.nan)  # x, y, z, aux (unit: mm)
        self.load_headers()

    def xy_to_angles(self, x, y):
        """convert from (x, y) coordinates to (twoth, gamma).
         (x, y) = (0, 0) corresponds to the detector center, and they should be given in cm.
         
        Reference
        B.B. He, Two-Dimensional X-Ray Diffraction (Wiley, 2011). 
        2.3.4 Pixel Position in Diffraction Space -- Flat Detector

        :param x: x coordinate in cm.
        :param y: y coordinate in cm
        :return: (twoth, gamma) in rad.
        """
        alpha = self.alpha
        D = self.distance

        twoth = arccos((x*sin(alpha) + D*cos(alpha))/np.sqrt(D**2 + x**2 + y**2))

        det = x*cos(alpha) - D*sin(alpha)
        sign = ((det >= 0) - 0.5)/0.5
        gamma = sign * arccos(-y / np.sqrt(y**2 + det**2))

        return twoth, gamma

    def rowcol_to_angles(self, row, col, detector_shape=None):
        # row ↔ y, col ↔ x
        dX, dY = self.densityXY
        cX, cY = self.centerXY
        if detector_shape is not None:
            print("Warning: No image data available, using detector_shape for dimensions.")
            nY, nX = self.detector_shape
        else:
            nX, nY = self.image.dim1, self.image.dim2
        x, y = (col - cX)/dX, -(row-(nY-cY))/dY
        return self.xy_to_angles(x, y)

    def angles_to_rowcol(self, twoth, gamma):
        """
        convert from (twoth, gamma) to (row, col).
        angles are given in rad.
                
        Reference
        B.B. He, Two-Dimensional X-Ray Diffraction (Wiley, 2011). 
        2.3.4 Pixel Position in Diffraction Space -- Flat Detector

        """
        # Countermeasure for divergence of tan(2θ) at 2θ=90°
        if np.isscalar(twoth):
            if np.isclose(twoth, np.pi/2, atol=1e-6, rtol=0):
                twoth += 1e-6
        else:
            twoth = np.array(twoth, dtype=float)
            twoth[np.isclose(twoth, np.pi/2, atol=1e-6, rtol=0)] += 1e-6

        alpha = self.alpha % (2 * np.pi)
        D = self.distance

        x = D * (cos(alpha)*tan(twoth)*sin(gamma) + sin(alpha)) / \
            (cos(alpha) - sin(alpha)*tan(twoth)*sin(gamma))
        y = -(x*sin(alpha) + D*cos(alpha)) * tan(twoth) * cos(gamma)

        # (x, y) corresponds to the lower left of the diagram, and (row, col) = (0, 0) corresponds to the upper left.
        # x corresponds to col, and y corresponds to row (very confusing).
        return (self.image.shape[-2] - y * self.densityXY[1] - self.centerXY[1],
                x * self.densityXY[0] + self.centerXY[0])

    def relim(self):
        rr, cc = np.indices((self.image.shape[-2], self.image.shape[-1]))
        twoth, gamma = self.rowcol_to_angles(rr, cc)
        if twoth.size > 0 and gamma.size > 0:
            self.limits = (np.min(twoth), np.max(twoth), np.min(gamma), np.max(gamma))
        return self.limits

    def convert(self, n_twoth=None, n_gamma=None):
        if self.image.data is None:
            raise ValueError('Cannot convert because image has not been loaded.')
        if n_twoth is None:
            n_twoth = self.image.shape[-1]
        if n_gamma is None:
            n_gamma = self.image.shape[-2]

        # determine range of twoth and gamma
        self.relim()
        seq_twoth = np.linspace(self.limits[0], self.limits[1], n_twoth)
        if self.alpha >= 0:
            seq_gamma = np.linspace(self.limits[2], self.limits[3], n_gamma)
        else:
            seq_gamma = np.linspace(self.limits[3], self.limits[2], n_gamma)
        self.indexes = tuple(np.rad2deg((seq_gamma, seq_twoth)))

        # create regular (twoth, gamma) grid and then convert it to (row, col)
        newrow, newcol = self.angles_to_rowcol(*np.meshgrid(seq_twoth, seq_gamma, indexing='xy'))

        # perform interpolation
        f = RegularGridInterpolator(
            (np.arange(self.image.shape[-2]), np.arange(self.image.shape[-1])),
            self.image.data,
            method='nearest',
            bounds_error=False,
            fill_value=0
         )
        new = f(np.c_[newrow.ravel(), newcol.ravel()]).reshape((n_gamma, n_twoth)).astype(self.image.data.dtype)
        if self.scale != 1 or self.offset != 0:
            new = new.astype(np.float32) * self.scale + self.offset
        self.data_converted = new

    def gridline(self, angle_deg, axis='twoth', delta_deg=0.1):
        """Extracts data for grid lines of constant 2θ and constant γ that can be plotted on a GADDS image.

        Note: If self.alpha or self.distance are changed manually, self.relim() must be executed before gridline."""
        angle = np.deg2rad(angle_deg)
        delta = np.deg2rad(delta_deg)
        if axis == 'twoth':
            if not self.limits[0] <= angle <= self.limits[1]:
                return [], []
            rows, cols = self.angles_to_rowcol(angle, np.arange(self.limits[2], self.limits[3], delta))
        elif axis == 'gamma':
            if not self.limits[2] <= angle <= self.limits[3]:
                return [], []
            rows, cols = self.angles_to_rowcol(np.arange(self.limits[0], self.limits[1], delta), angle)
        else:
            raise ValueError('unknown axis: %s' % axis)
        idx = (0 <= rows) & (rows < self.image.shape[-2]) & (0 <= cols) & (cols < self.image.shape[-1])
        return cols[idx], rows[idx]

    def load_headers(self):
        """Reads necessary parameters from the image file header."""
        image = self.image
        if isinstance(image, BrukerImage):
            """gfrm files have versions 86 and 100 (can be confirmed at the beginning of the file when opened with a text editor).
            Version 86 is described in detail in gaddsref.pdf included with GADDS, but there is no explanation for version 100.
            Here, the parameters are inferred from the values displayed when opened with DIFFRAC.EVA.
            """
            if 'UNWARPED' not in image.header['TYPE']:
                logger.warning('This frame has NOT been UNWARPED (corrected), and may contain some error.', stack_info=True)

            # In order: 2theta, omega, phi, chi
            diffractometer_angles = [float(angles) for angles in image.header['ANGLES'].split()]
            self.alpha, _, _, self.chi = np.deg2rad(diffractometer_angles)

            self.goniometer_pos = tuple(float(x) for x in image.header['AXES2'].split())

            # CENTER
            # ver 86: two values, x and y, are recorded.
            # ver100: There are four values, but the latter two seem to be actually used.
            self.centerXY = tuple(float(x) for x in image.header['CENTER'].split()[-2:])

            # linear scale and offset
            if 'LINEAR' in image.header:
                self.scale, self.offset = (float(x) for x in image.header['LINEAR'].split()[:2])

            # PIXPERCM: Number of pixels per cm (when the frame is 512x512)
            m = re.search('PIXPERCM:([0-9\\.]+)', image.header['DETTYPE'])
            if m:
                pixpercm = float(m.groups()[0])
            else:
                try:
                    # For version 100, the DETTYPE format seems to be "name pixpercm cmtogrid (and so on)".
                    pixpercm = float(image.header['DETTYPE'].split()[1])
                except (ValueError, IndexError):
                    pixpercm = DETTYPE[image.header['DETTYPE']][0]
            nrows, ncols = image.data.shape
            self.densityXY = (pixpercm * ncols / 512, pixpercm * nrows / 512)

            # detector distance (distance between sample and detector plane)
            # The value of CMTOGRID is added to the first number in the DISTANC field.
            distanc = float(image.header['DISTANC'].split()[0])
            m = re.search('CMTOGRID:([0-9\\.]+)', image.header['DETTYPE'])
            if m:
                cmtogrid = float(m.groups()[0])
            else:
                try:
                    cmtogrid = float(image.header['DETTYPE'].split()[2])
                except (ValueError, IndexError):
                    cmtogrid = DETTYPE[image.header['DETTYPE']][1]
            self.distance = distanc + cmtogrid
        else:
            pass
        # Only call relim if image data is available
        if self.image.data is not None:
            self.relim()

    def integrate(self, twotheta_range, gamma_range, mode='2theta'):
        """
        Integrate the image data within specified 2θ and γ angular ranges.
        This method sums the converted image data over the specified ranges of 2θ (two-theta) and γ (gamma) angles.
        The integration can be performed along either the 2θ or γ axis, as specified by the `mode` parameter.

        Parameters
        ----------
        twotheta_range : tuple of float
            The (min, max) range of 2θ angles in degrees over which to integrate.
        gamma_range : tuple of float
            The (min, max) range of γ angles in degrees over which to integrate.
        mode : {'2theta', 'gamma'}, optional
            The axis along which to integrate:
                - '2theta': Integrate over γ, returning intensity as a function of 2θ.
                - 'gamma': Integrate over 2θ, returning intensity as a function of γ.
            Default is '2theta'.
        
        Returns
        -------
        intensity_slice : numpy.ndarray
            The integrated intensity values along the selected axis.
        int_borders : tuple of numpy.ndarray
            Tuple containing arrays of coordinates for the integration borders:
                (twoth_lower_border, twoth_upper_border, gamma_lower_border, gamma_upper_border),
            where each border is an array of (row, col) coordinates corresponding to the edges of the integrated region.

        Raises
        ------
        ValueError
            If the specified angular ranges are outside the image limits or if the data has not been converted.

        Notes
        -----
        - The method assumes that `self.data_converted` contains the processed image data and that `self.indexes`
          provides the mapping from angular values to data indices.
        - The method requires that the data has been converted prior to integration.

        """

        if self.data_converted.size == 0:
            raise ValueError('Data has not been converted yet. Please call convert() first.')

        gamma_in_rad = np.deg2rad(gamma_range)
        twotheta_in_rad = np.deg2rad(twotheta_range)

        if not self.limits[0] <= twotheta_in_rad[0] <= self.limits[1] or \
            not self.limits[0] <= twotheta_in_rad[1] <= self.limits[1] or \
            not self.limits[2] <= gamma_in_rad[0] <= self.limits[3] or \
            not self.limits[2] <= gamma_in_rad[1] <= self.limits[3]:
            raise ValueError('Specified range is outside the limits of the image.')

        gamma_mask = (self.indexes[0] >= gamma_range[0]) & (self.indexes[0] <= gamma_range[1])
        twoth_mask = (self.indexes[1] >=  twotheta_range[0]) & (self.indexes[1] <= twotheta_range[1])

        if mode == '2theta':
            sum_axis = 0
        elif mode == 'gamma':
            sum_axis = 1
        else:
            raise ValueError('Unknown mode: %s' % mode)
        
        intensity_slice = np.sum(self.data_converted[gamma_mask, :][:, twoth_mask], axis=sum_axis)

        int_border_step = np.deg2rad(0.1)
        twoth_lower_border = self.angles_to_rowcol(twotheta_in_rad[0], np.arange(gamma_in_rad[0], gamma_in_rad[1], int_border_step))
        twoth_upper_border = self.angles_to_rowcol(twotheta_in_rad[1], np.arange(gamma_in_rad[0], gamma_in_rad[1], int_border_step))
        gamma_lower_border = self.angles_to_rowcol(np.arange(twotheta_in_rad[0], twotheta_in_rad[1], int_border_step), gamma_in_rad[0])
        gamma_upper_border = self.angles_to_rowcol(np.arange(twotheta_in_rad[0], twotheta_in_rad[1], int_border_step), gamma_in_rad[1])
        
        # Reverse the order of the coordinates to match the plotting convention
        int_borders = (twoth_lower_border[::-1], twoth_upper_border[::-1], gamma_lower_border[::-1], gamma_upper_border[::-1])
        
        return intensity_slice, int_borders

if __name__ == '__main__':
    # usage example
    import matplotlib.pyplot as plt
    import sys
    try:
        f = sys.argv[1]
    except IndexError:
        raise ValueError('please specify filename.')
    areaimage = AreaDetectorImage(f)
    areaimage.convert()
    matrix_original = areaimage.image.data

    dx = areaimage.indexes[1][1] - areaimage.indexes[1][0]
    dy = areaimage.indexes[0][1] - areaimage.indexes[0][0]
    extent = (
        areaimage.indexes[1][0]-dx/2, areaimage.indexes[1][-1]+dx/2,
        areaimage.indexes[0][-1]-dy/2, areaimage.indexes[0][0]+dy/2
    )
    im = plt.imshow(areaimage.data_converted,
                interpolation='nearest',
                vmin=0, vmax=10,
                aspect='auto',
                origin='upper',
                extent=extent)
    plt.colorbar()
    plt.show()
