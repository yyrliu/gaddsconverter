from scipy.interpolate import RegularGridInterpolator
import numpy as np

# Import the extended class with reverse conversion capabilities
from gadds import AreaDetectorImage

class AreaDetectorImageConverter(AreaDetectorImage):
    """
    Extended AreaDetectorImage class that supports reverse conversion from 2θ-γ space back to detector image.
    """
    
    def __init__(self, image=None):
        """
        Initialize the converter with an optional image.
        
        :param image: filename, file-like, or fabio image object (optional)
        """
        super().__init__(image)
        if image:
            self.detector_shape = (self.image.data.shape[-1], self.image.data.shape[-2])  # Store original detector shape for reverse conversion
        
    def set_detector_parameters(self, alpha_rad, distance_cm, center_xy, density_xy, 
                               detector_shape, scale=1, offset=0):
        """
        Manually set detector parameters for reverse conversion when no original image is available.
        
        :param alpha_deg: 2θ center angle in rad
        :param distance_cm: detector distance in cm
        :param center_xy: detector center (x, y) in pixels
        :param density_xy: pixel density (x, y) in pixels/cm
        :param detector_shape: target detector shape (height, width)
        :param scale: linear scale factor
        :param offset: linear offset
        """
        self.alpha = alpha_rad
        self.distance = distance_cm
        self.centerXY = center_xy
        self.densityXY = density_xy
        self.detector_shape = detector_shape
        self.scale = scale
        self.offset = offset
        
        # Create a dummy image with the specified shape for calculations
        if self.image.data is None:
            self.image.data = np.zeros(detector_shape, dtype=np.float32)
        
        self.indexes = (np.array([]), np.array([]))

        self.relim()

    def set_converted_data_with_coordinates(self, data_converted, gamma_coords, twoth_coords):
        """
        Set the converted data along with coordinate arrays.
        
        :param data_converted: The 2D converted data array
        :param gamma_coords: Array of gamma coordinates in degrees
        :param twoth_coords: Array of 2theta coordinates in degrees
        """
        self.data_converted = data_converted
        self.indexes = (gamma_coords, twoth_coords)  # Store in degrees as expected

    def convert_back_to_detector(self, n_row=None, n_col=None, method='nearest'):
        
        if self.data_converted is None:
            raise ValueError('Cannot convert because converted image does not exist.')
        
        if len(self.indexes[0]) == 0 or len(self.indexes[1]) == 0:
            raise ValueError('Coordinate indexes not properly set. Use set_converted_data_with_coordinates() first.')
        
        if n_row is None:
            n_row = self.data_converted.shape[-2]  # Default to original gamma dimension
        if n_col is None:
            n_col = self.data_converted.shape[-1]  # Default to original twoth dimension

        # KEY FIX: Create pixel grid that covers the SAME detector area
        # but with different sampling density
        
        # Get the ORIGINAL detector dimensions to preserve field of view
        original_rows = self.data_converted.shape[-2]
        original_cols = self.data_converted.shape[-1]
        
        # Create sequences that span the SAME range as original detector
        # but with custom resolution
        seq_row = np.linspace(0, original_rows-1, n_row)
        seq_col = np.linspace(0, original_cols-1, n_col)
        
        # Convert these to angular coordinates (preserves full FOV)
        new_twoth, new_gamma = self.rowcol_to_angles(*np.meshgrid(seq_row, seq_col, indexing='ij'))
        
        # Get coordinate arrays from indexes (already in degrees)
        seq_gamma = np.deg2rad(self.indexes[0])  # gamma coordinates converted to radians
        seq_twoth = np.deg2rad(self.indexes[1])  # 2theta coordinates converted to radians
        
        f = RegularGridInterpolator(
            (seq_gamma, seq_twoth),
            self.data_converted,
            method=method,
        )
        
        new = f(np.c_[new_gamma.ravel(), new_twoth.ravel()]).reshape((n_row, n_col)).astype(self.data_converted.dtype)
        if self.scale != 1 or self.offset != 0:
            new = new.astype(np.float32) * self.scale + self.offset
        
        return new
        