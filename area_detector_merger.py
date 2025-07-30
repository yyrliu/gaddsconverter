#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Area Detector Merging Module

This module provides functionality for merging multiple GADDS area detector images
with advanced overlap handling techniques. It implements two main approaches:

1. Data-driven merging: Uses interpolated detector data coverage to determine boundaries
2. Convex hull merging: Creates a convex hull around the overlap region for smooth boundaries

Classes:
    AreaDetectorMerger: Main class for merging area detector images

Dependencies:
    - numpy
    - scipy
    - matplotlib (for visualization)
    - gadds (AreaDetectorImage class)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from typing import List, Tuple, Optional, Dict, Any
from gadds import AreaDetectorImage


class AreaDetectorMerger:
    """
    A class for merging multiple GADDS area detector images with advanced overlap handling.
    
    This class provides two main merging strategies:
    1. Data-driven approach: Uses actual detector data coverage for boundary detection
    2. Convex hull approach: Creates geometric boundaries around overlap regions
    
    Attributes:
        area_detectors (List[AreaDetectorImage]): List of area detector images to merge
        common_grid (Dict): Common coordinate grid information
        merged_data (np.ndarray): Raw merged data before normalization
        weight_grid (np.ndarray): Weight grid for normalization
        normalized_data (np.ndarray): Final normalized merged data
    """
    
    def __init__(self, area_detectors: List[AreaDetectorImage]):
        """
        Initialize the merger with a list of area detector images.
        
        Args:
            area_detectors: List of AreaDetectorImage objects to merge
        """
        self.area_detectors = area_detectors
        self.common_grid = {}
        self.merged_data = None
        self.weight_grid = None
        self.normalized_data = None
        self.detector_coverage_masks = []
        
        # Convex hull specific attributes
        self.convex_hull_weight_grid = None
        self.convex_hull_normalized_data = None
        self.hull_points = None
        self.hull_area = None

        # Safe guards for untested configurations
        if len(area_detectors) > 2:
            raise NotImplementedError("Only tested with 2 detectors so far, for more than 2 detectors, please only use the data-driven method.")
        
        detector_chis = [detector.chi for detector in area_detectors]
        if not detector_chis.count(detector_chis[0]) == len(detector_chis):
            raise NotImplementedError("All detectors must have the same chi angle for merging.")
        
        self._setup_common_grid()
    
    def _setup_common_grid(self) -> None:
        """
        Create a common coordinate grid that covers all detectors.
        Uses the finest resolution available from all detectors.
        """
        # Get coordinate ranges for each detector
        detector_ranges = []
        for area_detector in self.area_detectors:
            twoth_min = area_detector.indexes[1].min()
            twoth_max = area_detector.indexes[1].max()
            gamma_min = area_detector.indexes[0].min()
            gamma_max = area_detector.indexes[0].max()
            detector_ranges.append((twoth_min, twoth_max, gamma_min, gamma_max))
        
        # Calculate overall bounds
        all_twoth_min = min(r[0] for r in detector_ranges)
        all_twoth_max = max(r[1] for r in detector_ranges)
        all_gamma_min = min(r[2] for r in detector_ranges)
        all_gamma_max = max(r[3] for r in detector_ranges)
        
        # Use finest resolution from all detectors
        twoth_step = min(
            det.indexes[1][1] - det.indexes[1][0] 
            for det in self.area_detectors
        )
        gamma_step = min(
            det.indexes[0][1] - det.indexes[0][0] 
            for det in self.area_detectors
        )
        
        # Create grid
        n_twoth = int((all_twoth_max - all_twoth_min) / twoth_step) + 1
        n_gamma = int((all_gamma_max - all_gamma_min) / gamma_step) + 1
        
        common_twoth = np.linspace(all_twoth_min, all_twoth_max, n_twoth)
        common_gamma = np.linspace(all_gamma_min, all_gamma_max, n_gamma)
        
        self.common_grid = {
            'twoth': common_twoth,
            'gamma': common_gamma,
            'twoth_step': twoth_step,
            'gamma_step': gamma_step,
            'n_twoth': n_twoth,
            'n_gamma': n_gamma,
            'bounds': (all_twoth_min, all_twoth_max, all_gamma_min, all_gamma_max)
        }
    
    def merge_data_driven(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Merge detectors using data-driven boundary detection.
        
        This method uses interpolated detector data coverage to determine boundaries.
        Weight = 1 for single detector coverage, Weight = 2 for actual overlap.
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (merged_data, weight_grid, normalized_data)
        """
        if verbose:
            print("=== Method 1: Data-Driven Weight Grid ===")
        
        # Initialize arrays
        n_gamma, n_twoth = self.common_grid['n_gamma'], self.common_grid['n_twoth']
        self.merged_data = np.zeros((n_gamma, n_twoth))
        self.weight_grid = np.zeros((n_gamma, n_twoth))
        
        # Create coordinate meshes
        twoth_mesh, gamma_mesh = np.meshgrid(
            self.common_grid['twoth'], 
            self.common_grid['gamma'], 
            indexing='xy'
        )
        
        self.detector_coverage_masks = []
        
        # Process each detector
        for i, area_detector in enumerate(self.area_detectors):
            if verbose:
                print(f"Processing detector {i}...")
            
            # Create data interpolator
            data_interpolator = RegularGridInterpolator(
                (area_detector.indexes[0], area_detector.indexes[1]),
                area_detector.data_converted,
                bounds_error=False, fill_value=0
            )
            interpolated_data = data_interpolator((gamma_mesh, twoth_mesh))
            
            # Create coverage interpolator for accurate boundaries
            detector_mask = area_detector.data_converted > 0
            coverage_interpolator = RegularGridInterpolator(
                (area_detector.indexes[0], area_detector.indexes[1]),
                detector_mask.astype(float),
                bounds_error=False, fill_value=0, method='linear'
            )
            coverage_values = coverage_interpolator((gamma_mesh, twoth_mesh))
            
            # Determine detector coverage with boundary constraints
            within_bounds = (
                (np.deg2rad(gamma_mesh) >= area_detector.limits[2]) & 
                (np.deg2rad(gamma_mesh) <= area_detector.limits[3]) &
                (np.deg2rad(twoth_mesh) >= area_detector.limits[0]) & 
                (np.deg2rad(twoth_mesh) <= area_detector.limits[1])
            )
            
            data_coverage_mask = interpolated_data > 0
            coverage_mask = (coverage_values > 0.1) & within_bounds
            final_coverage_mask = data_coverage_mask | (coverage_mask & (interpolated_data >= 0))
            
            self.detector_coverage_masks.append(final_coverage_mask)
            
            # Accumulate data and weights
            self.merged_data[final_coverage_mask] += interpolated_data[final_coverage_mask]
            self.weight_grid[final_coverage_mask] += 1
        
        # Apply normalization
        self.normalized_data = np.divide(
            self.merged_data, 
            self.weight_grid, 
            out=np.zeros_like(self.merged_data), 
            where=self.weight_grid != 0
        )
        
        if verbose:
            overlap_pixels = np.sum(self.weight_grid > 1)
            total_pixels = np.sum(self.weight_grid > 0)
            print("Coverage analysis:")
            for i, mask in enumerate(self.detector_coverage_masks):
                print(f"  Detector {i}: {np.sum(mask):,} pixels")
            print(f"  Overlap region: {overlap_pixels:,} pixels")
            print(f"  Total coverage: {total_pixels:,} pixels")
            print("✓ Data-driven merge completed")
        
        return self.merged_data, self.weight_grid, self.normalized_data
    
    def merge_convex_hull(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Merge detectors using convex hull boundary expansion.
        
        This method creates a convex hull around the original overlap region
        and assigns weight = 2 to all points inside the hull.
        
        Args:
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (merged_data, convex_hull_weight_grid, convex_hull_normalized_data)
        """
        if verbose:
            print("=== Method 2: Convex Hull Weight Grid ===")
        
        # First ensure we have data-driven results
        if self.weight_grid is None:
            self.merge_data_driven(verbose=False)
        
        # Find all points where original weight_grid == 2 (actual overlap region)
        overlap_points_indices = np.where(self.weight_grid == 2)
        overlap_gamma_indices = overlap_points_indices[0]  # Row indices (γ)
        overlap_twoth_indices = overlap_points_indices[1]  # Column indices (2θ)
        
        # Convert indices to actual coordinates
        overlap_gamma_coords = self.common_grid['gamma'][overlap_gamma_indices]
        overlap_twoth_coords = self.common_grid['twoth'][overlap_twoth_indices]
        
        if verbose:
            print(f"Original overlap region: {len(overlap_gamma_coords):,} points")
            print(f"  2θ range: [{np.min(overlap_twoth_coords):.2f}, {np.max(overlap_twoth_coords):.2f}]°")
            print(f"  γ range: [{np.min(overlap_gamma_coords):.2f}, {np.max(overlap_gamma_coords):.2f}]°")
        
        # Initialize with copy of original weight grid
        self.convex_hull_weight_grid = self.weight_grid.copy()
        
        if len(overlap_gamma_coords) > 3:  # Need at least 3 points for hull
            try:
                # Create convex hull around overlap points
                overlap_points_2d = np.column_stack((overlap_twoth_coords, overlap_gamma_coords))
                hull = ConvexHull(overlap_points_2d)
                self.hull_points = overlap_points_2d[hull.vertices]
                hull_points_closed = np.vstack([self.hull_points, self.hull_points[0]])
                
                # Calculate hull area
                self.hull_area = 0.5 * np.abs(
                    np.dot(hull_points_closed[:-1, 0], hull_points_closed[1:, 1]) - 
                    np.dot(hull_points_closed[1:, 0], hull_points_closed[:-1, 1])
                )
                
                if verbose:
                    print(f"Convex hull created with {len(self.hull_points)} vertices")
                    print(f"Hull area: {self.hull_area:.2f} deg²")
                
                # Create mesh for testing point inclusion
                twoth_mesh, gamma_mesh = np.meshgrid(
                    self.common_grid['twoth'], 
                    self.common_grid['gamma'], 
                    indexing='xy'
                )
                
                # Test all grid points for inclusion in hull
                hull_path = Path(self.hull_points)
                mesh_points = np.column_stack((twoth_mesh.ravel(), gamma_mesh.ravel()))
                inside_hull = hull_path.contains_points(mesh_points)
                inside_hull_2d = inside_hull.reshape(twoth_mesh.shape)
                
                # Set weight = 2 for points inside hull that have detector coverage
                points_with_coverage = self.convex_hull_weight_grid > 0
                points_in_hull = inside_hull_2d & points_with_coverage
                self.convex_hull_weight_grid[points_in_hull] = 2
                
                # Calculate statistics
                original_overlap_points = np.sum(self.weight_grid == 2)
                hull_overlap_points = np.sum(self.convex_hull_weight_grid == 2)
                additional_points = hull_overlap_points - original_overlap_points
                expansion_pct = (100 * additional_points / original_overlap_points 
                               if original_overlap_points > 0 else 0)
                
                if verbose:
                    print("Hull expansion analysis:")
                    print(f"  Original overlap: {original_overlap_points:,} points")
                    print(f"  Hull overlap: {hull_overlap_points:,} points") 
                    print(f"  Additional points: {additional_points:,} ({expansion_pct:.1f}% increase)")
                
                hull_success = True
                
            except Exception as e:
                if verbose:
                    print(f"Error creating convex hull: {e}")
                hull_success = False
                
        else:
            if verbose:
                print("Insufficient overlap points for convex hull")
            hull_success = False
        
        # Apply normalization with new weights
        self.convex_hull_normalized_data = np.divide(
            self.merged_data,
            self.convex_hull_weight_grid,
            out=np.zeros_like(self.merged_data),
            where=self.convex_hull_weight_grid != 0
        )
        
        if verbose:
            print(f"✓ Convex hull merge {'completed' if hull_success else 'failed'}")
        
        return self.merged_data, self.convex_hull_weight_grid, self.convex_hull_normalized_data
    
    def get_detector_info(self) -> List[Dict[str, Any]]:
        """
        Get information about each detector.
        
        Returns:
            List of dictionaries containing detector information
        """
        detector_info = []
        for i, area_detector in enumerate(self.area_detectors):
            info = {
                'index': i,
                'alpha_deg': np.rad2deg(area_detector.alpha),
                'distance_cm': area_detector.distance,
                'center_xy': area_detector.centerXY,
                'density_xy': area_detector.densityXY,
                'scale': area_detector.scale,
                'offset': area_detector.offset,
                'shape': area_detector.data_converted.shape,
                'twoth_range': (area_detector.indexes[1].min(), area_detector.indexes[1].max()),
                'gamma_range': (area_detector.indexes[0].min(), area_detector.indexes[0].max())
            }
            
            # Add header information if available
            if hasattr(area_detector.image, 'header') and area_detector.image.header:
                info['title'] = area_detector.image.header.get('TITLE', '').strip()
            else:
                info['title'] = 'No header information available'
                
            detector_info.append(info)
        
        return detector_info
    
    def print_detector_info(self) -> None:
        """Print detailed information about each detector."""
        detector_info = self.get_detector_info()
        
        for info in detector_info:
            print(f"=== Detector {info['index']} Information ===")
            print(f"TITLE: {info['title']}")
            print(f"Alpha (2θ center): {info['alpha_deg']:.2f}°")
            print(f"Distance: {info['distance_cm']:.2f} cm")
            print(f"Detector center (x, y): {info['center_xy']}")
            print(f"Pixel density (x, y): {info['density_xy']} pixels/cm")
            print(f"Scale factor: {info['scale']}")
            print(f"Offset: {info['offset']}")
            print(f"Converted image shape: {info['shape']}")
            print(f"2θ range: [{info['twoth_range'][0]:.2f}, {info['twoth_range'][1]:.2f}]°")
            print(f"γ range: [{info['gamma_range'][0]:.2f}, {info['gamma_range'][1]:.2f}]°")
            print("=== end ===\n")
    
    def get_quality_metrics(self) -> Dict[str, float]:
        """
        Calculate quality metrics comparing the two merging methods.
        
        Returns:
            Dictionary containing various quality metrics
        """
        if self.normalized_data is None or self.convex_hull_normalized_data is None:
            raise ValueError("Both merging methods must be run before calculating metrics")
        
        # Calculate differences
        data_difference = self.convex_hull_normalized_data - self.normalized_data
        data_difference[self.weight_grid == 0] = np.nan
        
        relative_diff = np.divide(
            data_difference, 
            self.normalized_data,
            out=np.zeros_like(data_difference),
            where=self.normalized_data != 0
        ) * 100
        relative_diff[self.weight_grid == 0] = np.nan
        
        # Calculate metrics
        rms_diff = np.sqrt(np.nanmean(data_difference**2))
        max_diff = np.nanmax(np.abs(data_difference))
        mean_rel_diff = np.nanmean(np.abs(relative_diff))
        
        original_overlap = np.sum(self.weight_grid == 2)
        hull_overlap = np.sum(self.convex_hull_weight_grid == 2)
        expansion_factor = hull_overlap / original_overlap if original_overlap > 0 else 0
        
        metrics = {
            'rms_difference': rms_diff,
            'max_difference': max_diff,
            'mean_relative_difference_pct': mean_rel_diff,
            'original_overlap_points': original_overlap,
            'hull_overlap_points': hull_overlap,
            'expansion_factor': expansion_factor,
            'hull_area_deg2': self.hull_area if self.hull_area is not None else 0,
            'hull_vertices': len(self.hull_points) if self.hull_points is not None else 0
        }
        
        return metrics
    
    def create_comparison_plot(self, figsize: Tuple[int, int] = (24, 16), 
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive side-by-side comparison plot.
        
        Args:
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure
            
        Returns:
            The matplotlib Figure object
        """
        if self.normalized_data is None or self.convex_hull_normalized_data is None:
            raise ValueError("Both merging methods must be run before creating comparison plot")
        
        fig = plt.figure(figsize=figsize)
        
        # Calculate display extent
        extent = [
            self.common_grid['twoth'].min(), 
            self.common_grid['twoth'].max(),
            self.common_grid['gamma'].max(), 
            self.common_grid['gamma'].min()
        ]
        
        # Row 1: Weight grids
        self._plot_weight_comparison(fig, extent)
        
        # Row 2: Normalized data
        self._plot_data_comparison(fig, extent)
        
        # Row 3: Analysis plots
        self._plot_analysis(fig)
        
        # Add grid lines to image plots
        self._add_grid_lines(fig, extent)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_weight_comparison(self, fig: plt.Figure, extent: List[float]) -> None:
        """Plot weight grid comparison (first row)."""
        # Original weight grid
        ax1 = plt.subplot(3, 4, 1)
        im1 = ax1.imshow(self.weight_grid, cmap='viridis', origin='upper', 
                        extent=extent, aspect='auto')
        ax1.set_title('Method 1: Original Weight Grid\n(Data-driven boundaries)', fontsize=12)
        ax1.set_xlabel('2θ (degrees)')
        ax1.set_ylabel('γ (degrees)')
        plt.colorbar(im1, ax=ax1, label='Weight', shrink=0.8)
        
        # Convex hull weight grid
        ax2 = plt.subplot(3, 4, 2)
        im2 = ax2.imshow(self.convex_hull_weight_grid, cmap='viridis', origin='upper',
                        extent=extent, aspect='auto')
        ax2.set_title('Method 2: Convex Hull Weight Grid\n(Geometric expansion)', fontsize=12)
        ax2.set_xlabel('2θ (degrees)')
        ax2.set_ylabel('γ (degrees)')
        
        # Add hull boundary if available
        if self.hull_points is not None:
            hull_points_closed = np.vstack([self.hull_points, self.hull_points[0]])
            ax2.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 'r-',
                    linewidth=2, alpha=0.8, label='Hull Boundary')
            ax2.legend(fontsize=10)
        
        plt.colorbar(im2, ax=ax2, label='Weight', shrink=0.8)
        
        # Weight difference
        ax3 = plt.subplot(3, 4, 3)
        weight_diff = self.convex_hull_weight_grid - self.weight_grid
        weight_diff[self.weight_grid == 0] = np.nan
        im3 = ax3.imshow(weight_diff, cmap='RdBu_r', origin='upper', extent=extent,
                        aspect='auto', vmin=-1, vmax=1)
        ax3.set_title('Weight Difference\n(Method 2 - Method 1)', fontsize=12)
        ax3.set_xlabel('2θ (degrees)')
        ax3.set_ylabel('γ (degrees)')
        plt.colorbar(im3, ax=ax3, label='Weight Δ', shrink=0.8)
        
        # Overlap regions overlay
        ax4 = plt.subplot(3, 4, 4)
        overlap_comparison = np.zeros_like(self.weight_grid)
        overlap_comparison[self.weight_grid == 2] = 1  # Original overlap
        overlap_comparison[self.convex_hull_weight_grid == 2] += 2  # Hull overlap
        overlap_comparison[self.weight_grid == 0] = np.nan
        im4 = ax4.imshow(overlap_comparison, cmap='Set1', origin='upper',
                        extent=extent, aspect='auto')
        ax4.set_title('Overlap Region Comparison\n(1=Original, 2=Hull, 3=Both)', fontsize=12)
        ax4.set_xlabel('2θ (degrees)')
        ax4.set_ylabel('γ (degrees)')
        plt.colorbar(im4, ax=ax4, label='Overlap Type', shrink=0.8)
    
    def _plot_data_comparison(self, fig: plt.Figure, extent: List[float]) -> None:
        """Plot normalized data comparison (second row)."""
        # Original normalized data
        ax5 = plt.subplot(3, 4, 5)
        normalized_display = self.normalized_data.copy()
        normalized_display[self.weight_grid == 0] = np.nan
        
        self._plot_intensity_data(ax5, normalized_display, extent,
                                'Method 1: Normalized Data\n(Original approach)')
        
        # Convex hull normalized data  
        ax6 = plt.subplot(3, 4, 6)
        hull_normalized_display = self.convex_hull_normalized_data.copy()
        hull_normalized_display[self.convex_hull_weight_grid == 0] = np.nan
        
        self._plot_intensity_data(ax6, hull_normalized_display, extent,
                                'Method 2: Normalized Data\n(Convex hull approach)')
        
        # Add hull boundary
        if self.hull_points is not None:
            hull_points_closed = np.vstack([self.hull_points, self.hull_points[0]])
            ax6.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 'r-',
                    linewidth=2, alpha=0.6)
        
        # Data difference
        ax7 = plt.subplot(3, 4, 7)
        data_difference = self.convex_hull_normalized_data - self.normalized_data
        data_difference[self.weight_grid == 0] = np.nan
        im7 = ax7.imshow(data_difference, cmap='RdBu_r', origin='upper',
                        extent=extent, aspect='auto')
        ax7.set_title('Data Difference\n(Method 2 - Method 1)', fontsize=12)
        ax7.set_xlabel('2θ (degrees)')
        ax7.set_ylabel('γ (degrees)')
        plt.colorbar(im7, ax=ax7, label='Intensity Δ', shrink=0.8)
        
        # Relative difference (%)
        ax8 = plt.subplot(3, 4, 8)
        relative_diff = np.divide(
            data_difference,
            self.normalized_data,
            out=np.zeros_like(data_difference),
            where=self.normalized_data != 0
        ) * 100
        relative_diff[self.weight_grid == 0] = np.nan
        im8 = ax8.imshow(relative_diff, cmap='RdBu_r', origin='upper',
                        extent=extent, aspect='auto', vmin=-50, vmax=50)
        ax8.set_title('Relative Difference\n(% change)', fontsize=12)
        ax8.set_xlabel('2θ (degrees)')
        ax8.set_ylabel('γ (degrees)')
        plt.colorbar(im8, ax=ax8, label='% Change', shrink=0.8)
    
    def _plot_intensity_data(self, ax: plt.Axes, data: np.ndarray, extent: List[float], 
                           title: str) -> None:
        """Helper method to plot intensity data with appropriate scaling."""
        if np.nansum(data) > 0:
            valid_data = data[data > 0]
            if len(valid_data) > 0:
                vmin = np.percentile(valid_data, 5)
                vmax = np.percentile(valid_data, 95)
                if vmin > 0 and vmax > vmin:
                    im = ax.imshow(data, cmap='viridis',
                                  norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                                  origin='upper', extent=extent, aspect='auto')
                else:
                    im = ax.imshow(data, cmap='viridis', origin='upper',
                                  extent=extent, aspect='auto')
            else:
                im = ax.imshow(data, cmap='viridis', origin='upper',
                              extent=extent, aspect='auto')
        else:
            im = ax.imshow(data, cmap='viridis', origin='upper',
                          extent=extent, aspect='auto')
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('2θ (degrees)')
        ax.set_ylabel('γ (degrees)')
        plt.colorbar(im, ax=ax, label='Intensity', shrink=0.8)
    
    def _plot_analysis(self, fig: plt.Figure) -> None:
        """Plot analysis charts (third row)."""
        # 1D comparison along 2θ (constant γ)
        ax9 = plt.subplot(3, 4, 9)
        gamma_center_idx = len(self.common_grid['gamma']) // 2
        slice_original = self.normalized_data[gamma_center_idx, :]
        slice_hull = self.convex_hull_normalized_data[gamma_center_idx, :]
        slice_weights_original = self.weight_grid[gamma_center_idx, :]
        
        valid_slice = slice_weights_original > 0
        ax9.plot(self.common_grid['twoth'][valid_slice], slice_original[valid_slice],
                'b-', linewidth=2, label='Original', alpha=0.8)
        ax9.plot(self.common_grid['twoth'][valid_slice], slice_hull[valid_slice],
                'r--', linewidth=2, label='Convex Hull', alpha=0.8)
        ax9.set_xlabel('2θ (degrees)')
        ax9.set_ylabel('Normalized Intensity')
        ax9.set_title(f'1D Comparison at γ = {self.common_grid["gamma"][gamma_center_idx]:.1f}°',
                     fontsize=12)
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # Weight distribution histogram
        ax10 = plt.subplot(3, 4, 10)
        weights_original = self.weight_grid[self.weight_grid > 0]
        weights_hull = self.convex_hull_weight_grid[self.convex_hull_weight_grid > 0]
        
        ax10.hist(weights_original, bins=np.arange(0.5, 3.5, 1), alpha=0.7,
                 label='Original', color='blue', density=True)
        ax10.hist(weights_hull, bins=np.arange(0.5, 3.5, 1), alpha=0.7,
                 label='Convex Hull', color='red', density=True)
        ax10.set_xlabel('Weight Value')
        ax10.set_ylabel('Density')
        ax10.set_title('Weight Distribution Comparison', fontsize=12)
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        # Statistics comparison
        ax11 = plt.subplot(3, 4, 11)
        self._plot_statistics_comparison(ax11)
        
        # Quality metrics
        ax12 = plt.subplot(3, 4, 12)
        self._plot_quality_metrics(ax12)
    
    def _plot_statistics_comparison(self, ax: plt.Axes) -> None:
        """Plot statistics comparison bar chart."""
        stats_data = {
            'Original': [
                np.sum(self.weight_grid == 1),
                np.sum(self.weight_grid == 2),
                np.sum(self.weight_grid > 0)
            ],
            'Convex Hull': [
                np.sum(self.convex_hull_weight_grid == 1),
                np.sum(self.convex_hull_weight_grid == 2),
                np.sum(self.convex_hull_weight_grid > 0)
            ]
        }
        
        x = np.arange(3)
        width = 0.35
        labels = ['Weight = 1', 'Weight = 2', 'Total Valid']
        
        bars1 = ax.bar(x - width/2, stats_data['Original'], width,
                      label='Original', alpha=0.8)
        bars2 = ax.bar(x + width/2, stats_data['Convex Hull'], width,
                      label='Convex Hull', alpha=0.8)
        
        ax.set_ylabel('Number of Points')
        ax.set_title('Point Count Comparison', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height/1000)}k',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
    
    def _plot_quality_metrics(self, ax: plt.Axes) -> None:
        """Plot quality metrics text box."""
        ax.axis('off')
        
        try:
            metrics = self.get_quality_metrics()
            metrics_text = f"""Quality Metrics:

RMS Difference: {metrics['rms_difference']:.2e}
Max Difference: {metrics['max_difference']:.2e}
Mean |Rel. Diff|: {metrics['mean_relative_difference_pct']:.1f}%

Overlap Analysis:
Original: {metrics['original_overlap_points']:,} points
Hull: {metrics['hull_overlap_points']:,} points
Expansion: {metrics['expansion_factor']:.2f}×

Hull Properties:
Area: {metrics['hull_area_deg2']:.1f} deg²
Vertices: {metrics['hull_vertices']}
"""
        except Exception:
            metrics_text = "Convex hull method failed\nNo metrics available"
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    def _add_grid_lines(self, fig: plt.Figure, extent: List[float]) -> None:
        """Add grid lines to image plots."""
        # Get all image axes (first 8 subplots)
        image_axes = [fig.axes[i] for i in range(min(8, len(fig.axes)))]
        
        for ax in image_axes:
            # 2θ grid lines every 5 degrees
            twoth_min, twoth_max = extent[0], extent[1]
            for twoth in range(int(twoth_min//5)*5, int(twoth_max//5)*5+5, 5):
                if twoth_min <= twoth <= twoth_max:
                    ax.axvline(x=twoth, color='white', alpha=0.3, linewidth=0.5)
            
            # γ grid lines every 15 degrees
            gamma_min, gamma_max = extent[3], extent[2]  # Note: reversed due to imshow
            for gamma in range(int(gamma_min//15)*15, int(gamma_max//15)*15+15, 15):
                if gamma_min <= gamma <= gamma_max:
                    ax.axhline(y=gamma, color='white', alpha=0.3, linewidth=0.5)


def load_and_merge_detectors(gfrm_files: List[str], 
                           method: str = 'data_driven',
                           verbose: bool = True) -> AreaDetectorMerger:
    """
    Convenience function to load GFRM files and merge them.
    
    Args:
        gfrm_files: List of paths to GFRM files
        method: Merging method - 'data_driven', 'convex_hull', or 'both'
        verbose: Whether to print progress information
        
    Returns:
        AreaDetectorMerger instance with merged data
    """
    # Load area detectors
    area_detectors = [AreaDetectorImage(gfrm_file) for gfrm_file in gfrm_files]
    
    # Convert to 2θ-γ space
    for i, area_detector in enumerate(area_detectors):
        if verbose:
            print(f"Converting detector {i} to 2θ-γ space...")
        area_detector.convert()
    
    # Create merger and run merging
    merger = AreaDetectorMerger(area_detectors)
    
    if verbose:
        merger.print_detector_info()
    
    if method in ['data_driven', 'both']:
        merger.merge_data_driven(verbose=verbose)
    
    if method in ['convex_hull', 'both']:
        merger.merge_convex_hull(verbose=verbose)
    
    return merger


# Example usage
if __name__ == "__main__":
    # Example with test files
    gfrm_files = [
        r"test\20250709_S_MeO_B01_000.gfrm",
        r"test\20250709_S_MeO_B01_001.gfrm"
    ]
    
    # Load and merge detectors
    merger = load_and_merge_detectors(gfrm_files)
    
    # Create comparison plot
    fig = merger.create_comparison_plot()
    plt.show()
    
    # Print quality metrics
    metrics = merger.get_quality_metrics()
    print("\nQuality Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
