#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Area Detector Merger with better scalability for multiple detectors.

This module fixes the limitations found in the original implementation:
1. Convex hull method now considers all overlap regions (weight >= 2)
2. Better handling of multi-detector overlaps
3. More sophisticated boundary detection for complex geometries
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from typing import List, Tuple, Dict, Any
from gadds import AreaDetectorImage


class ImprovedAreaDetectorMerger:
    """
    Improved version of AreaDetectorMerger with better multi-detector support.
    
    Key improvements:
    - Convex hull method considers all overlap regions (weight >= 2)
    - Better statistics for multi-detector scenarios
    - More robust boundary detection
    - Proper handling of complex overlap patterns
    """
    
    def __init__(self, area_detectors: List[AreaDetectorImage]):
        """Initialize the improved merger."""
        self.area_detectors = area_detectors
        self.n_detectors = len(area_detectors)
        self.common_grid = {}
        self.merged_data = None
        self.weight_grid = None
        self.normalized_data = None
        self.detector_coverage_masks = []
        
        # Enhanced convex hull attributes
        self.convex_hull_weight_grid = None
        self.convex_hull_normalized_data = None
        self.hull_points = None
        self.hull_area = None
        self.overlap_threshold = 2  # Minimum weight to consider as overlap
        
        self._setup_common_grid()
    
    def _setup_common_grid(self) -> None:
        """Create a common coordinate grid that covers all detectors."""
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
        """Data-driven merging approach (same as original, works well)."""
        if verbose:
            print(f"=== Data-Driven Merging ({self.n_detectors} detectors) ===")
        
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
            self._print_coverage_analysis()
        
        return self.merged_data, self.weight_grid, self.normalized_data
    
    def merge_convex_hull_improved(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Improved convex hull merging that handles multiple detectors properly.
        
        Key improvements:
        - Considers all overlap regions (weight >= 2) instead of just weight == 2
        - Better handling of complex multi-detector overlaps
        - More conservative expansion in high-overlap scenarios
        """
        if verbose:
            print(f"=== Improved Convex Hull Merging ({self.n_detectors} detectors) ===")
        
        # First ensure we have data-driven results
        if self.weight_grid is None:
            self.merge_data_driven(verbose=False)
        
        # Find all points where weight >= 2 (any overlap region)
        overlap_mask = self.weight_grid >= self.overlap_threshold
        overlap_points_indices = np.where(overlap_mask)
        overlap_gamma_indices = overlap_points_indices[0]  # Row indices (γ)
        overlap_twoth_indices = overlap_points_indices[1]  # Column indices (2θ)
        
        # Convert indices to actual coordinates
        overlap_gamma_coords = self.common_grid['gamma'][overlap_gamma_indices]
        overlap_twoth_coords = self.common_grid['twoth'][overlap_twoth_indices]
        
        if verbose:
            total_overlap_points = len(overlap_gamma_coords)
            print(f"All overlap regions (weight >= {self.overlap_threshold}): {total_overlap_points:,} points")
            if total_overlap_points > 0:
                print(f"  2θ range: [{np.min(overlap_twoth_coords):.2f}, {np.max(overlap_twoth_coords):.2f}]°")
                print(f"  γ range: [{np.min(overlap_gamma_coords):.2f}, {np.max(overlap_gamma_coords):.2f}]°")
        
        # Initialize with copy of original weight grid
        self.convex_hull_weight_grid = self.weight_grid.copy()
        
        if len(overlap_gamma_coords) > 3:  # Need at least 3 points for hull
            try:
                # Create convex hull around all overlap points
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
                
                # Enhanced weight assignment strategy
                self._apply_improved_hull_weights(inside_hull_2d, verbose)
                
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
            print(f"✓ Improved convex hull merge {'completed' if hull_success else 'failed'}")
            if hull_success:
                self._print_hull_statistics()
        
        return self.merged_data, self.convex_hull_weight_grid, self.convex_hull_normalized_data
    
    def _apply_improved_hull_weights(self, inside_hull_2d: np.ndarray, verbose: bool = True) -> None:
        """
        Apply improved weight assignment strategy for convex hull method.
        
        Strategy:
        1. Preserve existing high-weight regions (weight >= 3)
        2. For points inside hull with weight = 1, set to weight = 2
        3. For points inside hull with weight >= 2, preserve original weight
        """
        points_with_coverage = self.convex_hull_weight_grid > 0
        points_in_hull = inside_hull_2d & points_with_coverage
        
        # Count original overlap regions by weight
        original_weight_counts = {}
        for w in range(1, self.n_detectors + 1):
            count = np.sum(self.weight_grid == w)
            if count > 0:
                original_weight_counts[w] = count
        
        # Strategy: Only modify weight=1 regions inside hull, preserve higher weights
        single_detector_in_hull = points_in_hull & (self.convex_hull_weight_grid == 1)
        self.convex_hull_weight_grid[single_detector_in_hull] = 2
        
        # Calculate statistics
        new_weight_counts = {}
        for w in range(1, self.n_detectors + 1):
            count = np.sum(self.convex_hull_weight_grid == w)
            if count > 0:
                new_weight_counts[w] = count
        
        if verbose:
            print("Weight modification analysis:")
            print("  Original → Hull method")
            for w in range(1, self.n_detectors + 1):
                orig_count = original_weight_counts.get(w, 0)
                new_count = new_weight_counts.get(w, 0)
                if orig_count > 0 or new_count > 0:
                    change = new_count - orig_count
                    if change != 0:
                        print(f"    Weight {w}: {orig_count:,} → {new_count:,} ({change:+,})")
                    else:
                        print(f"    Weight {w}: {orig_count:,} (unchanged)")
    
    def _print_coverage_analysis(self) -> None:
        """Print detailed coverage analysis for multiple detectors."""
        total_pixels = np.sum(self.weight_grid > 0)
        print("Coverage analysis:")
        
        # Individual detector coverage
        for i, mask in enumerate(self.detector_coverage_masks):
            print(f"  Detector {i}: {np.sum(mask):,} pixels")
        
        # Weight distribution
        print("Weight distribution:")
        for w in range(1, self.n_detectors + 1):
            count = np.sum(self.weight_grid == w)
            if count > 0:
                percentage = 100 * count / total_pixels
                print(f"  Weight {w}: {count:,} pixels ({percentage:.1f}%)")
        
        # Overlap summary
        overlap_pixels = np.sum(self.weight_grid >= 2)
        if overlap_pixels > 0:
            overlap_percentage = 100 * overlap_pixels / total_pixels
            print(f"  Total overlap: {overlap_pixels:,} pixels ({overlap_percentage:.1f}%)")
        
        print(f"  Total coverage: {total_pixels:,} pixels")
        print("✓ Data-driven merge completed")
    
    def _print_hull_statistics(self) -> None:
        """Print detailed statistics for hull method."""
        # Calculate expansion statistics
        original_overlaps = {}
        hull_overlaps = {}
        
        for w in range(2, self.n_detectors + 1):
            original_overlaps[w] = np.sum(self.weight_grid == w)
            hull_overlaps[w] = np.sum(self.convex_hull_weight_grid == w)
        
        total_original_overlap = sum(original_overlaps.values())
        total_hull_overlap = sum(hull_overlaps.values())
        
        if total_original_overlap > 0:
            total_expansion = total_hull_overlap - total_original_overlap
            expansion_pct = 100 * total_expansion / total_original_overlap
            
            print("Hull method statistics:")
            print(f"  Original total overlap: {total_original_overlap:,} pixels")
            print(f"  Hull total overlap: {total_hull_overlap:,} pixels")
            print(f"  Net change: {total_expansion:+,} pixels ({expansion_pct:+.1f}%)")
    
    def get_detailed_quality_metrics(self) -> Dict[str, Any]:
        """Get comprehensive quality metrics for multi-detector scenarios."""
        if self.normalized_data is None or self.convex_hull_normalized_data is None:
            raise ValueError("Both merging methods must be run before calculating metrics")
        
        # Basic difference metrics
        data_difference = self.convex_hull_normalized_data - self.normalized_data
        data_difference[self.weight_grid == 0] = np.nan
        
        relative_diff = np.divide(
            data_difference, 
            self.normalized_data,
            out=np.zeros_like(data_difference),
            where=self.normalized_data != 0
        ) * 100
        relative_diff[self.weight_grid == 0] = np.nan
        
        # Enhanced metrics for multi-detector
        metrics = {
            'n_detectors': self.n_detectors,
            'rms_difference': np.sqrt(np.nanmean(data_difference**2)),
            'max_difference': np.nanmax(np.abs(data_difference)),
            'mean_relative_difference_pct': np.nanmean(np.abs(relative_diff)),
            'hull_area_deg2': self.hull_area if self.hull_area is not None else 0,
            'hull_vertices': len(self.hull_points) if self.hull_points is not None else 0,
        }
        
        # Weight distribution analysis
        for w in range(1, self.n_detectors + 1):
            original_count = np.sum(self.weight_grid == w)
            hull_count = np.sum(self.convex_hull_weight_grid == w)
            metrics[f'original_weight_{w}'] = original_count
            metrics[f'hull_weight_{w}'] = hull_count
            if original_count > 0:
                metrics[f'weight_{w}_change_pct'] = 100 * (hull_count - original_count) / original_count
        
        # Overall overlap analysis
        original_total_overlap = np.sum(self.weight_grid >= 2)
        hull_total_overlap = np.sum(self.convex_hull_weight_grid >= 2)
        metrics['original_total_overlap'] = original_total_overlap
        metrics['hull_total_overlap'] = hull_total_overlap
        
        if original_total_overlap > 0:
            metrics['total_overlap_expansion_factor'] = hull_total_overlap / original_total_overlap
        else:
            metrics['total_overlap_expansion_factor'] = 0
        
        return metrics


def test_improved_merger():
    """Test the improved merger with multiple detectors."""
    print("="*60)
    print("TESTING IMPROVED AREA DETECTOR MERGER")
    print("="*60)
    
    # Import the synthetic detector creator from the test file
    import sys
    sys.path.append('.')
    from test_scalability import create_test_detectors
    
    test_cases = [3, 4, 5]
    
    for n_detectors in test_cases:
        print(f"\n{'='*50}")
        print(f"TESTING IMPROVED MERGER WITH {n_detectors} DETECTORS")
        print(f"{'='*50}")
        
        # Create synthetic detectors
        detectors = create_test_detectors(n_detectors)
        
        # Test improved merger
        merger = ImprovedAreaDetectorMerger(detectors)
        
        # Run both methods
        merger.merge_data_driven(verbose=True)
        merger.merge_convex_hull_improved(verbose=True)
        
        # Get detailed metrics
        metrics = merger.get_detailed_quality_metrics()
        
        print("\nDetailed Quality Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'pct' in key or 'factor' in key:
                    print(f"  {key}: {value:.2f}")
                elif value < 1e-3:
                    print(f"  {key}: {value:.2e}")
                else:
                    print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value:,}")


if __name__ == "__main__":
    test_improved_merger()
