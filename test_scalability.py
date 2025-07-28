#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to evaluate scalability of area detector merger with multiple detectors.

This script creates synthetic area detector data to test the merger with more than 2 detectors.
"""

import numpy as np
import matplotlib.pyplot as plt
from area_detector_merger import AreaDetectorMerger
from typing import List


class SyntheticAreaDetector:
    """Create synthetic area detector for testing."""
    
    def __init__(self, center_2theta: float, center_gamma: float, 
                 width_2theta: float = 10.0, width_gamma: float = 15.0,
                 intensity_peak: float = 1000.0):
        """
        Create a synthetic area detector.
        
        Args:
            center_2theta: Center position in 2theta (degrees)
            center_gamma: Center position in gamma (degrees) 
            width_2theta: Width in 2theta direction (degrees)
            width_gamma: Width in gamma direction (degrees)
            intensity_peak: Peak intensity value
        """
        # Create coordinate grids
        n_2theta, n_gamma = 256, 256
        
        # Define coordinate ranges around center
        twoth_min = center_2theta - width_2theta/2
        twoth_max = center_2theta + width_2theta/2
        gamma_min = center_gamma - width_gamma/2
        gamma_max = center_gamma + width_gamma/2
        
        twoth_coords = np.linspace(twoth_min, twoth_max, n_2theta)
        gamma_coords = np.linspace(gamma_min, gamma_max, n_gamma)
        
        # Create 2D intensity pattern (Gaussian-like)
        twoth_mesh, gamma_mesh = np.meshgrid(twoth_coords, gamma_coords, indexing='xy')
        
        # Create intensity with some pattern
        intensity = intensity_peak * np.exp(
            -((twoth_mesh - center_2theta)**2 / (width_2theta/3)**2 + 
              (gamma_mesh - center_gamma)**2 / (width_gamma/3)**2)
        )
        
        # Add some noise and structure
        intensity += np.random.normal(0, intensity_peak * 0.05, intensity.shape)
        intensity = np.maximum(intensity, 0)  # Ensure non-negative
        
        # Store data
        self.data_converted = intensity
        self.indexes = (gamma_coords, twoth_coords)
        
        # Set reasonable limits (in radians for compatibility)
        self.limits = (
            np.deg2rad(twoth_min), np.deg2rad(twoth_max),
            np.deg2rad(gamma_min), np.deg2rad(gamma_max)
        )
        
        # Mock other attributes
        self.alpha = np.deg2rad(center_2theta)
        self.distance = 10.0  # cm
        self.centerXY = (128, 128)
        self.densityXY = (50, 50)
        self.scale = 1.0
        self.offset = 0
        
        # Mock image with header
        class MockImage:
            def __init__(self):
                self.header = {'TITLE': f'Synthetic detector at 2θ={center_2theta:.1f}°, γ={center_gamma:.1f}°'}
        
        self.image = MockImage()


def create_test_detectors(n_detectors: int = 4) -> List[SyntheticAreaDetector]:
    """Create multiple synthetic detectors with overlapping regions."""
    detectors = []
    
    if n_detectors == 3:
        # 3 detectors with pairwise overlaps
        configs = [
            (20.0, 0.0),    # Detector 0: center at (20°, 0°)
            (25.0, 5.0),    # Detector 1: overlaps with 0 and 2
            (30.0, 0.0),    # Detector 2: overlaps with 1
        ]
    elif n_detectors == 4:
        # 4 detectors in a 2x2 pattern
        configs = [
            (20.0, -5.0),   # Bottom left
            (27.0, -5.0),   # Bottom right  
            (20.0, 5.0),    # Top left
            (27.0, 5.0),    # Top right
        ]
    else:
        # Create n detectors in a line with overlaps
        base_2theta = 20.0
        spacing = 6.0  # Overlap since detector width is 10°
        configs = [(base_2theta + i * spacing, 0.0) for i in range(n_detectors)]
    
    for i, (center_2theta, center_gamma) in enumerate(configs):
        detector = SyntheticAreaDetector(
            center_2theta=center_2theta,
            center_gamma=center_gamma,
            intensity_peak=1000 + i * 100  # Slightly different intensities
        )
        detectors.append(detector)
    
    return detectors


def test_merger_scalability():
    """Test the merger with different numbers of detectors."""
    print("="*60)
    print("TESTING AREA DETECTOR MERGER SCALABILITY")
    print("="*60)
    
    test_cases = [2, 3, 4, 5]
    
    for n_detectors in test_cases:
        print(f"\n{'='*50}")
        print(f"TESTING WITH {n_detectors} DETECTORS")
        print(f"{'='*50}")
        
        try:
            # Create synthetic detectors
            detectors = create_test_detectors(n_detectors)
            
            # Create merger
            merger = AreaDetectorMerger(detectors)
            
            # Print detector info
            print("\nDetector configuration:")
            for i, det in enumerate(detectors):
                twoth_range = (det.indexes[1].min(), det.indexes[1].max())
                gamma_range = (det.indexes[0].min(), det.indexes[0].max())
                print(f"  Detector {i}: 2θ=[{twoth_range[0]:.1f}, {twoth_range[1]:.1f}]°, "
                      f"γ=[{gamma_range[0]:.1f}, {gamma_range[1]:.1f}]°")
            
            # Test data-driven merging
            print("\nTesting data-driven merging...")
            merged_data, weight_grid, normalized_data = merger.merge_data_driven(verbose=True)
            
            # Analyze weight distribution
            unique_weights, counts = np.unique(weight_grid[weight_grid > 0], return_counts=True)
            print("\nWeight distribution:")
            for weight, count in zip(unique_weights, counts):
                print(f"  Weight {weight:.0f}: {count:,} pixels")
            
            max_weight = np.max(weight_grid)
            print(f"  Maximum weight: {max_weight:.0f} (max possible: {n_detectors})")
            
            # Test convex hull merging
            print("\nTesting convex hull merging...")
            try:
                _, hull_weight_grid, hull_normalized_data = merger.merge_convex_hull(verbose=True)
                
                # Check if convex hull worked properly
                hull_unique_weights, hull_counts = np.unique(hull_weight_grid[hull_weight_grid > 0], return_counts=True)
                print("Convex hull weight distribution:")
                for weight, count in zip(hull_unique_weights, hull_counts):
                    print(f"  Weight {weight:.0f}: {count:,} pixels")
                
                print(f"✓ Convex hull method completed for {n_detectors} detectors")
                
            except Exception as e:
                print(f"✗ Convex hull method failed: {e}")
            
            # Create simple visualization
            if n_detectors <= 4:  # Only visualize for smaller cases
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                extent = [
                    merger.common_grid['twoth'].min(),
                    merger.common_grid['twoth'].max(),
                    merger.common_grid['gamma'].max(),
                    merger.common_grid['gamma'].min()
                ]
                
                # Plot weight grid
                im1 = axes[0].imshow(weight_grid, cmap='viridis', origin='upper',
                                   extent=extent, aspect='auto')
                axes[0].set_title(f'Weight Grid ({n_detectors} detectors)')
                axes[0].set_xlabel('2θ (degrees)')
                axes[0].set_ylabel('γ (degrees)')
                plt.colorbar(im1, ax=axes[0], label='Weight')
                
                # Plot normalized data
                normalized_display = normalized_data.copy()
                normalized_display[weight_grid == 0] = np.nan
                
                im2 = axes[1].imshow(normalized_display, cmap='viridis', origin='upper',
                                   extent=extent, aspect='auto')
                axes[1].set_title(f'Normalized Data ({n_detectors} detectors)')
                axes[1].set_xlabel('2θ (degrees)')
                axes[1].set_ylabel('γ (degrees)')
                plt.colorbar(im2, ax=axes[1], label='Intensity')
                
                plt.tight_layout()
                plt.show()
            
            print(f"✓ Successfully processed {n_detectors} detectors")
            
        except Exception as e:
            print(f"✗ Failed with {n_detectors} detectors: {e}")
            import traceback
            traceback.print_exc()


def analyze_weight_logic():
    """Analyze the weight logic limitations."""
    print("\n" + "="*60)
    print("ANALYZING WEIGHT LOGIC LIMITATIONS")
    print("="*60)
    
    # Create 3 overlapping detectors
    detectors = create_test_detectors(3)
    merger = AreaDetectorMerger(detectors)
    
    # Run data-driven merging
    merged_data, weight_grid, normalized_data = merger.merge_data_driven(verbose=False)
    
    # Analyze overlap patterns
    print("\nOverlap analysis for 3 detectors:")
    
    # Find different overlap regions
    no_coverage = np.sum(weight_grid == 0)
    single_coverage = np.sum(weight_grid == 1)
    dual_overlap = np.sum(weight_grid == 2)
    triple_overlap = np.sum(weight_grid == 3)
    
    total_pixels = np.sum(weight_grid > 0)
    
    print(f"  No coverage: {no_coverage:,} pixels")
    print(f"  Single detector: {single_coverage:,} pixels ({100*single_coverage/total_pixels:.1f}%)")
    print(f"  Two detectors overlap: {dual_overlap:,} pixels ({100*dual_overlap/total_pixels:.1f}%)")
    print(f"  Three detectors overlap: {triple_overlap:,} pixels ({100*triple_overlap/total_pixels:.1f}%)")
    
    # Test convex hull limitation
    print("\nConvex hull method limitation:")
    print(f"  Current method only considers weight=2 regions: {dual_overlap:,} pixels")
    print(f"  But with 3 detectors, we also have weight=3 regions: {triple_overlap:,} pixels")
    print(f"  Total overlap regions: {dual_overlap + triple_overlap:,} pixels")
    
    if triple_overlap > 0:
        print(f"  ⚠️  Convex hull method will miss {100*triple_overlap/(dual_overlap+triple_overlap):.1f}% of overlap regions!")


if __name__ == "__main__":
    # Test scalability
    test_merger_scalability()
    
    # Analyze limitations
    analyze_weight_logic()
