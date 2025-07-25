#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script demonstrating the Area Detector Merger module.

This script shows how to use the AreaDetectorMerger class to merge
GADDS area detector images using both data-driven and convex hull approaches.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from area_detector_merger import AreaDetectorMerger, load_and_merge_detectors


def main():
    """Main example function."""
    # Set up matplotlib for better plots
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # Define GFRM files to merge
    gfrm_files = [
        r"test\20250709_S_MeO_B01_000.gfrm",
        r"test\20250709_S_MeO_B01_001.gfrm"
    ]
    
    print("=== GADDS Area Detector Merging Example ===\n")
    
    # Method 1: Using the convenience function
    print("Method 1: Using convenience function load_and_merge_detectors()")
    merger = load_and_merge_detectors(gfrm_files, method='both', verbose=True)
    
    # Create and show comparison plot
    print("\nCreating comparison plot...")
    fig = merger.create_comparison_plot()
    plt.show()
    
    # Print quality metrics
    print("\n" + "="*50)
    print("QUALITY METRICS")
    print("="*50)
    metrics = merger.get_quality_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'pct' in key:
                print(f"{key}: {value:.2f}%")
            elif value < 1e-3:
                print(f"{key}: {value:.2e}")
            else:
                print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value:,}")
    
    print("\n" + "="*50)
    print("METHOD 2: Step-by-step usage")
    print("="*50)
    
    # Method 2: Step-by-step approach for more control
    from gadds import AreaDetectorImage
    
    # Load area detectors manually
    area_detectors = [AreaDetectorImage(gfrm_file) for gfrm_file in gfrm_files]
    
    # Convert to 2θ-γ space
    for i, area_detector in enumerate(area_detectors):
        print(f"Converting detector {i} to 2θ-γ space...")
        area_detector.convert(n_twoth=512, n_gamma=512)
    
    # Create merger instance
    merger2 = AreaDetectorMerger(area_detectors)
    
    # Print detector information
    print("\nDetector Information:")
    merger2.print_detector_info()
    
    # Run data-driven merging
    print("Running data-driven merging...")
    merged_data, weight_grid, normalized_data = merger2.merge_data_driven(verbose=True)
    
    # Run convex hull merging
    print("\nRunning convex hull merging...")
    _, hull_weight_grid, hull_normalized_data = merger2.merge_convex_hull(verbose=True)
    
    # Create simple comparison plots
    print("\nCreating simple comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Calculate display extent
    extent = [
        merger2.common_grid['twoth'].min(), 
        merger2.common_grid['twoth'].max(),
        merger2.common_grid['gamma'].max(), 
        merger2.common_grid['gamma'].min()
    ]
    
    # Plot weight grids
    im1 = axes[0, 0].imshow(weight_grid, cmap='viridis', origin='upper', 
                           extent=extent, aspect='auto')
    axes[0, 0].set_title('Data-Driven Weight Grid')
    axes[0, 0].set_xlabel('2θ (degrees)')
    axes[0, 0].set_ylabel('γ (degrees)')
    plt.colorbar(im1, ax=axes[0, 0], label='Weight')
    
    im2 = axes[0, 1].imshow(hull_weight_grid, cmap='viridis', origin='upper',
                           extent=extent, aspect='auto')
    axes[0, 1].set_title('Convex Hull Weight Grid')
    axes[0, 1].set_xlabel('2θ (degrees)')
    axes[0, 1].set_ylabel('γ (degrees)')
    plt.colorbar(im2, ax=axes[0, 1], label='Weight')
    
    # Plot normalized data
    normalized_display = normalized_data.copy()
    normalized_display[weight_grid == 0] = np.nan
    
    hull_normalized_display = hull_normalized_data.copy()
    hull_normalized_display[hull_weight_grid == 0] = np.nan
    
    # Use log scale for better visualization
    valid_data = normalized_display[normalized_display > 0]
    if len(valid_data) > 0:
        vmin = np.percentile(valid_data, 5)
        vmax = np.percentile(valid_data, 95)
        if vmin > 0:
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None
    else:
        norm = None
    
    im3 = axes[1, 0].imshow(normalized_display, cmap='viridis', origin='upper',
                           extent=extent, aspect='auto', norm=norm)
    axes[1, 0].set_title('Data-Driven Normalized Data')
    axes[1, 0].set_xlabel('2θ (degrees)')
    axes[1, 0].set_ylabel('γ (degrees)')
    plt.colorbar(im3, ax=axes[1, 0], label='Intensity')
    
    im4 = axes[1, 1].imshow(hull_normalized_display, cmap='viridis', origin='upper',
                           extent=extent, aspect='auto', norm=norm)
    axes[1, 1].set_title('Convex Hull Normalized Data')
    axes[1, 1].set_xlabel('2θ (degrees)')
    axes[1, 1].set_ylabel('γ (degrees)')
    plt.colorbar(im4, ax=axes[1, 1], label='Intensity')
    
    plt.tight_layout()
    plt.show()
    
    # Print final comparison
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    original_overlap = np.sum(weight_grid == 2)
    hull_overlap = np.sum(hull_weight_grid == 2)
    expansion_factor = hull_overlap / original_overlap if original_overlap > 0 else 0
    
    print(f"Original overlap region: {original_overlap:,} pixels")
    print(f"Convex hull overlap region: {hull_overlap:,} pixels")
    print(f"Expansion factor: {expansion_factor:.2f}×")
    
    # Calculate RMS difference
    data_diff = hull_normalized_data - normalized_data
    data_diff[weight_grid == 0] = np.nan
    rms_diff = np.sqrt(np.nanmean(data_diff**2))
    print(f"RMS difference between methods: {rms_diff:.2e}")
    
    print("\n✓ Area detector merging example completed successfully!")


if __name__ == "__main__":
    main()
