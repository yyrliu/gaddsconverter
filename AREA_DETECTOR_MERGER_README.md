# Area Detector Merger

A Python module for merging multiple GADDS area detector images with advanced overlap handling techniques.

## Overview

This module provides the `AreaDetectorMerger` class that implements two main approaches for merging area detector data:

1. **Data-driven merging**: Uses interpolated detector data coverage to determine boundaries
2. **Convex hull merging**: Creates a convex hull around the overlap region for smooth boundaries

## Features

- **Automatic grid alignment**: Creates a common coordinate system covering all detectors
- **Smart boundary detection**: Handles irregular detector geometries and coverage areas
- **Weight-based normalization**: Proper handling of overlapping regions
- **Quality metrics**: Comprehensive comparison between merging methods
- **Visualization tools**: Built-in plotting functions for analysis
- **Flexible API**: Both convenience functions and step-by-step control

## Installation

The module depends on the following packages:
- numpy
- scipy
- matplotlib
- gadds (from this project)

Install dependencies if needed:
```bash
pip install numpy scipy matplotlib
```

## Quick Start

### Simple Usage

```python
from area_detector_merger import load_and_merge_detectors

# Define your GFRM files
gfrm_files = [
    "detector1.gfrm",
    "detector2.gfrm"
]

# Load and merge detectors using both methods
merger = load_and_merge_detectors(gfrm_files, method='both', verbose=True)

# Create comparison plot
fig = merger.create_comparison_plot()
plt.show()

# Get quality metrics
metrics = merger.get_quality_metrics()
print(metrics)
```

### Advanced Usage

```python
from area_detector_merger import AreaDetectorMerger
from gadds import AreaDetectorImage

# Load area detectors manually
area_detectors = [AreaDetectorImage(file) for file in gfrm_files]

# Convert to 2θ-γ space
for detector in area_detectors:
    detector.convert(n_twoth=512, n_gamma=512)

# Create merger
merger = AreaDetectorMerger(area_detectors)

# Run specific merging method
merged_data, weight_grid, normalized_data = merger.merge_data_driven(verbose=True)

# Or run convex hull method
_, hull_weights, hull_data = merger.merge_convex_hull(verbose=True)

# Get detector information
detector_info = merger.get_detector_info()
```

## Class Reference

### AreaDetectorMerger

Main class for merging area detector images.

#### Constructor
```python
AreaDetectorMerger(area_detectors: List[AreaDetectorImage])
```

#### Key Methods

- `merge_data_driven(verbose=True)`: Data-driven merging approach
- `merge_convex_hull(verbose=True)`: Convex hull merging approach
- `get_detector_info()`: Get information about each detector
- `print_detector_info()`: Print detailed detector information
- `get_quality_metrics()`: Calculate quality metrics comparing methods
- `create_comparison_plot()`: Generate comprehensive comparison visualization

#### Properties

- `area_detectors`: List of input area detector images
- `common_grid`: Common coordinate grid information
- `merged_data`: Raw merged data before normalization
- `weight_grid`: Weight grid for data-driven normalization
- `normalized_data`: Final normalized merged data (data-driven)
- `convex_hull_weight_grid`: Weight grid for convex hull method
- `convex_hull_normalized_data`: Final normalized merged data (convex hull)

### Convenience Functions

```python
load_and_merge_detectors(gfrm_files, method='both', verbose=True)
```

Load GFRM files and merge them using specified method(s).

**Parameters:**
- `gfrm_files`: List of paths to GFRM files
- `method`: 'data_driven', 'convex_hull', or 'both'
- `verbose`: Whether to print progress information

**Returns:** AreaDetectorMerger instance with merged data

## Merging Methods

### Method 1: Data-Driven Approach

- **Boundary Detection**: Uses interpolated detector data coverage to determine boundaries
- **Weight Assignment**: Weight = 1 for single detector coverage, Weight = 2 for actual overlap
- **Characteristics**: Creates irregular but accurate boundaries based on actual data coverage

**Advantages:**
- Accurate representation of actual detector coverage
- Preserves original data boundaries
- No assumptions about detector geometry

**Use when:** You want precise boundaries based on actual data coverage

### Method 2: Convex Hull Approach

- **Boundary Detection**: Creates a convex hull around the original overlap region
- **Weight Assignment**: Weight = 1 outside hull, Weight = 2 for all points inside hull
- **Characteristics**: Creates a smooth, geometric boundary that expands the overlap region

**Advantages:**
- Smooth, regular boundaries
- May reduce artifacts at overlap boundaries
- Mathematically well-defined regions

**Use when:** You prefer smooth boundaries and can tolerate some expansion of the overlap region

## Quality Metrics

The module calculates several quality metrics to compare the two methods:

- **RMS Difference**: Root mean square difference between normalized data
- **Max Difference**: Maximum absolute difference
- **Mean Relative Difference**: Average percentage change
- **Overlap Analysis**: Point counts and expansion factors
- **Hull Properties**: Area and geometric characteristics

## Visualization

The `create_comparison_plot()` method generates a comprehensive 3×4 grid showing:

**Row 1: Weight Grids**
- Original weight grid (data-driven)
- Convex hull weight grid
- Weight difference map
- Overlap region comparison

**Row 2: Normalized Data**
- Data-driven normalized data
- Convex hull normalized data
- Data difference map
- Relative difference (%)

**Row 3: Analysis**
- 1D intensity comparison
- Weight distribution histogram
- Point count statistics
- Quality metrics summary

## Example Output

When running the merger, you'll see output like:

```
=== Method 1: Data-Driven Weight Grid ===
Processing detector 0...
Processing detector 1...
Coverage analysis:
  Detector 0: 95,432 pixels
  Detector 1: 87,256 pixels
  Overlap region: 12,847 pixels
✓ Data-driven merge completed

=== Method 2: Convex Hull Weight Grid ===
Original overlap region: 12,847 points
Convex hull created with 8 vertices
Hull expansion analysis:
  Original overlap: 12,847 points
  Hull overlap: 15,234 points
  Additional points: 2,387 (18.6% increase)
✓ Convex hull merge completed
```

## Files

- `area_detector_merger.py`: Main module with AreaDetectorMerger class
- `improved_area_detector_merger.py`: Enhanced version with better multi-detector support
- `example_merger.py`: Example script demonstrating usage
- `test_scalability.py`: Scalability testing script for multiple detectors
- `AREA_DETECTOR_MERGER_README.md`: This documentation file
- `SCALABILITY_ANALYSIS.md`: Detailed scalability analysis and test results

## Notes

- The module automatically handles coordinate system alignment between detectors
- Grid resolution is set to the finest available from all input detectors
- Normalization is performed as: `Final Data = Raw Data ÷ Weight Grid`
- The convex hull method may expand overlap regions beyond actual data coverage
- Both methods respect detector geometry and physical limits

### Scalability with Multiple Detectors (3+)

The merger has been tested with multiple detectors:

**✅ Data-driven method**: Scales excellently to any number of detectors
- Correctly handles weights up to the number of detectors
- Proper normalization for complex overlap patterns
- Linear performance scaling

**⚠️ Convex hull method**: Has limitations with 3+ detectors
- Current implementation only considers 2-detector overlaps (`weight == 2`)
- May miss regions where 3+ detectors overlap
- Becomes increasingly aggressive with more detectors

**Recommendation**: For 3+ detectors, prefer the data-driven method or use the improved convex hull implementation provided in `improved_area_detector_merger.py`.

See `SCALABILITY_ANALYSIS.md` for detailed analysis and test results.

## Troubleshooting

**Import errors**: Ensure all dependencies are installed and the gadds module is available

**Memory issues**: For large datasets, consider reducing the grid resolution in the `convert()` calls

**Visualization problems**: Matplotlib backend issues can sometimes cause plotting problems - try different backends

**Empty overlap regions**: If detectors don't actually overlap, the convex hull method may fail gracefully

## Contributing

This module was extracted from the Jupyter notebook `merge_gfrm.ipynb` to provide a reusable, well-documented interface for area detector merging.
