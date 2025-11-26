# psd_utilities
psd_utilities is a Python package containing utility functions for robust, reproducible processing and manipulation of particle size distribution (PSD) data. It is designed for scientific and engineering applications where data integrity and clarity are paramount.

## Features
Deduplication and Sorting: Ensure PSD arrays have unique, sorted sieve sizes.
Validation: Check for monotonicity, correct boundary values, and normalisation of both cumulative and range PSDs.
Conversion Utilities: Seamlessly convert between passing/retained and cumulative/range forms.
Reindexing: Align multiple PSDs to a common set of sieves, handling missing data gracefully.
Boundary Filling: Fill inactive regions at the top and bottom of reindexed PSDs with zeros for reliable interpolation.
Interpolation: Fill missing values in reindexed PSDs using monotonic PCHIP interpolation for physically plausible results.

## Installation
Requires Python 3.12 or newer.

Install via pip:

pip install psd_utilities

## Dependencies
numpy (version 2.3 or newer)
scipy (version 1.14.1 or newer)

## Usage
The core functions operate on NumPy arrays of shape (n, 2), where the first column contains sieve sizes and the second contains fractions (either cumulative or range, passing or retained).

### Example Workflow
```
import numpy as np
from psd_utilities import (
    normalise_psd,
    reindex_psds,
    fill_top_and_bottom_of_reindexed_range_psds_with_0,
    interpolate_reindexed_filled_range_psds,
)

# Example PSDs
psd1 = np.array([[2.36, 0.0], [4.75, 0.3], [9.5, 0.7], [19.0, 0.0]])
psd2 = np.array([[4.75, 0.2], [9.5, 0.5], [19.0, 0.3]])

# Normalise PSDs
psd1_norm = normalise_psd(psd1, cumulative=False, passing=True)
psd2_norm = normalise_psd(psd2, cumulative=False, passing=True)

# Reindex to common sieves
all_sieves, aligned_psds = reindex_psds([psd1_norm, psd2_norm])

# Fill boundaries with zeros
filled_psds = fill_top_and_bottom_of_reindexed_range_psds_with_0(aligned_psds)

# Interpolate missing values
interpolated_psds = interpolate_reindexed_filled_range_psds(psds=filled_psds, sieves=all_sieves)
```

## API Reference
All functions are fully documented in the source code. Key functions include:

```
deduplicate_sort_sieves(psd, material_name='material')
check_cumulative_psd(psd_unique_sorted, passing, material_name='material')
check_range_psd(psd_unique_sorted, material_name='material')
convert_between_retained_and_passing(psd_unique_sorted, cumulative, copy=True)
convert_cumulative_to_range(psd_unique_sorted, passing, copy=True)
convert_range_to_cumulative(psd_unique_sorted, passing, copy=True)
normalise_psd(psd, cumulative, passing, material_name=None, _prepend_zero=False)
reindex_psds(psds)
fill_top_and_bottom_of_reindexed_range_psds_with_0(psds, copy=True)
interpolate_reindexed_filled_range_psds(psds, sieves, copy=True)
```

See the docstrings in the source code for detailed descriptions, parameters, and examples.

## Error Handling
All validation errors are raised as PSDError (defined in exceptions.py). Ensure you catch this exception when processing user-supplied or external data.

## Contributing & Support
This package is maintained by Milan Kundra.

For bug reports, feature requests, or support, please contact the author or use the issue tracker if available.

## Licence
This project is licensed under the MIT Licence.