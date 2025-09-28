# Standard Demographic Classifiers

This directory contains reusable demographic classifiers that can be shared across different regional implementations.

## Overview

The standard classifiers provide common demographic classification logic for:
- **Age**: Classification into age bands (0-9, 10-19, 20-29, etc.)
- **Sex**: Classification by gender (Male, Female)
- **Age/Sex Pyramid**: Combined age and gender classification

## Design Benefits

1. **DRY Principle**: Eliminates code duplication across regions
2. **Consistency**: Ensures identical classification logic across different data sources
3. **Maintainability**: Updates to core logic only need to happen in one place
4. **Extensibility**: Easy to add new regions without reimplementing common functionality
5. **Flexibility**: Allows region-specific customization through inheritance

## Usage Pattern

Regional classifiers inherit from standard classifiers and provide region-specific identification:

```python
from ..standard.age import StandardAgeClassifier

class RegionAgeClassifier(StandardAgeClassifier):
    def __init__(self):
        super().__init__('region_name')  # e.g., 'uk_census', 'unpd'
    
    # Override methods for region-specific customization if needed
```

## Files

- `age.py`: Standard age band classification
- `sex.py`: Standard gender/sex classification  
- `age_sex_pyramid.py`: Standard combined age/sex classification
- `__init__.py`: Module exports

## Customization

While the standard implementations work for most cases, regions can override specific methods:

```python
def get_age_band_labels(self) -> Tuple[List[int], List[str]]:
    """Override for region-specific age bands."""
    # Custom age bands for this region
    bins = [0, 5, 15, 25, 35, 45, 55, 65, 75, float("inf")]
    labels = ["0-4", "5-14", "15-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
    return bins, labels
```

## Integration

Standard classifiers are used by:
- UK Census classifiers (via `uk/*_standard.py` files)
- UN Population Division classifiers (via `unpd/*.py` files)
- Any future regional implementations

See `examples/standard_classifiers_demo.py` for a complete demonstration.