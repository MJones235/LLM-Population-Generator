# UK Census Data Preprocessors

This module contains preprocessors for converting UK Census data from the standard ONS (Office for National Statistics) CSV format to the standardized format expected by the Population Generator library.

## Overview

The UK Census data preprocessors handle the complexity of parsing UK Census data files and converting them to the simple `category,percentage` format expected by the `DataLoader` class.

## Features

- **Consistent Mappings**: Uses the same category mappings as the corresponding UK classifiers to ensure synthetic and target data align
- **Validation**: Built-in validation against classifier categories to catch errors early
- **Area Filtering**: Filter data by local authority code or name
- **Standard Output**: Outputs data in the exact format expected by `DataLoader`

## Available Preprocessors

### UKAgePreprocessor
Converts UK Census age data (18 age categories) to standardized age bands.

**Input format**: UK Census age data with columns:
- `Upper tier local authorities Code`
- `Upper tier local authorities`
- `Age (18 categories) Code`
- `Age (18 categories)` 
- `Observation`

**Output format**: CSV with `category,percentage` columns using age bands like "0-9", "10-19", etc.

### UKSexPreprocessor
Converts UK Census sex/gender data (2 categories) to standardized format.

**Input format**: UK Census sex data with columns:
- `Upper tier local authorities Code`
- `Upper tier local authorities`
- `Sex (2 categories) Code`
- `Sex (2 categories)`
- `Observation`

**Output format**: CSV with `category,percentage` columns using "Male", "Female" categories.

### UKHouseholdSizePreprocessor
Converts UK Census household size data (9 categories) to standardized format.

### UKHouseholdCompositionPreprocessor  
Converts UK Census household composition data (8 categories) to standardized format.

## Basic Usage

```python
from population_generator.contrib.data_preprocessors.uk import UKAgePreprocessor
from population_generator.utils.data_loading import DataLoader

# Initialize preprocessor
preprocessor = UKAgePreprocessor()

# Preprocess raw UK Census data
preprocessor.preprocess_file(
    input_path="raw_data/age_data.csv",
    output_path="targets/uk_age_distribution.csv",
    area_code="E08000021"  # Optional: filter by area
)

# Load processed data for use in population generation
data_loader = DataLoader()
age_targets = data_loader.load_target_data("targets/uk_age_distribution.csv")
```

## Advanced Usage

### Area Filtering

Filter data for a specific local authority:

```python
# By area code
preprocessor.preprocess_file(
    input_path="age_data.csv",
    output_path="newcastle_age.csv", 
    area_code="E08000021"
)

# By area name
preprocessor.preprocess_file(
    input_path="age_data.csv",
    output_path="newcastle_age.csv",
    area_name="Newcastle upon Tyne"
)
```

### Validation

Validate processed data against classifier categories:

```python
result = preprocessor.preprocess_file(input_path, output_path)

# Validate the data
is_valid, errors = preprocessor.validate_with_classifier(result)

if not is_valid:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
```

## Input Data Format

The preprocessors expect UK Census data in the standard ONS CSV format. Here's an example of the age data format:

```csv
Upper tier local authorities Code,Upper tier local authorities,Age (18 categories) Code,Age (18 categories),Observation
E08000021,Newcastle upon Tyne,1,Aged 4 years and under,15651
E08000021,Newcastle upon Tyne,2,Aged 5 to 9 years,16704
E08000021,Newcastle upon Tyne,3,Aged 10 to 14 years,17321
...
```

## Output Data Format

All preprocessors output data in the standardized format expected by the `DataLoader`:

```csv
category,percentage
0-9,10.8
10-19,15.6
20-29,18.2
30-39,16.4
...
```

## Integration with Population Generator

The preprocessed data can be directly used with the population generator:

```python
from population_generator.core.generator import PopulationGenerator
from population_generator.contrib.classifiers.uk import UKAgeClassifier

# Load preprocessed target data
data_loader = DataLoader() 
age_targets = data_loader.load_target_data("targets/uk_age_distribution.csv")

# Configure population generator
generator = PopulationGenerator()
generator.add_classifier(
    UKAgeClassifier(),
    target_data=age_targets
)

# Generate population
population = generator.generate(population_size=1000)
```

## Category Mappings

The preprocessors use category mappings that align with the corresponding classifiers:

### Age Categories
- UK Census "Aged 4 years and under" + "Aged 5 to 9 years" → "0-9"  
- UK Census "Aged 10 to 14 years" + "Aged 15 to 19 years" → "10-19"
- And so on...

### Sex Categories
- UK Census "Female" → "Female"
- UK Census "Male" → "Male"

This ensures that synthetic population statistics will match the target data when using the corresponding classifiers.

## Error Handling

The preprocessors include comprehensive error handling:

- **File validation**: Checks input files exist and have correct format
- **Column validation**: Verifies expected columns are present
- **Data validation**: Ensures percentages sum to ~100% 
- **Category validation**: Checks against classifier categories
- **Area filtering**: Handles missing area codes/names gracefully

## Contributing

To add preprocessors for new UK Census variables:

1. Extend `UKCensusPreprocessor` 
2. Implement `_build_category_mapping()` and `_get_expected_columns()`
3. Ensure mappings align with corresponding classifier
4. Add comprehensive tests
5. Update this documentation