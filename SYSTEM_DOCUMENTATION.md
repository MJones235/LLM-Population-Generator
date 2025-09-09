# Population Generator: Complete System Documentation

## Overview

The Population Generator now features a complete UK demographic classifier suite with flexible data loading capabilities. This documentation covers all components working together as an integrated system.

## System Architecture

### 1. Core Components

- **DemographicClassifier Framework**: Base classes for household and individual-level analysis
- **UK Classifier Suite**: 5 specialized classifiers for UK demographics
- **Flexible Data Loading**: Multi-format data input system with auto-detection
- **Statistics Integration**: Seamless integration with prompt generation and target data

### 2. UK Classifier Suite (Complete)

#### Available Classifiers

1. **UKHouseholdSizeClassifier** - Household size distribution (1-8+ people)
2. **UKSocioeconomicClassifier** - Socioeconomic status analysis
3. **UKAgeSexPyramidClassifier** - Combined age/gender pyramid analysis
4. **UKAgeClassifier** - Age distribution analysis only
5. **UKSexClassifier** - Gender/sex distribution analysis only

#### Usage Example

```python
from population_generator.contrib.classifiers.uk import (
    UKHouseholdSizeClassifier,
    UKAgeSexPyramidClassifier,
    UKAgeClassifier,
    UKSexClassifier
)

# Initialize classifiers
household_classifier = UKHouseholdSizeClassifier()
age_sex_classifier = UKAgeSexPyramidClassifier()
age_classifier = UKAgeClassifier()
sex_classifier = UKSexClassifier()

# Analyze generated households
households = [...]  # Your generated households
household_stats = household_classifier.analyze_households(households)
age_sex_stats = age_sex_classifier.analyze_households(households)
age_stats = age_classifier.analyze_households(households)
sex_stats = sex_classifier.analyze_households(households)
```

### 3. Flexible Data Loading System

#### Supported Formats

**Raw UK Census CSV**
```csv
Categories,Observation
"1 person in household",12345
"2 people in household",11567
"3 people in household",5789
...
```

**Preprocessed CSV**
```csv
Category,Percentage
1,34.2
2,31.9
3,15.9
...
```

**JSON with Metadata**
```json
{
  "metadata": {
    "name": "UK Household Size Distribution 2021",
    "source": "UK Census 2021",
    "year": 2021,
    "description": "Household size distribution from UK Census 2021"
  },
  "data": {
    "1": 29.2,
    "2": 34.8,
    "3": 15.7,
    ...
  }
}
```

#### Data Loading Usage

```python
from population_generator.utils.data_loading import FlexibleDataManager

# Initialize data manager
data_manager = FlexibleDataManager()

# Load any supported format - automatic detection!
data = data_manager.load_target_data("path/to/your/data.csv")
data = data_manager.load_target_data("path/to/your/data.json")

# Get metadata if available
metadata = data_manager.get_metadata("path/to/your/data.json")
print(f"Dataset: {metadata.name} from {metadata.source}")
```

### 4. Integrated Target Data Usage

#### With Statistics Manager

```python
from population_generator.utils.statistics import StatisticsManager

# Initialize with flexible data loading
stats_manager = StatisticsManager(enable_flexible_loading=True)

# Add target data files - any format!
stats_manager.add_target_data_file("raw_census_data.csv")
stats_manager.add_target_data_file("preprocessed_data.csv") 
stats_manager.add_target_data_file("structured_data.json")

# Generate with integrated targets
households = generator.generate_households(
    num_households=100,
    target_data_files=["any_format_data.csv"]
)
```

#### With Population Generator

```python
from population_generator.core.generator import PopulationGenerator
from population_generator.contrib.classifiers.uk import UKHouseholdSizeClassifier

# Set up generator with classifier and flexible data
generator = PopulationGenerator()
generator.prompt_manager.register_classifier('HOUSEHOLD_SIZE_STATS', UKHouseholdSizeClassifier())

# Generate with target data (any format)
households = generator.generate_households(
    num_households=50,
    target_data_files=["uk_census_raw.csv", "preprocessed.csv", "metadata.json"]
)
```

## Advanced Features

### 1. Age/Sex Pyramid Classifier

The `UKAgeSexPyramidClassifier` provides detailed demographic analysis:

- **Age Bands**: 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80+
- **Gender Integration**: Male/Female breakdown for each age band
- **18 Total Categories**: e.g., "Male_0-9", "Female_20-29", etc.

```python
from population_generator.contrib.classifiers.uk.age_sex_pyramid import UKAgeSexPyramidClassifier

classifier = UKAgeSexPyramidClassifier()
distribution = classifier.analyze_households(households)

# Results like:
# {
#   "Male_0-9": 5.2,
#   "Female_0-9": 4.8,
#   "Male_20-29": 12.3,
#   "Female_20-29": 11.8,
#   ...
# }
```

### 2. Individual Age and Sex Classifiers

For focused analysis on single demographic dimensions:

**Age-Only Analysis**
```python
from population_generator.contrib.classifiers.uk.age import UKAgeClassifier

age_classifier = UKAgeClassifier()
age_dist = age_classifier.analyze_households(households)
# {"0-9": 10.0, "10-19": 12.5, "20-29": 24.1, ...}
```

**Sex-Only Analysis**
```python
from population_generator.contrib.classifiers.uk.sex import UKSexClassifier

sex_classifier = UKSexClassifier()
sex_dist = sex_classifier.analyze_households(households)
# {"Male": 49.2, "Female": 49.8, "Non-binary": 0.8, "Unknown": 0.2}
```

### 3. Data Format Auto-Detection

The system automatically detects data format based on:

- **File extension** (.csv, .json)
- **Column patterns** (Categories/Observation, Category/Percentage)
- **Content structure** (metadata presence, value types)
- **Fallback mechanisms** (multiple detection strategies)

### 4. Robust Error Handling

- **Missing columns**: Clear error messages with suggested fixes
- **Invalid data**: Graceful handling with warnings
- **Format mismatches**: Automatic fallback to alternative loaders
- **Empty datasets**: Proper validation and user feedback

## Complete Working Examples

### 1. Multi-Format Target Data Demo

```python
# examples/target_data_integration_demo.py
from population_generator.core.generator import PopulationGenerator
from population_generator.contrib.classifiers.uk import UKHouseholdSizeClassifier

def demonstrate_flexible_targets():
    # Initialize generator
    generator = PopulationGenerator()
    generator.prompt_manager.register_classifier('HOUSEHOLD_SIZE_STATS', UKHouseholdSizeClassifier())
    
    # Test different data formats
    data_files = [
        "raw_uk_census.csv",                     # Raw UK Census format
        "preprocessed.csv",                      # Preprocessed format
        "json_with_metadata.json"               # JSON with metadata
    ]
    
    for data_file in data_files:
        print(f"\n📁 Testing: {data_file}")
        
        # Generate with this target data
        households = generator.generate_households(
            num_households=6,
            target_data_files=[data_file]
        )
        
        # Analyze results
        classifier = UKHouseholdSizeClassifier()
        observed = classifier.analyze_households(households)
        
        print(f"✅ Generated {len(households)} households")
        print(f"📊 Distribution: {observed}")
```

### 2. All-Classifier Analysis Demo

```python
# examples/comprehensive_classifier_demo.py
from population_generator.contrib.classifiers.uk import *

def analyze_all_demographics(households):
    """Analyze households with all available UK classifiers."""
    
    classifiers = {
        'Household Size': UKHouseholdSizeClassifier(),
        'Socioeconomic': UKSocioeconomicClassifier(), 
        'Age-Sex Pyramid': UKAgeSexPyramidClassifier(),
        'Age Distribution': UKAgeClassifier(),
        'Sex Distribution': UKSexClassifier()
    }
    
    results = {}
    for name, classifier in classifiers.items():
        print(f"\n📊 Analyzing: {name}")
        distribution = classifier.analyze_households(households)
        results[name] = distribution
        
        # Print top categories
        sorted_categories = sorted(distribution.items(), 
                                 key=lambda x: x[1], reverse=True)
        for category, percentage in sorted_categories[:5]:
            print(f"  {category}: {percentage:.1f}%")
    
    return results
```

### 3. Custom Data Integration

```python
# examples/custom_data_integration.py
from population_generator.utils.data_loading import FlexibleDataManager

def integrate_custom_data():
    """Show how to integrate your own data formats."""
    
    data_manager = FlexibleDataManager()
    
    # Your custom data files
    files = [
        "your_census_data.csv",
        "your_survey_data.json", 
        "your_preprocessed_stats.csv"
    ]
    
    for file_path in files:
        try:
            # Auto-detect and load
            data = data_manager.load_target_data(file_path)
            metadata = data_manager.get_metadata(file_path)
            
            print(f"✅ Loaded: {metadata.name}")
            print(f"   Source: {metadata.source}")
            print(f"   Categories: {len(data)}")
            print(f"   Sample: {dict(list(data.items())[:3])}")
            
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")
```

## Best Practices

### 1. Data Preparation

- **Raw UK Census**: Use official CSV downloads directly
- **Preprocessed Data**: Ensure Category/Percentage column names
- **JSON Format**: Include metadata for better documentation
- **Data Quality**: Verify percentages sum to ~100%

### 2. Classifier Selection

- **Household Analysis**: Use UKHouseholdSizeClassifier for family structure
- **Demographic Overview**: Use UKAgeSexPyramidClassifier for population structure  
- **Focused Analysis**: Use UKAgeClassifier or UKSexClassifier for specific dimensions
- **Economic Analysis**: Use UKSocioeconomicClassifier for social class distribution

### 3. Integration Patterns

- **Multi-Target Generation**: Use multiple data files for complex targets
- **Progressive Refinement**: Start with basic classifiers, add detailed ones
- **Validation**: Always compare observed vs target distributions
- **Documentation**: Use metadata-rich JSON format for important datasets

### 4. Performance Optimization

- **Data Caching**: FlexibleDataManager caches loaded data automatically
- **Batch Processing**: Generate larger batches for better statistical accuracy
- **Selective Analysis**: Use specific classifiers rather than running all
- **Memory Management**: Clear caches with `force_reload=True` when needed

## Troubleshooting

### Common Issues

**"Could not find category columns"**
- Check CSV column names match expected patterns
- Ensure data file is properly formatted
- Try different loader explicitly

**"No observations found"**  
- Verify data contains non-zero values
- Check for missing or invalid data rows
- Examine raw data format

**"Format not supported"**
- Check file extension (.csv, .json supported)
- Verify internal data structure matches expected format
- Consider preprocessing data to supported format

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Force specific loader
data_manager = FlexibleDataManager()
data = data_manager.loaders[0].load(file_path)  # Try specific loader
```

## System Status

✅ **Complete Components**:
- 5 UK demographic classifiers
- Flexible data loading (3 formats)
- Auto-detection system  
- Statistics integration
- Error handling & validation
- Comprehensive documentation

🚀 **Ready for Production Use**: The system is fully functional and tested with real UK Census data formats.

## Next Steps

1. **Advanced Classifiers**: Add regional, occupational, or custom demographic classifiers
2. **Data Formats**: Support for Excel, XML, or database connections
3. **Visualization**: Integrate with plotting libraries for demographic charts
4. **API Development**: REST API for web-based population generation
5. **Performance**: Optimization for large-scale population generation

The Population Generator is now a complete, flexible system ready for real-world demographic analysis and synthetic population generation!
