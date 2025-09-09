# Example Data Directory

This directory contains all example data files organized by purpose.

## 📁 Directory Structure

```
data/
├── formats/                    # Data format examples
│   ├── raw_uk_census.csv      # Raw UK Census format example
│   ├── preprocessed.csv       # Clean percentage format
│   └── json_with_metadata.json # JSON format with metadata
├── targets/                   # Target data for generation demos
│   ├── uk_household_sizes.json    # Household size targets
│   ├── uk_age_distribution.csv    # Age distribution targets
│   └── uk_household_composition.json # Household composition targets
└── schemas/                   # JSON schemas for validation
    └── household_basic.json   # Basic household schema
```

## 🎯 Purpose of Each Directory

### `formats/` - Data Format Examples
Shows exactly how each supported data format should be structured:
- **`raw_uk_census.csv`** - Direct from UK Census with codes and observations
- **`preprocessed.csv`** - Simple Category/Percentage format  
- **`json_with_metadata.json`** - Structured JSON with rich metadata

These files are used by demos to show the FlexibleDataManager's auto-detection capabilities.

### `targets/` - Target Data for Demos  
Contains target distributions that demos can use to guide population generation:
- Small, focused datasets for specific demographic aspects
- Used by StatisticsManager for target vs observed comparisons
- Shows how real target data integrates with generation

### `schemas/` - JSON Schemas
Validation schemas for generated household data:
- Ensures LLM outputs match expected structure
- Used by PopulationGenerator for validation

## 📊 Data Size & Licensing

All files are small (~28KB total) and either:
- **Derived/aggregated**: Based on public UK Census data but summarized
- **Synthetic**: Created for demonstration purposes
- **Public domain**: No licensing restrictions

This keeps the package lightweight while providing complete working examples.

## 🚀 Usage in Your Code

```python
from population_generator.utils.data_loading import FlexibleDataManager

# Load any format automatically
data_manager = FlexibleDataManager()
data = data_manager.load_target_data("examples/data/formats/raw_uk_census.csv")

# Or use in generation
from population_generator.core.generator import PopulationGenerator
generator = PopulationGenerator(data_path="examples/data/formats")
```

The data organization makes it clear what each file's purpose is and how to use the flexible loading system!
