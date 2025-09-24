"""Example of using UK Census data preprocessors.

This script demonstrates how to use the UK Census data preprocessors
to convert raw ONS data files into the standardized format expected
by the population generator library.
"""

from pathlib import Path
from population_generator.contrib.data_preprocessors.uk import (
    UKAgePreprocessor,
    UKSexPreprocessor,
    UKHouseholdSizePreprocessor,
    UKHouseholdCompositionPreprocessor
)


def preprocess_uk_census_data():
    """Example of preprocessing UK Census data files."""
    
    # Define paths (adjust these to your actual data files)
    data_dir = Path("examples/data/raw_uk_census")
    output_dir = Path("examples/data/targets")
    
    # Area to focus on (example: Newcastle upon Tyne)
    area_code = "E08000021"
    area_name = "Newcastle upon Tyne"
    
    # Initialize preprocessors
    age_preprocessor = UKAgePreprocessor()
    sex_preprocessor = UKSexPreprocessor()
    household_size_preprocessor = UKHouseholdSizePreprocessor()
    household_composition_preprocessor = UKHouseholdCompositionPreprocessor()
    
    # Process age data
    print("Processing age data...")
    try:
        age_data = age_preprocessor.preprocess_file(
            input_path=data_dir / "age_data.csv",
            output_path=output_dir / "uk_age_distribution.csv",
            area_code=area_code
        )
        
        # Validate against classifier
        is_valid, errors = age_preprocessor.validate_with_classifier(age_data)
        if is_valid:
            print("✓ Age data processed successfully")
            print(f"  Age distribution: {age_data}")
        else:
            print("✗ Age data validation failed:")
            for error in errors:
                print(f"    - {error}")
                
    except Exception as e:
        print(f"✗ Age data processing failed: {e}")
    
    # Process sex data  
    print("\nProcessing sex data...")
    try:
        sex_data = sex_preprocessor.preprocess_file(
            input_path=data_dir / "sex_data.csv",
            output_path=output_dir / "uk_sex_distribution.csv", 
            area_code=area_code
        )
        
        # Validate against classifier
        is_valid, errors = sex_preprocessor.validate_with_classifier(sex_data)
        if is_valid:
            print("✓ Sex data processed successfully")
            print(f"  Sex distribution: {sex_data}")
        else:
            print("✗ Sex data validation failed:")
            for error in errors:
                print(f"    - {error}")
                
    except Exception as e:
        print(f"✗ Sex data processing failed: {e}")
    
    # Process household size data
    print("\nProcessing household size data...")
    try:
        household_size_data = household_size_preprocessor.preprocess_file(
            input_path=data_dir / "household_size_data.csv",
            output_path=output_dir / "uk_household_size.csv",
            area_code=area_code
        )
        
        # Validate against classifier
        is_valid, errors = household_size_preprocessor.validate_with_classifier(household_size_data)
        if is_valid:
            print("✓ Household size data processed successfully")
            print(f"  Household size distribution: {household_size_data}")
        else:
            print("✗ Household size data validation failed:")
            for error in errors:
                print(f"    - {error}")
                
    except Exception as e:
        print(f"✗ Household size data processing failed: {e}")
    
    # Process household composition data
    print("\nProcessing household composition data...")
    try:
        household_composition_data = household_composition_preprocessor.preprocess_file(
            input_path=data_dir / "household_composition_data.csv", 
            output_path=output_dir / "uk_household_composition.csv",
            area_code=area_code
        )
        
        # Validate against classifier
        is_valid, errors = household_composition_preprocessor.validate_with_classifier(household_composition_data)
        if is_valid:
            print("✓ Household composition data processed successfully")
            print(f"  Household composition distribution: {household_composition_data}")
        else:
            print("✗ Household composition data validation failed:")
            for error in errors:
                print(f"    - {error}")
                
    except Exception as e:
        print(f"✗ Household composition data processing failed: {e}")


if __name__ == "__main__":
    preprocess_uk_census_data()