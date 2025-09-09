#!/usr/bin/env python3
"""
Test Target Data Integration with Classifiers

This test demonstrates the improved target data system where:
1. Classifiers are registered with the generator
2. Target data files are automatically connected to classifiers
3. Prompts show both current and target distributions in comparison format
"""

from population_generator.core.generator import PopulationGenerator
from population_generator.contrib.classifiers.uk import UKHouseholdSizeClassifier
from pathlib import Path

def test_target_integration():
    """Test the complete target data integration."""
    
    print("🧪 Testing Target Data Integration")
    print("=" * 50)
    
    # Initialize generator
    generator = PopulationGenerator()
    
    # Register classifier - this connects it to the statistics system
    household_classifier = UKHouseholdSizeClassifier()
    generator.prompt_manager.register_classifier(
        placeholder="HOUSEHOLD_SIZE_STATS",
        classifier=household_classifier,
        target_file="examples/data/formats/raw_uk_census.csv"
    )
    
    print("\n📊 Testing target data loading:")
    
    # Check that target data was loaded
    stats_manager = generator.prompt_manager.statistics_manager
    print(f"   Registered providers: {list(stats_manager.providers.keys())}")
    print(f"   Placeholder mappings: {stats_manager.placeholder_mappings}")
    
    # Test with a sample prompt containing the placeholder
    test_prompt = """Generate a realistic UK household.

Target household size distribution:
{HOUSEHOLD_SIZE_STATS}

Create a household that helps achieve these target percentages.
Return JSON: [{"name": "string", "age": number, "gender": "string"}]"""
    
    print(f"\n📝 Original prompt template:")
    print(f"   {test_prompt[:100]}...")
    
    # Test first batch (no current data yet)
    processed_prompt = stats_manager.replace_placeholders_in_prompt(
        test_prompt,
        synthetic_df=None,
        format_type="comparison"
    )
    
    print(f"\n🔄 First batch prompt (no current data):")
    print(f"   {processed_prompt[:150]}...")
    
    # Create some sample synthetic data for testing
    import pandas as pd
    
    sample_households = [
        {"people": [{"name": "John", "age": 35, "gender": "Male"}]},  # Size 1
        {"people": [{"name": "Sarah", "age": 28, "gender": "Female"}, {"name": "Mike", "age": 32, "gender": "Male"}]},  # Size 2
        {"people": [{"name": "Emma", "age": 45, "gender": "Female"}]},  # Size 1
    ]
    
    # Convert to DataFrame format
    synthetic_df = pd.DataFrame(
        [
            dict(**person, household_id=household_idx) 
            for household_idx, household in enumerate(sample_households) 
            for person in household['people']
        ]
    )
    
    print(f"\n📈 Sample synthetic data: {len(synthetic_df)} individuals in {len(sample_households)} households")
    
    # Test second batch (with current data for comparison)
    processed_prompt = stats_manager.replace_placeholders_in_prompt(
        test_prompt,
        synthetic_df=synthetic_df,
        format_type="comparison"
    )
    
    print(f"\n🔄 Second batch prompt (with comparison):")
    print(processed_prompt)
    
    print(f"\n✅ Target data integration test complete!")

if __name__ == "__main__":
    test_target_integration()
