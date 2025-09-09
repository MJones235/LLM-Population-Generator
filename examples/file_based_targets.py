"""Example showing how to load target data from files."""

from population_generator import PopulationGenerator
from population_generator.contrib.classifiers.uk import UKHouseholdSizeClassifier, UKHouseholdCompositionClassifier


def age_distribution_calculator(synthetic_df):
    """Calculate age distribution from synthetic data."""
    if 'age' not in synthetic_df.columns:
        return {}
    
    age_bins = [(0, 17, "0-17"), (18, 34, "18-34"), (35, 54, "35-54"), (55, 100, "55+")]
    total = len(synthetic_df)
    
    distribution = {}
    for min_age, max_age, label in age_bins:
        count = len(synthetic_df[(synthetic_df['age'] >= min_age) & (synthetic_df['age'] <= max_age)])
        distribution[label] = round((count / total) * 100, 1) if total > 0 else 0.0
    
    return distribution


def main():
    """Demonstrate loading target data from files."""
    print("📁 Loading Target Data from Files Demo")
    print("=" * 50)
    
    # Initialize generator
    generator = PopulationGenerator(
        data_path="./examples/data", 
        prompts_path="./examples/prompts"
    )
    
    # Register classifiers with target data loaded from files
    print("\\n📊 Registering statistics with file-based targets:")
    
    # Household size from JSON file
    uk_size_classifier = UKHouseholdSizeClassifier()
    generator.prompt_manager.register_household_size_classifier(
        uk_size_classifier,
        target_file="targets/uk_household_sizes.json"
    )
    print("   ✅ Household size stats (JSON file)")
    
    # Household composition from JSON file
    uk_comp_classifier = UKHouseholdCompositionClassifier()
    generator.prompt_manager.register_household_composition_classifier(
        uk_comp_classifier,
        target_file="targets/uk_household_composition.json"  
    )
    print("   ✅ Household composition stats (JSON file)")
    
    # Age distribution from CSV file
    generator.prompt_manager.register_custom_statistic(
        "AGE_STATS",
        "age_distribution",
        age_distribution_calculator,
        target_file="targets/uk_age_distribution.csv"
    )
    print("   ✅ Age distribution stats (CSV file)")
    
    # Create prompt template that uses these placeholders
    template_with_placeholders = '''Generate realistic households for {LOCATION}.

Current Performance vs Targets:
{HOUSEHOLD_SIZE_STATS}

{HOUSEHOLD_COMPOSITION_STATS}

{AGE_STATS}

Please generate households that help improve alignment with target distributions.

Return JSON array: [{"name": "string", "age": number, "gender": "string", "relationship": "string"}]'''
    
    print("\\n📝 Example prompt template:")
    print("-" * 30)
    print(template_with_placeholders[:300] + "...")
    
    print("\\n💡 Key Benefits:")
    print("   • Target data managed in separate files")
    print("   • Easy to update targets without code changes")
    print("   • Supports JSON and CSV formats")
    print("   • Automatic loading and validation")
    print("   • Mix file-based and programmatic target data")
    
    print("\\n📁 File structure:")
    print("   examples/data/targets/")
    print("   ├── uk_household_sizes.json")
    print("   ├── uk_household_composition.json")
    print("   └── uk_age_distribution.csv")


if __name__ == "__main__":
    main()
