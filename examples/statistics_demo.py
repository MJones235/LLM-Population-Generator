"""Example demonstrating automatic statistics feedback in prompts.

This demo shows how the new statistics system automatically:
1. Registers statistics providers (classifiers + custom functions)  
2. Replaces placeholders in prompts with target vs observed comparisons
3. Updates statistics after each generation batch for real-time feedback
"""

import pandas as pd
from population_generator import PopulationGenerator
from population_generator.llm.base import BaseLLM
from population_generator.contrib.classifiers.uk import UKHouseholdSizeClassifier, UKHouseholdCompositionClassifier


class MockLLM(BaseLLM):
    """Mock LLM for demonstration."""
    
    def __init__(self):
        self.model_name = "mock-llm"
        self.is_local = False
    
    def generate_text(self, prompt, timeout=30):
        return '''[
            {"name": "John Smith", "age": 42, "gender": "Male", "relationship": "Head"},
            {"name": "Sarah Smith", "age": 38, "gender": "Female", "relationship": "Spouse"},
            {"name": "Emma Smith", "age": 12, "gender": "Female", "relationship": "Child"}
        ]'''
    
    def get_model_metadata(self):
        return {"model": self.model_name, "type": "mock"}


def age_distribution_calculator(synthetic_df: pd.DataFrame) -> dict:
    """Custom function to calculate age distribution."""
    if 'age' not in synthetic_df.columns:
        return {}
    
    age_bins = [(0, 17, "0-17"), (18, 34, "18-34"), (35, 54, "35-54"), (55, 100, "55+")]
    total = len(synthetic_df)
    
    distribution = {}
    for min_age, max_age, label in age_bins:
        count = len(synthetic_df[(synthetic_df['age'] >= min_age) & (synthetic_df['age'] <= max_age)])
        distribution[label] = round((count / total) * 100, 1) if total > 0 else 0.0
    
    return distribution


def gender_distribution_calculator(synthetic_df: pd.DataFrame) -> dict:
    """Custom function to calculate gender distribution."""
    if 'gender' not in synthetic_df.columns:
        return {}
    
    gender_counts = synthetic_df['gender'].value_counts()
    total = len(synthetic_df)
    
    return {
        gender: round((count / total) * 100, 1) 
        for gender, count in gender_counts.items()
    } if total > 0 else {}


def main():
    """Demonstrate automatic statistics feedback system."""
    print("🏠 Automatic Statistics Feedback Demo")
    print("=" * 50)
    print("This demo shows how statistics placeholders are automatically")
    print("updated after each batch to guide LLM generation toward targets.")
    print()
    
    # Step 1: Initialize generator
    print("🔧 Step 1: Initialize Population Generator")
    generator = PopulationGenerator(
        data_path="./examples/data",
        prompts_path="./examples/prompts"
    )
    print("   ✅ Generator initialized")
    
    # Step 2: Register statistics providers with target data
    print("\n📊 Step 2: Register Statistics Providers")
    
    # Register household size classifier with target data
    uk_size_classifier = UKHouseholdSizeClassifier()
    target_size_data = {1: 30.0, 2: 35.0, 3: 15.0, 4: 12.0, 5: 5.0, 6: 2.0, 7: 1.0, 8: 1.0}
    
    generator.prompt_manager.register_household_size_classifier(
        uk_size_classifier, 
        target_data=target_size_data
    )
    print("   ✅ Household size statistics registered")
    
    # Register household composition classifier
    uk_comp_classifier = UKHouseholdCompositionClassifier()
    target_comp_data = {
        "single_person": 30.0,
        "couple_no_children": 25.0, 
        "couple_with_children": 25.0,
        "single_parent": 10.0,
        "other": 10.0
    }
    
    generator.prompt_manager.register_household_composition_classifier(
        uk_comp_classifier,
        target_data=target_comp_data
    )
    print("   ✅ Household composition statistics registered")
    
    # Register custom age statistics
    target_age_data = {"0-17": 20.0, "18-34": 25.0, "35-54": 35.0, "55+": 20.0}
    generator.prompt_manager.register_custom_statistic(
        "AGE_STATS", 
        "age_distribution",
        age_distribution_calculator,
        target_data=target_age_data
    )
    print("   ✅ Custom age statistics registered")
    
    # Register custom gender statistics  
    target_gender_data = {"Male": 49.0, "Female": 51.0}
    generator.prompt_manager.register_custom_statistic(
        "GENDER_STATS",
        "gender_distribution", 
        gender_distribution_calculator,
        target_data=target_gender_data
    )
    print("   ✅ Custom gender statistics registered")
    
    # Step 3: Create prompt template with statistics placeholders
    print("\n📝 Step 3: Create Prompt Template with Placeholders")
    prompt_with_stats = '''Generate a realistic household for {LOCATION}.

{HOUSEHOLD_SIZE_STATS}

{HOUSEHOLD_COMPOSITION_STATS}

{AGE_STATS}

{GENDER_STATS}

Based on the above statistics, generate household members that help achieve the target distributions.

Return as JSON array:
[
    {"name": "string", "age": number, "gender": "string", "relationship": "string"}
]'''
    print("   ✅ Prompt template created with 4 statistics placeholders")
    
    # Define schema
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "gender": {"type": "string"},
                "relationship": {"type": "string"}
            },
            "required": ["name", "age", "gender", "relationship"]
        }
    }
    
    # Step 4: Demonstrate automatic statistics feedback
    print("\\n🚀 Step 4: Generate Households with Automatic Feedback")
    print("The system will:")
    print("   • Start with empty placeholders (no data yet)")
    print("   • Generate first batch of households")
    print("   • Compute statistics from generated data")
    print("   • Update placeholders with target vs observed comparisons")
    print("   • Continue generation with improved guidance")
    print()
    
    households = generator.generate_households(
        n_households=6,              # Generate 6 households total
        model=MockLLM(),            # Use mock LLM for demo
        base_prompt=prompt_with_stats,  # Template with placeholders
        schema=schema,              # JSON validation schema
        location="Manchester",      # Target location
        batch_size=2               # Process 2 households per batch
    )
    
    # Step 5: Analyze results
    print(f"\\n📊 Step 5: Results Analysis")
    print(f"   ✅ Generated {len(households)} households successfully")
    
    # Convert to DataFrame for manual statistics demonstration
    all_people = []
    for i, household in enumerate(households):
        for person in household:
            person_data = person.copy()
            person_data['household_id'] = i
            all_people.append(person_data)
    
    synthetic_df = pd.DataFrame(all_people)
    print(f"   📈 Total people generated: {len(synthetic_df)}")
    print(f"   🏠 Average household size: {len(synthetic_df) / len(households):.1f}")
    
    # Show what the final prompt looked like with statistics
    print("\\n📋 Example of Final Prompt with Statistics:")
    print("=" * 50)
    final_prompt = generator.prompt_manager.statistics_manager.replace_placeholders_in_prompt(
        prompt_with_stats.replace("{LOCATION}", "Manchester"),
        synthetic_df=synthetic_df,
        format_type="comparison"
    )
    print(final_prompt)
    
    print("\\n🎉 Demo Complete!")
    print("\\n💡 Key Features Demonstrated:")
    print("   ✅ Automatic placeholder replacement during generation")
    print("   ✅ Real-time statistics feedback after each batch")
    print("   ✅ Integration of classifiers and custom functions")
    print("   ✅ Target vs observed comparisons for LLM guidance")
    print("   ✅ No manual prompt management required")
    
    print("\\n� What Happened Automatically:")
    print("   1. First batch: Empty placeholders (no data yet)")
    print("   2. Generated 2 households → computed initial statistics")
    print("   3. Second batch: Placeholders filled with target vs observed")
    print("   4. Generated 2 more households → updated statistics")
    print("   5. Third batch: Refreshed statistics guided final generation")
    
    print("\\n🚀 Benefits:")
    print("   • LLM gets smarter with each batch")
    print("   • Automatic convergence toward target distributions")
    print("   • User just defines targets, system handles the rest")
    print("   • Extensible to any demographic characteristic")


if __name__ == "__main__":
    main()
