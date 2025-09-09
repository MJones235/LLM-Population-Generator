"""Example demonstrating all five UK classifiers working together.

This example shows the complete suite of UK demographic classifiers:
- UKHouseholdSizeClassifier (household-level)
- UKHouseholdCompositionClassifier (household-level) 
- UKAgeSexPyramidClassifier (individual-level)
- UKAgeClassifier (individual-level)
- UKSexClassifier (individual-level)
"""

from population_generator import PopulationGenerator
from population_generator.llm.base import BaseLLM
from population_generator.contrib.classifiers.uk import (
    UKHouseholdSizeClassifier, 
    UKHouseholdCompositionClassifier,
    UKAgeSexPyramidClassifier,
    UKAgeClassifier,
    UKSexClassifier
)
import pandas as pd


class MockLLM(BaseLLM):
    """Mock LLM for demonstration."""
    
    def __init__(self):
        self.model_name = "mock-all-classifiers-demo"
        self.is_local = False
        self._response_templates = [
            '''[
                {"name": "Sarah Johnson", "age": 34, "gender": "Female", "relationship": "Head"},
                {"name": "Michael Johnson", "age": 36, "gender": "Male", "relationship": "Spouse"},
                {"name": "Emma Johnson", "age": 8, "gender": "Female", "relationship": "Child"},
                {"name": "Luke Johnson", "age": 5, "gender": "Male", "relationship": "Child"}
            ]''',
            '''[
                {"name": "Margaret Smith", "age": 72, "gender": "Female", "relationship": "Head"}
            ]''',
            '''[
                {"name": "David Wilson", "age": 28, "gender": "Male", "relationship": "Head"},
                {"name": "Lisa Wilson", "age": 26, "gender": "Female", "relationship": "Spouse"}
            ]''',
            '''[
                {"name": "Amanda Brown", "age": 42, "gender": "Female", "relationship": "Head"},
                {"name": "Jack Brown", "age": 16, "gender": "Male", "relationship": "Child"},
                {"name": "Sophie Brown", "age": 13, "gender": "Female", "relationship": "Child"}
            ]''',
            '''[
                {"name": "Alex Taylor", "age": 29, "gender": "Male", "relationship": "Head"}
            ]''',
            '''[
                {"name": "Jennifer Davis", "age": 31, "gender": "Female", "relationship": "Head"},
                {"name": "Christopher Davis", "age": 33, "gender": "Male", "relationship": "Spouse"},
                {"name": "Oliver Davis", "age": 2, "gender": "Male", "relationship": "Child"},
                {"name": "Sophia Davis", "age": 4, "gender": "Female", "relationship": "Child"},
                {"name": "Grandma Davis", "age": 68, "gender": "Female", "relationship": "Parent"}
            ]'''
        ]
        self._current_response = 0
    
    def generate_text(self, prompt, timeout=30):
        response = self._response_templates[self._current_response % len(self._response_templates)]
        self._current_response += 1
        return response
    
    def get_model_metadata(self):
        return {"model": self.model_name, "type": "mock"}


def print_separator(title: str, char: str = "="):
    """Print a formatted section separator."""
    print(f"\n{char*60}")
    print(f"  {title}")
    print(f"{char*60}")


def analyze_with_all_classifiers(households: list):
    """Analyze generated households using all five classifiers."""
    
    # Convert households to DataFrame for analysis
    all_people = []
    for hh_id, household in enumerate(households, 1):
        for person in household:
            person_data = person.copy()
            person_data['household_id'] = hh_id
            all_people.append(person_data)
    
    synthetic_df = pd.DataFrame(all_people)
    
    print(f"📊 Dataset Overview:")
    print(f"   Households: {len(households)}")
    print(f"   Total people: {len(synthetic_df)}")
    print(f"   Age range: {synthetic_df['age'].min()}-{synthetic_df['age'].max()} years")
    print(f"   Average household size: {len(synthetic_df)/len(households):.1f}")
    
    # Create all classifiers
    classifiers = {
        "Household Size": UKHouseholdSizeClassifier(),
        "Household Composition": UKHouseholdCompositionClassifier(),
        "Age Distribution": UKAgeClassifier(),
        "Sex Distribution": UKSexClassifier(),
        "Age/Sex Pyramid": UKAgeSexPyramidClassifier()
    }
    
    # Analyze with each classifier
    for name, classifier in classifiers.items():
        print(f"\n🔍 {name} Analysis:")
        try:
            distribution = classifier.compute_observed_distribution(synthetic_df)
            
            # Show only non-zero categories for readability
            non_zero_dist = {k: v for k, v in distribution.items() if v > 0}
            
            if name == "Age/Sex Pyramid" and len(non_zero_dist) > 8:
                # For age/sex pyramid, show top categories
                sorted_dist = sorted(non_zero_dist.items(), key=lambda x: x[1], reverse=True)[:8]
                for category, percentage in sorted_dist:
                    print(f"   {category}: {percentage}%")
                if len(non_zero_dist) > 8:
                    print(f"   ... and {len(non_zero_dist) - 8} more categories")
            else:
                # Show all for other classifiers
                for category, percentage in sorted(non_zero_dist.items()):
                    print(f"   {category}: {percentage}%")
                    
        except Exception as e:
            print(f"   Error analyzing with {name}: {e}")
    

def main():
    """Run comprehensive demo of all UK classifiers."""
    
    print_separator("Complete UK Classifiers Suite Demo")
    print("This demo showcases all five UK demographic classifiers:")
    print("• Household Size (household-level)")
    print("• Household Composition (household-level)")
    print("• Age Distribution (individual-level)")
    print("• Sex Distribution (individual-level)")
    print("• Age/Sex Pyramid (individual-level)")
    
    # Initialize components
    generator = PopulationGenerator()
    llm = MockLLM()
    
    print_separator("Registering All Classifiers")
    
    # Register all classifiers with unique placeholders
    generator.prompt_manager.register_classifier('HOUSEHOLD_SIZE_STATS', UKHouseholdSizeClassifier())
    generator.prompt_manager.register_classifier('HOUSEHOLD_COMPOSITION_STATS', UKHouseholdCompositionClassifier())
    generator.prompt_manager.register_classifier('AGE_STATS', UKAgeClassifier())
    generator.prompt_manager.register_classifier('SEX_STATS', UKSexClassifier())
    generator.prompt_manager.register_classifier('AGE_SEX_PYRAMID_STATS', UKAgeSexPyramidClassifier())
    
    print("✅ Household Size Classifier")
    print("✅ Household Composition Classifier") 
    print("✅ Age Distribution Classifier")
    print("✅ Sex Distribution Classifier")
    print("✅ Age/Sex Pyramid Classifier")
    
    print_separator("Generating Households with All Statistics")
    
    # Define comprehensive prompt with all statistics
    base_prompt = """Generate a realistic UK household.

Current Demographics:
- Household sizes: {HOUSEHOLD_SIZE_STATS}
- Household types: {HOUSEHOLD_COMPOSITION_STATS}
- Age distribution: {AGE_STATS}
- Gender distribution: {SEX_STATS}
- Age/Gender pyramid: {AGE_SEX_PYRAMID_STATS}

Create a household that fits UK demographic patterns and helps balance these statistics.
Return JSON: [{"name": "string", "age": number, "gender": "string", "relationship": "string"}]"""
    
    # JSON schema for validation
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
    
    try:
        # Generate diverse households
        households = generator.generate_households(
            n_households=6,
            model=llm,
            base_prompt=base_prompt,
            schema=schema,
            location="UK",
            batch_size=2
        )
        
        print(f"✅ Successfully generated {len(households)} households")
        
        # Display generated households
        print_separator("Generated Households")
        for i, household in enumerate(households, 1):
            print(f"\n🏠 Household {i} ({len(household)} people):")
            for person in household:
                rel = person.get('relationship', 'Unknown')
                print(f"   👤 {person['name']:<20} | Age: {person['age']:<3} | {person['gender']:<10} | {rel}")
        
        # Comprehensive analysis with all classifiers
        print_separator("Comprehensive Demographic Analysis")
        analyze_with_all_classifiers(households)
        
        print_separator("Demo Complete", "🎉")
        print("🎉 Successfully demonstrated all five UK classifiers working together!")
        print("\n💡 Key Features Demonstrated:")
        print("   ✅ Household-level classification (size & composition)")
        print("   ✅ Individual-level classification (age, sex, age/sex pyramid)")
        print("   ✅ Statistics integration with prompt generation")
        print("   ✅ Generic DemographicClassifier architecture")
        print("   ✅ Extensible contrib structure")
        print("   ✅ Multiple classifier types (household vs individual)")
        print("   ✅ Flexible statistics placeholders system")
        print("\n🚀 The package now provides a complete demographic analysis toolkit!")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
