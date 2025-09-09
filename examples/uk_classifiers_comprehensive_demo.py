"""Example demonstrating the complete UK classifiers suite.

This example shows how to use all three UK classifiers together:
- UKHouseholdSizeClassifier (household-level)
- UKHouseholdCompositionClassifier (household-level) 
- UKAgeSexPyramidClassifier (individual-level)
"""

from population_generator import PopulationGenerator
from population_generator.llm.base import BaseLLM
from population_generator.contrib.classifiers.uk import (
    UKHouseholdSizeClassifier, 
    UKHouseholdCompositionClassifier,
    UKAgeSexPyramidClassifier
)
import pandas as pd


class MockLLM(BaseLLM):
    """Mock LLM for demonstration."""
    
    def __init__(self):
        self.model_name = "mock-comprehensive-demo"
        self.is_local = False
        self._response_templates = [
            # Diverse household types for demonstration
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
                {"name": "Robert Davis", "age": 45, "gender": "Male", "relationship": "Head"}
            ]''',
            '''[
                {"name": "Jennifer Taylor", "age": 29, "gender": "Female", "relationship": "Head"},
                {"name": "Christopher Taylor", "age": 31, "gender": "Male", "relationship": "Spouse"},
                {"name": "Oliver Taylor", "age": 2, "gender": "Male", "relationship": "Child"}
            ]'''
        ]
        self._current_response = 0
    
    def generate_text(self, prompt, timeout=30):
        response = self._response_templates[self._current_response % len(self._response_templates)]
        self._current_response += 1
        return response
    
    def get_model_metadata(self):
        return {"model": self.model_name, "type": "mock"}


def print_separator(title: str):
    """Print a formatted section separator."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def analyze_generated_data(households: list, classifiers: dict):
    """Analyze generated households using all classifiers."""
    
    # Convert households to DataFrame for analysis
    all_people = []
    for hh_id, household in enumerate(households, 1):
        for person in household:
            person_data = person.copy()
            person_data['household_id'] = hh_id
            all_people.append(person_data)
    
    synthetic_df = pd.DataFrame(all_people)
    
    print(f"\n📊 Analysis of {len(households)} generated households:")
    print(f"   Total people: {len(synthetic_df)}")
    print(f"   Age range: {synthetic_df['age'].min()}-{synthetic_df['age'].max()} years")
    print(f"   Gender distribution: {synthetic_df['gender'].value_counts().to_dict()}")
    
    # Analyze with each classifier
    for name, classifier in classifiers.items():
        print(f"\n🔍 {name} Analysis:")
        try:
            distribution = classifier.compute_observed_distribution(synthetic_df)
            
            # Show only non-zero categories for readability
            non_zero_dist = {k: v for k, v in distribution.items() if v > 0}
            
            if len(non_zero_dist) <= 10:  # Show all if reasonable number
                for category, percentage in sorted(non_zero_dist.items()):
                    print(f"   {category}: {percentage}%")
            else:  # Show top categories if too many
                sorted_dist = sorted(non_zero_dist.items(), key=lambda x: x[1], reverse=True)[:8]
                for category, percentage in sorted_dist:
                    print(f"   {category}: {percentage}%")
                if len(non_zero_dist) > 8:
                    print(f"   ... and {len(non_zero_dist) - 8} more categories")
                    
        except Exception as e:
            print(f"   Error analyzing with {name}: {e}")


def main():
    """Run comprehensive UK classifiers demonstration."""
    
    print_separator("UK Classifiers Comprehensive Demo")
    print("This demo showcases all three UK demographic classifiers:")
    print("• Household Size (household-level analysis)")
    print("• Household Composition (household-level analysis)")  
    print("• Age/Sex Pyramid (individual-level analysis)")
    
    # Initialize components
    generator = PopulationGenerator()
    llm = MockLLM()
    
    # Create all UK classifiers
    classifiers = {
        "Household Size": UKHouseholdSizeClassifier(),
        "Household Composition": UKHouseholdCompositionClassifier(), 
        "Age/Sex Pyramid": UKAgeSexPyramidClassifier()
    }
    
    print_separator("Registering Classifiers")
    
    # Register classifiers with the generator
    generator.prompt_manager.register_classifier('HOUSEHOLD_SIZE_STATS', classifiers["Household Size"])
    generator.prompt_manager.register_classifier('HOUSEHOLD_COMPOSITION_STATS', classifiers["Household Composition"])
    generator.prompt_manager.register_classifier('AGE_SEX_STATS', classifiers["Age/Sex Pyramid"])
    
    print("✅ Registered Household Size Classifier")
    print("✅ Registered Household Composition Classifier") 
    print("✅ Registered Age/Sex Pyramid Classifier")
    
    print_separator("Generating Households")
    
    # Define prompt with statistics placeholders
    base_prompt = """Generate a realistic UK household.

Current Demographics:
- Household sizes: {HOUSEHOLD_SIZE_STATS}
- Household types: {HOUSEHOLD_COMPOSITION_STATS}
- Age/Gender distribution: {AGE_SEX_STATS}

Create a household that fits UK demographic patterns.
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
            location="Manchester",
            batch_size=2
        )
        
        print(f"✅ Successfully generated {len(households)} households")
        
        # Display generated households
        print_separator("Generated Households")
        for i, household in enumerate(households, 1):
            print(f"\n🏠 Household {i} ({len(household)} people):")
            for person in household:
                rel = person.get('relationship', 'Unknown')
                print(f"   👤 {person['name']:<20} | Age: {person['age']:<3} | {person['gender']:<8} | {rel}")
        
        # Comprehensive analysis
        print_separator("Demographic Analysis")
        analyze_generated_data(households, classifiers)
        
        print_separator("Demo Complete")
        print("🎉 Successfully demonstrated all UK classifiers working together!")
        print("\n💡 Key Features Shown:")
        print("   ✅ Household-level classification (size & composition)")
        print("   ✅ Individual-level classification (age/sex pyramid)")
        print("   ✅ Statistics integration with prompt generation")
        print("   ✅ Generic DemographicClassifier architecture")
        print("   ✅ Extensible contrib structure")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
