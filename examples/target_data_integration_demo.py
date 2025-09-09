"""Example showing target data integration with flexible data loading.

This demonstrates how classifiers can use target data from different formats
to guide population generation toward specific demographic goals.
"""

from population_generator import PopulationGenerator
from population_generator.llm.base import BaseLLM
from population_generator.contrib.classifiers.uk import UKHouseholdSizeClassifier
import pandas as pd


class MockLLM(BaseLLM):
    """Mock LLM for demonstration."""
    
    def __init__(self):
        self.model_name = "mock-target-data-demo"
        self.is_local = False
        self._response_templates = [
            '''[{"name": "Alice Johnson", "age": 34, "gender": "Female", "relationship": "Head"}]''',
            '''[{"name": "Sarah Wilson", "age": 28, "gender": "Female", "relationship": "Head"}, {"name": "Mike Wilson", "age": 30, "gender": "Male", "relationship": "Spouse"}]''',
            '''[{"name": "David Brown", "age": 45, "gender": "Male", "relationship": "Head"}, {"name": "Emma Brown", "age": 42, "gender": "Female", "relationship": "Spouse"}, {"name": "Jack Brown", "age": 15, "gender": "Male", "relationship": "Child"}]''',
            '''[{"name": "Robert Smith", "age": 52, "gender": "Male", "relationship": "Head"}, {"name": "Jennifer Smith", "age": 48, "gender": "Female", "relationship": "Spouse"}, {"name": "Chloe Smith", "age": 18, "gender": "Female", "relationship": "Child"}, {"name": "Luke Smith", "age": 16, "gender": "Male", "relationship": "Child"}]''',
            '''[{"name": "Margaret Davis", "age": 73, "gender": "Female", "relationship": "Head"}]''',
            '''[{"name": "Tom Taylor", "age": 29, "gender": "Male", "relationship": "Head"}, {"name": "Amy Taylor", "age": 27, "gender": "Female", "relationship": "Spouse"}]'''
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


def analyze_target_vs_observed(households: list, classifier, target_data: dict):
    """Compare target vs observed distributions."""
    
    # Convert households to DataFrame
    all_people = []
    for hh_id, household in enumerate(households, 1):
        for person in household:
            person_data = person.copy()
            person_data['household_id'] = hh_id
            all_people.append(person_data)
    
    synthetic_df = pd.DataFrame(all_people)
    observed_data = classifier.compute_observed_distribution(synthetic_df)
    
    print(f"📊 Target vs Observed Analysis:")
    print(f"{'Category':<8} {'Target':<8} {'Observed':<10} {'Difference':<10}")
    print("-" * 40)
    
    for category in sorted(set(list(target_data.keys()) + list(observed_data.keys()))):
        target_pct = target_data.get(category, 0.0)
        observed_pct = observed_data.get(category, 0.0)
        diff = observed_pct - target_pct
        
        print(f"{category:<8} {target_pct:<8.1f} {observed_pct:<10.1f} {diff:+.1f}")


def main():
    """Demonstrate target data integration with flexible loading."""
    
    print_separator("Target Data Integration Demo")
    print("This demo shows how different data formats can be used as targets:")
    print("• Raw UK Census files → automatic conversion to classifier format")
    print("• Preprocessed CSV files → direct usage")
    print("• JSON files with metadata → structured loading")
    
    # Initialize components
    generator = PopulationGenerator(data_path="examples/data/formats")
    llm = MockLLM()
    classifier = UKHouseholdSizeClassifier()
    
    # Test different target data sources
    target_files = [
        ("raw_uk_census.csv", "Raw UK Census Data"),
        ("preprocessed.csv", "Preprocessed CSV Data"),
        ("json_with_metadata.json", "JSON with Metadata")
    ]
    
    for filename, description in target_files:
        print_separator(f"Using {description}")
        print(f"📁 Source file: {filename}")
        
        try:
            # Register classifier with target data file
            generator.prompt_manager.register_classifier(
                'HOUSEHOLD_SIZE_STATS', 
                classifier,
                target_file=filename
            )
            
            # Create prompt that references target statistics
            base_prompt = """Generate a realistic UK household.

Target household size distribution:
{HOUSEHOLD_SIZE_STATS}

Create a household that helps achieve these target percentages.
Return JSON: [{"name": "string", "age": number, "gender": "string", "relationship": "string"}]"""
            
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
            
            # Generate households
            households = generator.generate_households(
                n_households=6,
                model=llm,
                base_prompt=base_prompt,
                schema=schema,
                location="UK",
                batch_size=3
            )
            
            print(f"✅ Generated {len(households)} households")
            
            # Show generated households
            print("\\n🏠 Generated Households:")
            for i, household in enumerate(households, 1):
                print(f"   Household {i}: {len(household)} people")
            
            # Load target data for comparison
            target_data = generator.prompt_manager.statistics_manager._load_target_data(filename)
            if target_data:
                print(f"\\n📈 Target Data Loaded: {len(target_data)} categories")
                analyze_target_vs_observed(households, classifier, target_data)
            
            print(f"\\n✅ Successfully demonstrated {description.lower()}")
            
        except Exception as e:
            print(f"❌ Error with {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    print_separator("Demo Complete")
    print("🎉 Target data integration successfully demonstrated!")
    print("\\n💡 Key Benefits:")
    print("   ✅ Any supported data format can be used as targets")
    print("   ✅ Automatic format detection and conversion")
    print("   ✅ Statistics appear in prompts to guide generation")  
    print("   ✅ Real-time comparison of target vs observed")
    print("   ✅ Flexible and extensible system")
    
    print("\\n🚀 Users can provide target data in their preferred format!")


if __name__ == "__main__":
    main()
