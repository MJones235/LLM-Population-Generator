#!/usr/bin/env python3
"""
Improved Target Integration Demo

This shows the clean approach where users explicitly register:
1. Placeholder names (user-controlled, flexible)
2. Classifiers (any classifier they want)
3. Target data files (optional, user-specified)

No auto-inference, no brittle name matching - just explicit, flexible configuration.
"""

from population_generator.core.generator import PopulationGenerator
from population_generator.contrib.classifiers.uk import UKHouseholdSizeClassifier
from population_generator.llm.base import BaseLLM
import json

class MockLLM(BaseLLM):
    """Simple mock LLM for testing."""
    
    def __init__(self):
        self.model_name = "mock-target-demo"
        self.is_local = False
        self._responses = [
            '[{"name": "Alice", "age": 35, "gender": "Female"}]',  # Single person
            '[{"name": "Bob", "age": 42, "gender": "Male"}, {"name": "Carol", "age": 38, "gender": "Female"}]',  # Couple
            '[{"name": "Dave", "age": 28, "gender": "Male"}]',  # Single person
        ]
        self._current = 0
    
    def generate_text(self, prompt, timeout=30):
        response = self._responses[self._current % len(self._responses)]
        self._current += 1
        return response
    
    def get_model_metadata(self):
        return {"model": self.model_name}

def test_clean_target_integration():
    """Test the clean, explicit target data integration approach."""
    
    print("🧪 Clean Target Integration Demo")
    print("=" * 50)
    
    # Initialize generator
    generator = PopulationGenerator()
    llm = MockLLM()
    
    # Create classifier
    household_classifier = UKHouseholdSizeClassifier()
    
    print("\n1️⃣ Registering classifier WITHOUT target data:")
    # Users explicitly specify placeholder name and classifier
    generator.prompt_manager.register_classifier(
        placeholder="HOUSEHOLD_SIZE_STATS",  # User controls the name
        classifier=household_classifier      # User specifies exact classifier
    )
    
    # Test prompt without target data
    test_prompt = """Generate a UK household.

Current household size distribution:
{HOUSEHOLD_SIZE_STATS}

Return JSON: [{"name": "string", "age": number, "gender": "string"}]"""
    
    print(f"\n   Prompt (first batch, no target): Shows only current stats")
    
    print("\n2️⃣ Now registering classifier WITH target data:")
    # Same explicit approach, but with target data
    generator.prompt_manager.register_classifier(
        placeholder="HOUSEHOLD_SIZE_STATS",  # Same placeholder name
        classifier=household_classifier,     # Same classifier
        target_file="examples/data/formats/raw_uk_census.csv"  # User specifies target file
    )
    
    print(f"   ✅ Registered with target data file")
    
    # Test generation
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "gender": {"type": "string"}
            },
            "required": ["name", "age", "gender"]
        }
    }
    
    print("\n3️⃣ Generating households:")
    households = generator.generate_households(
        n_households=3,
        model=llm,
        base_prompt=test_prompt,
        schema=schema,
        location="London",
        batch_size=2
    )
    
    print(f"\n✅ Generated {len(households)} households")
    for i, household in enumerate(households, 1):
        print(f"   Household {i}: {len(household)} people - {[p['name'] for p in household]}")
    
    print("\n4️⃣ Key Benefits of This Approach:")
    print("   ✅ User controls placeholder names (flexible)")
    print("   ✅ User specifies exact classifiers (no guessing)")  
    print("   ✅ User chooses target data files (explicit)")
    print("   ✅ No brittle name matching or auto-inference")
    print("   ✅ Works with any custom classifier or placeholder name")
    
    print("\n5️⃣ Example with custom names:")
    print("   generator.prompt_manager.register_classifier('MY_CUSTOM_STATS', my_classifier, target_file='my_data.csv')")
    print("   # Then use {MY_CUSTOM_STATS} in prompts")
    
    print(f"\n🎉 Clean target integration complete!")

if __name__ == "__main__":
    test_clean_target_integration()
