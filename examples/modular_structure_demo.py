"""Example showing the new modular classifier structure."""

from population_generator import PopulationGenerator
from population_generator.llm.base import BaseLLM

# Core package - base classes for building classifiers
from population_generator.classifiers.base import DemographicClassifier, HouseholdLevelClassifier

# Contributed implementations - clearly separate  
from population_generator.contrib.classifiers.uk import UKHouseholdSizeClassifier, UKHouseholdCompositionClassifier


class MockLLM(BaseLLM):
    """Mock LLM for demonstration."""
    
    def __init__(self):
        self.model_name = "mock-llm"
        self.is_local = False
    
    def generate_text(self, prompt, timeout=30):
        return '''[
            {"name": "John Smith", "age": 42, "gender": "Male", "relationship": "Head"},
            {"name": "Sarah Smith", "age": 38, "gender": "Female", "relationship": "Spouse"}
        ]'''
    
    def get_model_metadata(self):
        return {"model": self.model_name, "type": "mock"}


def main():
    """Demonstrate the new modular structure."""
    print("🏗️  New Modular Package Structure Demo")
    print("=" * 50)
    
    print("\\n📦 Package Structure:")
    print("   population_generator/")
    print("   ├── core/                    # Core generation logic")
    print("   ├── llm/                     # LLM interfaces")
    print("   ├── utils/                   # Utilities")
    print("   ├── classifiers/             # Abstract base classes only")
    print("   │   ├── household_size/")
    print("   │   └── household_type/")
    print("   └── contrib/                 # Contributed implementations")
    print("       ├── classifiers/")
    print("       │   └── uk/              # UK-specific classifiers")
    print("       └── data/                # Sample data files")
    
    print("\\n🎯 Benefits:")
    print("   ✅ Clear separation of core vs contributed code")
    print("   ✅ Base classes always available for custom implementations")
    print("   ✅ Reference implementations for common use cases")
    print("   ✅ Easy to extend with new regions/countries")
    print("   ✅ Optional usage - users can ignore contrib if they want")
    
    print("\\n📖 Usage Examples:")
    print("\\n1. Using Core Base Classes:")
    print("   from population_generator.classifiers.base import HouseholdLevelClassifier")
    print("   # Implement your own custom classifier")
    
    print("\\n2. Using Contributed Implementations:")
    print("   from population_generator.contrib.classifiers.uk import UKHouseholdSizeClassifier")
    print("   # Use ready-made UK Census classifier")
    
    print("\\n🚀 Demo: Generate households with UK classifiers")
    
    # Initialize generator
    generator = PopulationGenerator(
        data_path="./examples/data",
        prompts_path="./examples/prompts"
    )
    
    # Register UK classifiers (from contrib)
    uk_size_classifier = UKHouseholdSizeClassifier()
    uk_comp_classifier = UKHouseholdCompositionClassifier()
    
    generator.prompt_manager.register_household_size_classifier(
        uk_size_classifier,
        target_data={1: 30.0, 2: 35.0, 3: 15.0, 4: 12.0, 5: 5.0, 6: 2.0, 7: 1.0, 8: 1.0}
    )
    
    generator.prompt_manager.register_household_composition_classifier(
        uk_comp_classifier,
        target_data={"single_person": 30.0, "couple_no_children": 25.0, "couple_with_children": 25.0, "single_parent": 10.0, "other": 10.0}
    )
    
    # Generate with statistics placeholders
    prompt = '''Generate a household for {LOCATION}.
    
{HOUSEHOLD_SIZE_STATS}
{HOUSEHOLD_COMPOSITION_STATS}

Create realistic household members based on the statistics above.
Return JSON: [{"name": "string", "age": number, "gender": "string", "relationship": "string"}]'''
    
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
    
    households = generator.generate_households(
        n_households=2,
        model=MockLLM(),
        base_prompt=prompt,
        schema=schema,
        location="London"
    )
    
    print(f"\\n   ✅ Generated {len(households)} households using UK contrib classifiers")
    print("\\n💡 This demonstrates how contrib classifiers integrate seamlessly")
    print("   with the core system while remaining clearly separated.")


if __name__ == "__main__":
    main()
