"""Basic usage example for the LLM Population Generator.

This example demonstrates the core concepts using a mock LLM implementation.
For real LLM usage with Azure OpenAI, see openai_example.py.
"""

from population_generator import PopulationGenerator
from population_generator.llm.base import BaseLLM

class MockLLM(BaseLLM):
    """Mock LLM implementation for demonstration purposes."""
    
    def __init__(self, model_name="mock-llm", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.is_local = False
        self._household_templates = [
            # Single person household
            '''[{"name": "Alex Johnson", "age": 28, "gender": "Male", "relationship": "Head"}]''',
            # Couple household
            '''[
                {"name": "Maria Garcia", "age": 35, "gender": "Female", "relationship": "Head"},
                {"name": "David Garcia", "age": 37, "gender": "Male", "relationship": "Spouse"}
            ]''',
            # Family with children
            '''[
                {"name": "John Smith", "age": 42, "gender": "Male", "relationship": "Head"},
                {"name": "Sarah Smith", "age": 38, "gender": "Female", "relationship": "Spouse"},
                {"name": "Emma Smith", "age": 12, "gender": "Female", "relationship": "Child"},
                {"name": "Luke Smith", "age": 8, "gender": "Male", "relationship": "Child"}
            ]''',
            # Multi-generational household
            '''[
                {"name": "Robert Chen", "age": 65, "gender": "Male", "relationship": "Head"},
                {"name": "Linda Chen", "age": 62, "gender": "Female", "relationship": "Spouse"},
                {"name": "Kevin Chen", "age": 34, "gender": "Male", "relationship": "Child"},
                {"name": "Amy Chen", "age": 6, "gender": "Female", "relationship": "Grandchild"}
            ]'''
        ]
        self._template_index = 0
    
    def generate_text(self, prompt, timeout=30):
        # Cycle through different household templates for variety
        template = self._household_templates[self._template_index % len(self._household_templates)]
        self._template_index += 1
        return template
    
    def get_model_metadata(self):
        return {"model": self.model_name, "temperature": self.temperature, "type": "mock"}


def main():
    """Run basic population generation example using mock LLM."""
    
    print("🏠 LLM Population Generator - Basic Usage Example")
    print("=" * 50)
    print("This example demonstrates core concepts using a mock LLM.")
    print("For real LLM usage with OpenAI, see openai_example.py")
    print()
    
    # Initialize the generator
    generator = PopulationGenerator(
        data_path="./examples/data",  # Path to your census/demographic data
        prompts_path="./examples/prompts"  # Path to your prompt templates
    )
    
    # Use mock LLM for demonstration
    llm = MockLLM(model_name="mock-demonstration-llm", temperature=0.7)
    print(f"🤖 Using mock LLM: {llm.model_name}")

    # Define a simple prompt template inline for this example
    # In real usage, you'd typically load this from a file using generator.prompt_manager
    base_prompt = """Generate a realistic household for {LOCATION}.
    
Include name, age, gender, and relationship to head of household for each person.
Make the household demographically realistic for the location.

Return as JSON array:
[
    {"name": "string", "age": number, "gender": "string", "relationship": "string"}
]"""
    
    # Define JSON schema for validation
    # In real usage, you'd typically load this from a schema file
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
    
    print("\n🏗️  Generating households...")
    print("Parameters:")
    print(f"  • Number of households: 6")
    print(f"  • Location: Manchester")
    print(f"  • Batch size: 2")
    print(f"  • Statistics: Automatic (if registered)")
    print()
    
    try:
        # Generate households using the core PopulationGenerator
        households = generator.generate_households(
            n_households=6,           # Generate 6 example households
            model=llm,               # Use our mock LLM
            base_prompt=base_prompt, # Our simple prompt template
            schema=schema,           # JSON schema for validation
            location="Manchester",   # Target location
            batch_size=2            # Process 2 households at a time
        )
        
        print(f"✅ Successfully generated {len(households)} households!")
        print("\n" + "="*50)
        print("GENERATED HOUSEHOLDS:")
        print("="*50)
        
        # Display all generated households
        for i, household in enumerate(households, 1):
            print(f"\n🏠 Household {i} ({len(household)} people):")
            for person in household:
                print(f"   👤 {person['name']:<15} | Age: {person['age']:<3} | {person['gender']:<8} | {person['relationship']}")
        
        print(f"\n📊 Summary: Generated {len(households)} diverse households with {sum(len(h) for h in households)} total people")
        print("💡 This demonstrates the basic population generation workflow.")
        print("   For real usage with actual LLMs, see openai_example.py")
                
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        print("This is unexpected since we're using a mock LLM. Please check the setup.")


if __name__ == "__main__":
    main()
