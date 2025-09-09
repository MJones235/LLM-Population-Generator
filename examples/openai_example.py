"""Complete example using Azure OpenAI model."""

import os
from dotenv import load_dotenv

from population_generator import PopulationGenerator
from population_generator.llm import OpenAIModel
from population_generator.contrib.classifiers.uk import UKHouseholdCompositionClassifier, UKHouseholdSizeClassifier


def main():
    """Run complete population generation with Azure OpenAI."""
    
    # Load environment variables
    load_dotenv(".env")

    # Get required configuration from environment
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    if not api_key:
        print("Error: AZURE_OPENAI_API_KEY environment variable not set")
        print("Create a .env file with: AZURE_OPENAI_API_KEY=your_azure_key_here")
        return
        
    if not azure_endpoint:
        print("Error: AZURE_OPENAI_ENDPOINT environment variable not set")
        print("Add to .env file: AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
        return
    
    location = "London"
    
    # Initialize the generator
    generator = PopulationGenerator(
        data_path="./examples/data",
        prompts_path="./examples/prompts"
    )
    
    # Configure Azure OpenAI model
    llm = OpenAIModel(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        model_name=os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        temperature=float(os.getenv("AZURE_OPENAI_TEMPERATURE", 0.7))
    )
    
    # Load prompt and schema
    try:
        prompt = generator.prompt_manager.load_prompt(
            "basic_household.txt",
            {"LOCATION": location, "TOTAL_HOUSEHOLDS": "5"}
        )
        schema = generator.data_loader.load_schema("household_basic.json")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Make sure you have the example data files in examples/data/ and examples/prompts/")
        return
    
    try:
        print("Generating 5 households using Azure OpenAI GPT-4o-mini...")
        
        # Generate households
        households = generator.generate_households(
            n_households=5,
            model=llm,
            base_prompt=prompt,
            schema=schema,
            location="London",
            batch_size=2
        )
        
        print(f"\n🎉 Successfully generated {len(households)} households!")
        
        # Display results
        for i, household in enumerate(households, 1):
            print(f"\n--- Household {i} ---")
            for person in household.get('household', []):
                print(f"{person.get('age', '?')} years old, {person.get('gender', '?')}, {person.get('relationship_to_head', '?')}")

    except Exception as e:
        print(f"Error during generation: {e}")


if __name__ == "__main__":
    main()
