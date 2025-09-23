"""Example using Azure OpenAI model with UK classifiers for statistical feedback."""

import os
from dotenv import load_dotenv

from population_generator import PopulationGenerator
from population_generator.llm import OpenAIModel
from population_generator.contrib.classifiers.uk import (
    UKHouseholdSizeClassifier,
    UKHouseholdCompositionClassifier, 
    UKAgeClassifier,
    UKSexClassifier
)


def main():
    """Demonstrate Azure OpenAI model with UK classifiers for statistical feedback."""
    
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
        prompt = generator.prompt_manager.load_prompt("basic_household.txt")
        schema = generator.data_loader.load_schema("household_basic.json")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Make sure you have the example data files in examples/data/ and examples/prompts/")
        return
    
    # Register UK classifiers with their corresponding target data
    print("Loading UK classifiers and target data...")
    
    # 1. Household Size Classifier
    household_size_classifier = UKHouseholdSizeClassifier()
    generator.prompt_manager.register_classifier(
        "HOUSEHOLD_SIZE_STATS",
        household_size_classifier,
        target_file="targets/uk_household_size.csv"
    )
    
    # 2. Household Composition Classifier  
    household_composition_classifier = UKHouseholdCompositionClassifier()
    generator.prompt_manager.register_classifier(
        "HOUSEHOLD_COMPOSITION_STATS",
        household_composition_classifier,
        target_file="targets/uk_household_composition.csv"
    )
    
    # 3. Age Classifier
    age_classifier = UKAgeClassifier()
    generator.prompt_manager.register_classifier(
        "AGE_STATS",
        age_classifier,
        target_file="targets/uk_age_distribution.csv"
    )
    
    # 4. Sex Classifier
    sex_classifier = UKSexClassifier()
    generator.prompt_manager.register_classifier(
        "SEX_STATS",
        sex_classifier,
        target_file="targets/uk_sex_distribution.csv"
    )
    
    print("✓ All UK classifiers registered with target data")
    
    try:
        print(f"\nGenerating 15 households for {location} using Azure OpenAI...")
        print("This will demonstrate statistical feedback between batches.\n")
        
        # Generate households with statistical feedback
        households = generator.generate_households(
            n_households=15,
            model=llm,
            base_prompt=prompt,
            schema=schema,
            location=location,
            batch_size=5  # Small batches to see feedback in action
        )
        
        print(f"\n🎉 Successfully generated {len(households)} households!")
        
        # Display summary statistics
        total_people = sum(len(h.get('household', [])) for h in households)
        print(f"Total people generated: {total_people}")
        
        # Show sample households
        print("\n--- Sample Households ---")
        for i, household in enumerate(households[:3], 1):
            print(f"\nHousehold {i}:")
            for person in household.get('household', []):
                age = person.get('age', '?')
                gender = person.get('gender', '?')  
                relationship = person.get('relationship', '?')  # Fixed: was relationship_to_head
                print(f"  • {age} year old {gender} ({relationship})")
        
        if len(households) > 3:
            print(f"\n... and {len(households) - 3} more households")

    except Exception as e:
        print(f"Error during generation: {e}")
        print("Make sure your Azure OpenAI credentials are correct and you have API access.")


if __name__ == "__main__":
    main()
