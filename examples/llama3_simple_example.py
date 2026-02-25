#!/usr/bin/env python3
"""
Simple Llama 3.2 Population Generator Example

This example demonstrates how to generate a small population (30 people) 
using the Ollama model with Llama 3.2:3b for quick local testing.

✅ TESTED AND WORKING - This example has been verified to work correctly
with Ollama and the population generator framework.

Prerequisites:
1. Install Ollama: https://ollama.ai/
2. Pull the model: ollama pull llama3.2:3b
3. Install dependencies: pip install langchain-ollama
4. Make sure Ollama server is running
"""

import sys
import json
from pathlib import Path

# Add parent directory to path so we can import population_generator
sys.path.insert(0, str(Path(__file__).parent.parent))

from population_generator import PopulationGenerator
from population_generator.llm import create_ollama_model


def main():
    """Generate a small population using Llama 3.2:3b."""
    
    print("🦙 Llama 3.2 Population Generator")
    print("=" * 40)
    
    # Create Ollama model with Llama 3.2:3b
    print("🔧 Setting up Llama 3.2:3b model...")
    try:
        llm = create_ollama_model(
            model_name="llama3.2:3b",
            temperature=0.3,  # Lower temperature for more consistent output
            top_p=0.9
        )
        print("✅ Llama 3.2:3b model ready!")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        print("\nTroubleshooting:")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Pull the model: ollama pull llama3.2:3b")
        print("3. Check if Ollama is running: ollama list")
        return
    
    # Simple person schema - just name, age, sex
    person_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Full name of the person"
            },
            "age": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Age in years"
            },
            "sex": {
                "type": "string",
                "enum": ["Male", "Female"],
                "description": "Biological sex"
            }
        },
        "required": ["name", "age", "sex"]
    }
    
    # Simple prompt for generating one person
    prompt_template = """Generate a realistic UK person with the following information:

- Full name (realistic UK name)
- Age (0-100 years old)
- Sex (Male or Female)

Make the person demographically plausible for the UK.
Respond with valid JSON only, no additional text."""

    # Setup population generator
    print("\n🏗️ Setting up population generator...")
    generator = PopulationGenerator(
        enable_debug_logging=True
    )
    
    print("✅ Generator configured!")
    
    # Generate small population (30 people)
    num_people = 30
    print(f"\n👥 Generating {num_people} people...")
    
    try:
        # Generate people one by one using the simplified schema
        results = []
        for i in range(num_people):
            person = llm.generate_json(prompt_template, person_schema, n_attempts=3)
            results.append(person)
            print(f"  Generated person {i+1}/{num_people}: {person['name']}, {person['age']}, {person['sex']}")
            
            # Show progress every 10 people
            if (i + 1) % 10 == 0:
                print(f"  ✓ Completed {i + 1}/{num_people} people")
        
        print(f"\n✅ Generated {len(results)} people successfully!")
        
        # Show sample people
        if results:
            print(f"\n📋 First 5 people generated:")
            for i, person in enumerate(results[:5], 1):
                print(f"  {i}. {person.get('name', 'N/A')}, age {person.get('age', 'N/A')}, {person.get('sex', 'N/A')}")
        
        # Show statistics
        print(f"\n📊 Generation Statistics:")
        
        # Show failure tracking statistics
        failure_stats = llm.failure_tracker.get_session_statistics()
        print(f"  Success/Failure Rates:")
        print(f"    - Total prompts: {failure_stats.total_prompts}")
        print(f"    - Successful: {failure_stats.successful_prompts}")
        print(f"    - Failed: {failure_stats.failed_prompts}")
        print(f"    - Success rate: {failure_stats.success_rate:.1%}")
        
        if failure_stats.total_attempts > failure_stats.total_prompts:
            print(f"  Retry Statistics:")
            print(f"    - Total attempts: {failure_stats.total_attempts}")
            print(f"    - Average attempts per prompt: {failure_stats.average_attempts_per_prompt:.1f}")
            print(f"    - Max attempts for single prompt: {failure_stats.max_attempts_for_single_prompt}")
        
        if failure_stats.json_parse_failures > 0 or failure_stats.schema_validation_failures > 0:
            print(f"  Failure Types:")
            if failure_stats.json_parse_failures > 0:
                print(f"    - JSON parse failures: {failure_stats.json_parse_failures}")
            if failure_stats.schema_validation_failures > 0:
                print(f"    - Schema validation failures: {failure_stats.schema_validation_failures}")
            if failure_stats.custom_validation_failures > 0:
                print(f"    - Custom validation failures: {failure_stats.custom_validation_failures}")
            if failure_stats.timeout_failures > 0:
                print(f"    - Timeout failures: {failure_stats.timeout_failures}")
            if failure_stats.model_error_failures > 0:
                print(f"    - Model error failures: {failure_stats.model_error_failures}")
        
        # Age and sex distribution
        ages = []
        sexes = {"Male": 0, "Female": 0}
        for person in results:
            age = person.get('age')
            sex = person.get('sex')
            if age is not None:
                ages.append(age)
            if sex in sexes:
                sexes[sex] += 1
        
        if ages:
            print(f"  Demographics:")
            print(f"    - Average age: {sum(ages) / len(ages):.1f}")
            print(f"    - Age range: {min(ages)}-{max(ages)}")
            print(f"    - Sex distribution: {sexes['Male']} male, {sexes['Female']} female")
        
        # Save results to JSON file
        output_file = Path("outputs/llama3_simple_example/people.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Population saved to: {output_file}")
        print(f"📏 File size: {output_file.stat().st_size} bytes")
        
    except Exception as e:
        print(f"❌ Population generation failed: {e}")
        
        # Show failure details from the LLM's failure tracker
        try:
            failure_stats = llm.failure_tracker.get_session_statistics()
            print(f"\n🔍 Failure Analysis:")
            print(f"  - Total prompts attempted: {failure_stats.total_prompts}")
            print(f"  - Successful prompts: {failure_stats.successful_prompts}")
            print(f"  - Failed prompts: {failure_stats.failed_prompts}")
            print(f"  - Success rate: {failure_stats.success_rate:.1%}")
            
            if failure_stats.total_attempts > failure_stats.total_prompts:
                print(f"  - Total attempts (with retries): {failure_stats.total_attempts}")
                print(f"  - Average attempts per prompt: {failure_stats.average_attempts_per_prompt:.1f}")
            
            # Show failure breakdown
            failure_counts = []
            if failure_stats.json_parse_failures > 0:
                failure_counts.append(f"JSON parsing: {failure_stats.json_parse_failures}")
            if failure_stats.schema_validation_failures > 0:
                failure_counts.append(f"Schema validation: {failure_stats.schema_validation_failures}")
            if failure_stats.custom_validation_failures > 0:
                failure_counts.append(f"Custom validation: {failure_stats.custom_validation_failures}")
            if failure_stats.timeout_failures > 0:
                failure_counts.append(f"Timeouts: {failure_stats.timeout_failures}")
            if failure_stats.model_error_failures > 0:
                failure_counts.append(f"Model errors: {failure_stats.model_error_failures}")
            
            if failure_counts:
                print(f"  - Failure breakdown: {', '.join(failure_counts)}")
                
        except Exception as fe:
            print(f"  Could not retrieve failure statistics: {fe}")


if __name__ == "__main__":
    main()