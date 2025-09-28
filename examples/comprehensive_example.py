#!/usr/bin/env python3
"""
Comprehensive LLM Population Generator Example

This example demonstrates ALL major features of the LLM Population Generator:
1. OpenAI model integration (GPT-4o-mini)
2. UK demographic classifiers with statistical feedback
3. Token tracking and cost analysis
4. Failure tracking and analysis
5. Custom validation rules
6. Data export with metadata
7. Comprehensive statistics and analysis

This serves as both a complete demonstration and a template for real-world usage.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Core components
from population_generator import PopulationGenerator
from population_generator.llm import OpenAIModel

# UK classifiers for demographic feedback
from population_generator.contrib.classifiers.uk import (
    UKHouseholdSizeClassifier,
    UKHouseholdCompositionClassifier, 
    UKAgeClassifier,
    UKSexClassifier
)

# Utilities
from population_generator.utils import create_pricing_config
from population_generator.utils.validation import (
    CustomValidator,
    FunctionValidationRule,
    create_custom_validator_for_households
)


def create_london_validation_rules() -> CustomValidator:
    """Create custom validation rules specific to London demographics."""
            
    def validate_exactly_one_head(data):
        """Each household must have exactly one person with relationship 'Head'."""
        if isinstance(data, dict) and 'household' in data:
            household_data = data['household']
        else:
            return "Data must be a household object"
        
        if not isinstance(household_data, list):
            return "Data must be a list of household members"
        
        heads = [person for person in household_data if 
                isinstance(person, dict) and 
                person.get("relationship", "").lower() == "head"]
        
        if len(heads) == 0:
            return "Household must have exactly one person with relationship 'Head' (found 0)"
        elif len(heads) > 1:
            return f"Household must have exactly one person with relationship 'Head' (found {len(heads)})"
        
        return ""  # Valid
    
    def validate_no_minors_living_alone(data):
        """No one under 18 should be able to live alone."""
        if isinstance(data, dict) and 'household' in data:
            household_data = data['household']
        else:
            return "Data must be a household object"
        
        if not isinstance(household_data, list):
            return "Data must be a list of household members"
        
        if len(household_data) == 1:
            person = household_data[0]
            if isinstance(person, dict) and isinstance(person.get("age"), (int, float)):
                if person["age"] < 18:
                    return f"Person aged {person['age']} cannot live alone (under 18)"
        
        return ""  # Valid
    
    # Create validator with custom rules
    validator = CustomValidator()
    
    # Add custom validation rules that handle the object format    
    validator.add_rule(FunctionValidationRule(
        name="exactly_one_head",
        validation_function=validate_exactly_one_head,
        description="Each household must have exactly one person with relationship 'Head'"
    ))
    
    validator.add_rule(FunctionValidationRule(
        name="no_minors_living_alone",
        validation_function=validate_no_minors_living_alone,
        description="No one under 18 should be able to live alone"
    ))
        
    return validator


def main():
    """Run comprehensive demonstration of all features."""
    print("=" * 80)
    print("🏠 COMPREHENSIVE LLM POPULATION GENERATOR DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Load environment variables
    load_dotenv(".env")
    
    # ========================================================================
    # 1. SETUP AND CONFIGURATION
    # ========================================================================
    print("📋 STEP 1: Setup and Configuration")
    print("-" * 40)
    
    # Check for required environment variables
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    if not api_key or not azure_endpoint:
        print("❌ Error: Missing Azure OpenAI credentials")
        print("Create a .env file with:")
        print("AZURE_OPENAI_API_KEY=your_azure_key_here")
        print("AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
        return
    
    # Initialize the generator
    generator = PopulationGenerator(
        data_path="./examples/data",
        prompts_path="./examples/prompts"
    )
    
    print("✅ Population generator initialized")
    
    # ========================================================================
    # 2. TOKEN TRACKING SETUP
    # ========================================================================
    print("\n💰 STEP 2: Token Tracking and Cost Analysis Setup")
    print("-" * 50)
    
    model_name = "gpt-4o-mini"
    
    # Configure pricing (verify current rates at Azure pricing page)
    pricing = create_pricing_config(
        input_cost_per_1k=0.00015,   # $0.00015 per 1K input tokens
        output_cost_per_1k=0.0006    # $0.0006 per 1K output tokens
    )
    
    # Enable cost tracking
    generator.enable_cost_tracking(model_name, pricing)
    print(f"✅ Cost tracking enabled for {model_name}")
    print(f"   Input cost: ${pricing['input']:.6f} per 1K tokens")
    print(f"   Output cost: ${pricing['output']:.6f} per 1K tokens")
    
    # ========================================================================
    # 3. LLM MODEL CONFIGURATION
    # ========================================================================
    print("\n🤖 STEP 3: LLM Model Configuration")
    print("-" * 38)
    
    # Configure OpenAI model with failure tracking
    llm = OpenAIModel(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        model_name=model_name,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        temperature=float(os.getenv("AZURE_OPENAI_TEMPERATURE", 0.7))
    )
    
    # Enable failure tracking
    llm.enable_failure_tracking(True)
    print(f"✅ {model_name} configured with failure tracking enabled")
    print(f"   Temperature: {llm.temperature}")
    print(f"   API Version: {llm.api_version}")
    
    # ========================================================================
    # 4. UK DEMOGRAPHIC CLASSIFIERS
    # ========================================================================
    print("\n📊 STEP 4: UK Demographic Classifiers and Statistical Feedback")
    print("-" * 65)
    
    try:
        # Register all UK classifiers with their target data
        classifiers = [
            ("HOUSEHOLD_SIZE_STATS", UKHouseholdSizeClassifier(), "targets/uk_household_size.csv"),
            ("HOUSEHOLD_COMPOSITION_STATS", UKHouseholdCompositionClassifier(), "targets/uk_household_composition.csv"),
            ("AGE_STATS", UKAgeClassifier(), "targets/uk_age_distribution.csv"),
            ("SEX_STATS", UKSexClassifier(), "targets/uk_sex_distribution.csv")
        ]
        
        for placeholder, classifier, target_file in classifiers:
            generator.prompt_manager.register_classifier(
                placeholder,
                classifier,
                target_file=target_file
            )
            print(f"✅ {classifier.__class__.__name__} registered")
        
        print(f"✅ All {len(classifiers)} UK demographic classifiers registered")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading classifier target data: {e}")
        print("Make sure example data files are present in examples/data/targets/")
        return
    
    # ========================================================================
    # 5. CUSTOM VALIDATION RULES
    # ========================================================================
    print("\n✅ STEP 5: Custom Validation Rules")
    print("-" * 35)
    
    # Create custom validator for London households
    validator = create_london_validation_rules()
    print(f"✅ Custom validation rules created ({len(validator.rules)} rules)")
    for rule in validator.rules:
        print(f"   • {rule.name}: {rule.description}")
    
    # ========================================================================
    # 6. LOAD PROMPTS AND SCHEMA
    # ========================================================================
    print("\n📝 STEP 6: Loading Prompts and Schema")
    print("-" * 37)
    
    try:
        prompt = generator.prompt_manager.load_prompt("basic_household.txt")
        schema = generator.data_loader.load_schema("household_basic.json")
        print("✅ Prompt and schema loaded successfully")
        print(f"   Prompt length: {len(prompt)} characters")
        print(f"   Schema type: {schema.get('type', 'unknown')}")
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        print("Make sure example data files are present in examples/prompts/ and examples/data/schemas/")
        return
    
    # ========================================================================
    # 7. POPULATION GENERATION
    # ========================================================================
    print("\n🏗️  STEP 7: Population Generation with All Features")
    print("-" * 52)
    
    location = "London"
    n_households = 30
    batch_size = 10
    
    print(f"Generating {n_households} households for {location}")
    print(f"Using batch size of {batch_size} for statistical feedback")
    print(f"With custom validation and failure tracking enabled")
    print()
    
    try:
        # Generate households with all features enabled
        households = generator.generate_households(
            n_households=n_households,
            model=llm,
            base_prompt=prompt,
            schema=schema,
            location=location,
            batch_size=batch_size,
            custom_validator=validator  # Custom validation
        )
        
        print(f"🎉 Successfully generated {len(households)} households!")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        print("Check your Azure OpenAI credentials and API access.")
        return
    
    # ========================================================================
    # 8. ANALYSIS AND STATISTICS
    # ========================================================================
    print("\n📈 STEP 8: Analysis and Statistics")
    print("-" * 34)
    
    # Basic statistics
    total_people = sum(len(h.get('household', [])) for h in households)
    avg_household_size = total_people / len(households) if households else 0
    
    print(f"📊 Generation Summary:")
    print(f"   Total households: {len(households)}")
    print(f"   Total people: {total_people}")
    print(f"   Average household size: {avg_household_size:.2f}")
    
    # Age distribution
    ages = []
    for household in households:
        for person in household.get('household', []):
            if isinstance(person.get('age'), (int, float)):
                ages.append(person['age'])
    
    if ages:
        print(f"\n👥 Age Statistics:")
        print(f"   Age range: {min(ages)} - {max(ages)}")
        print(f"   Average age: {sum(ages) / len(ages):.1f}")
    
    # Gender distribution
    genders = []
    for household in households:
        for person in household.get('household', []):
            gender = person.get('gender', '').lower()
            if gender:
                genders.append(gender)
    
    if genders:
        gender_counts = {}
        for gender in genders:
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
        
        print(f"\n⚧️  Gender Distribution:")
        for gender, count in gender_counts.items():
            percentage = (count / len(genders)) * 100
            print(f"   {gender.title()}: {count} ({percentage:.1f}%)")
    
    # ========================================================================
    # 9. COST ANALYSIS
    # ========================================================================
    print("\n💰 STEP 9: Cost Analysis")
    print("-" * 23)
    
    cost_summary = generator.get_cost_summary()
    if "error" not in cost_summary:
        print(f"💳 Token Usage and Cost:")
        print(f"   Total tokens: {cost_summary['total_tokens']:,}")
        print(f"   Input tokens: {cost_summary['total_input_tokens']:,}")
        print(f"   Output tokens: {cost_summary['total_output_tokens']:,}")
        print(f"   Total cost: ${cost_summary['estimated_cost']['total']:.4f}")
        print(f"   Cost per household: ${cost_summary['estimated_cost']['total'] / len(households):.4f}")
        print(f"   Input cost: ${cost_summary['estimated_cost']['input']:.4f}")
        print(f"   Output cost: ${cost_summary['estimated_cost']['output']:.4f}")
    else:
        print(f"❌ Cost tracking error: {cost_summary['error']}")
    
    # ========================================================================
    # 10. FAILURE ANALYSIS
    # ========================================================================
    print("\n🔍 STEP 10: Failure Tracking Analysis")
    print("-" * 37)
    
    failure_stats = llm.get_failure_statistics()
    success_metrics = failure_stats['generation_success_metrics']
    print(f"📋 Generation Attempt Summary:")
    print(f"   Total prompts: {success_metrics['total_prompts']}")
    print(f"   Successful: {success_metrics['successful_prompts']}")
    print(f"   Failed: {success_metrics['failed_prompts']}")
    print(f"   Success rate: {success_metrics['success_rate']:.1%}")
    
    if success_metrics['failed_prompts'] > 0:
        print(f"\n❌ Failure Analysis:")
        failure_breakdown = failure_stats['failure_type_breakdown']
        if failure_breakdown['json_parsing_errors'] > 0:
            print(f"   JSON parsing errors: {failure_breakdown['json_parsing_errors']} occurrences")
        if failure_breakdown['schema_validation_errors'] > 0:
            print(f"   Schema validation errors: {failure_breakdown['schema_validation_errors']} occurrences")
        if failure_breakdown['custom_validation_errors'] > 0:
            print(f"   Custom validation errors: {failure_breakdown['custom_validation_errors']} occurrences")
        if failure_breakdown['timeout_errors'] > 0:
            print(f"   Timeout errors: {failure_breakdown['timeout_errors']} occurrences")
        if failure_breakdown['model_errors'] > 0:
            print(f"   Model errors: {failure_breakdown['model_errors']} occurrences")
    
    # ========================================================================
    # 11. DATA EXPORT
    # ========================================================================
    print("\n💾 STEP 11: Data Export and Persistence")
    print("-" * 38)
    
    # Create output directory
    output_dir = Path("./outputs/comprehensive_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare comprehensive metadata
    model_info = {
        "name": llm.model_name,
        "type": "OpenAI",
        "temperature": llm.temperature,
        "api_version": llm.api_version,
        "provider": "Azure OpenAI"
    }
    
    generation_parameters = {
        "n_households": n_households,
        "batch_size": batch_size,
        "location": location,
        "schema_type": "household_basic",
        "validation_enabled": True,
        "failure_tracking_enabled": True,
        "cost_tracking_enabled": True,
        "statistical_feedback_enabled": True,
        "classifiers_used": [c[0] for c in classifiers]
    }
    
    try:
        # Save comprehensive data
        
        saved_files = generator.save_population_data(
            households=households,
            model_info=model_info,
            generation_parameters=generation_parameters,
            output_dir=output_dir,
            output_name="uk_population_comprehensive",
            llm_model=llm
        )
        
        print(f"✅ Data saved successfully to {output_dir}")
        for file_path in saved_files:
            file_size = Path(file_path).stat().st_size
            print(f"   📄 {Path(file_path).name} ({file_size:,} bytes)")
        
    except Exception as e:
        print(f"❌ Error saving data: {e}")
    
    # ========================================================================
    # 12. SAMPLE OUTPUT
    # ========================================================================
    print("\n👨‍👩‍👧‍👦 STEP 12: Sample Generated Households")
    print("-" * 42)
    
    # Show first few households as examples
    sample_count = min(3, len(households))
    for i, household in enumerate(households[:sample_count], 1):
        print(f"\n🏠 Household {i}:")
        household_members = household.get('household', [])
        for j, person in enumerate(household_members, 1):
            age = person.get('age', '?')
            gender = person.get('gender', '?')
            relationship = person.get('relationship', '?')
            name = person.get('name', '?')
            print(f"   {j}. {name}, {age} years old, {gender} ({relationship})")
    
    if len(households) > sample_count:
        print(f"\n... and {len(households) - sample_count} more households")
        print(f"See full data in: {output_dir}")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print("\n" + "=" * 80)
    print("🎯 COMPREHENSIVE DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("✅ All major features demonstrated:")
    print("   • OpenAI GPT-4o-mini integration")
    print("   • UK demographic classifiers with statistical feedback")
    print("   • Token tracking and cost analysis")
    print("   • Failure tracking and analysis")
    print("   • Custom validation rules")
    print("   • Comprehensive data export")
    print("   • Statistical analysis")
    print()
    print(f"📊 Final Summary:")
    print(f"   Generated: {len(households)} households ({total_people} people)")
    if "error" not in cost_summary:
        print(f"   Cost: ${cost_summary['estimated_cost']['total']:.4f}")
    print(f"   Success Rate: {success_metrics['success_rate']:.1%}")
    print(f"   Output: {output_dir}")
    print()
    print("🚀 Ready for production use!")


if __name__ == "__main__":
    main()