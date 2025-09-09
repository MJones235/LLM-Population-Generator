"""Simple test to verify the package installation."""

def test_import():
    """Test that main package imports work."""
    try:
        from population_generator import PopulationGenerator, Config
        from population_generator.classifiers import (
            HouseholdSizeClassifier, 
            UKHouseholdSizeClassifier,
            HouseholdCompositionClassifier, 
            UKHouseholdCompositionClassifier
        )
        from population_generator.llm import BaseLLM
        print("✓ All imports successful!")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_classifier_creation():
    """Test that classifiers can be instantiated."""
    try:
        from population_generator.classifiers import (
            UKHouseholdSizeClassifier,
            UKHouseholdCompositionClassifier
        )
        
        size_classifier = UKHouseholdSizeClassifier()
        comp_classifier = UKHouseholdCompositionClassifier()
        
        print(f"✓ Size classifier: {size_classifier.get_name()}")
        print(f"✓ Composition classifier: {comp_classifier.get_name()}")
        return True
    except Exception as e:
        print(f"✗ Classifier creation failed: {e}")
        return False


def test_config_creation():
    """Test that configuration can be created."""
    try:
        from population_generator import Config
        
        config = Config()
        batch_size = config.get("generation.default_batch_size", 10)
        
        print(f"✓ Config created, default batch size: {batch_size}")
        return True
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing LLM Population Generator package installation...\n")
    
    tests = [
        test_import,
        test_classifier_creation, 
        test_config_creation
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 Package installation successful!")
    else:
        print("❌ Some tests failed. Check the errors above.")
