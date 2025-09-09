"""Example of creating custom classifiers."""

from population_generator.classifiers.base import HouseholdLevelClassifier
from typing import Dict, List
import pandas as pd


class CustomHouseholdSizeClassifier(HouseholdLevelClassifier):
    """Example custom household size classifier."""
    
    def get_name(self) -> str:
        return "custom_region"
    
    def classify_household(self, household_df: pd.DataFrame, **kwargs) -> str:
        """Classify household by size using custom categories."""
        size = len(household_df)
        if size == 1:
            return "Single person"
        elif size == 2:
            return "Two person"
        elif size <= 4:
            return "Small family (3-4)"
        elif size <= 6:
            return "Large family (5-6)"
        else:
            return "Very large (7+)"


class CustomHouseholdCompositionClassifier(HouseholdLevelClassifier):
    """Example custom household composition classifier."""
    
    def get_name(self) -> str:
        return "custom_region"
    
    def classify_household(self, household_df: pd.DataFrame, relationship_col: str = "relationship", **kwargs) -> str:
        """Classify household structure based on relationships."""
        n = len(household_df)
        roles = set(household_df[relationship_col])
        
        if n == 1:
            return "Single person"
        elif "Child" in roles and ("Head" in roles or "Spouse" in roles):
            return "Couple with children"  
        elif "Child" in roles:
            return "Single parent"
        elif n == 2 and "Spouse" in roles:
            return "Couple no children"
        else:
            return "Other"
    
    def get_label_order(self) -> List[str]:
        """Return ordered list of labels for display."""
        return [
            "Single person",
            "Couple only", 
            "Couple with children",
            "Single parent",
            "Extended family",
            "Other arrangement"
        ]


def main():
    """Example of using custom classifiers."""
    
    # Create sample data
    sample_data = pd.DataFrame([
        {"household_id": 1, "name": "John", "age": 35, "relationship": "Head"},
        {"household_id": 1, "name": "Jane", "age": 32, "relationship": "Spouse"},
        {"household_id": 1, "name": "Billy", "age": 8, "relationship": "Child"},
        {"household_id": 2, "name": "Mary", "age": 45, "relationship": "Head"},
        {"household_id": 2, "name": "Tom", "age": 15, "relationship": "Child"},
        {"household_id": 3, "name": "Bob", "age": 28, "relationship": "Head"},
    ])
    
    # Test custom classifiers
    size_classifier = CustomHouseholdSizeClassifier()
    comp_classifier = CustomHouseholdCompositionClassifier()
    
    print("Custom Size Distribution:")
    size_dist = size_classifier.compute_observed_distribution(sample_data)
    for category, percentage in size_dist.items():
        print(f"  {category}: {percentage:.1f}%")
    
    print("\nCustom Composition Distribution:")
    comp_dist = comp_classifier.compute_observed_distribution(sample_data)
    for category, percentage in comp_dist.items():
        print(f"  {category}: {percentage:.1f}%")


if __name__ == "__main__":
    main()
