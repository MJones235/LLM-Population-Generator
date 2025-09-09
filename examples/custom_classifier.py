"""Example of creating custom classifiers."""

from population_generator.classifiers.household_size.base import HouseholdSizeClassifier
from population_generator.classifiers.household_type.base import HouseholdCompositionClassifier
from typing import Dict, List
import pandas as pd


class CustomHouseholdSizeClassifier(HouseholdSizeClassifier):
    """Example custom household size classifier."""
    
    def get_name(self) -> str:
        return "custom_region"
    
    def compute_observed_distribution(self, synthetic_df: pd.DataFrame) -> Dict[str, float]:
        """Compute household size distribution with custom buckets."""
        household_sizes = synthetic_df.groupby("household_id").size()
        
        # Custom size buckets
        def categorize_size(size):
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
        
        size_categories = household_sizes.apply(categorize_size)
        counts = size_categories.value_counts(normalize=True) * 100
        
        return counts.to_dict()


class CustomHouseholdCompositionClassifier(HouseholdCompositionClassifier):
    """Example custom household composition classifier."""
    
    def get_name(self) -> str:
        return "custom_region"
    
    def compute_observed_distribution(self, synthetic_df: pd.DataFrame, relationship_col: str = "relationship") -> Dict[str, float]:
        """Compute household composition with custom categories."""
        household_labels = synthetic_df.groupby("household_id").apply(
            lambda x: self.classify_household_structure(x, relationship_col)
        )
        counts = household_labels.value_counts(normalize=True) * 100
        return counts.to_dict()
    
    def classify_household_structure(self, group: pd.DataFrame, relationship_col: str = "relationship") -> str:
        """Custom household classification logic."""
        n = len(group)
        roles = set(group[relationship_col])
        
        if n == 1:
            return "Single person"
        
        # Check for couples
        if any(role in {"Spouse", "Partner"} for role in roles):
            if any(role == "Child" for role in roles):
                return "Couple with children"
            else:
                return "Couple only"
        
        # Check for single parents
        if "Child" in roles and n == 2:
            return "Single parent"
        
        # Check for extended family indicators
        extended_roles = {"Parent", "Grandparent", "Sibling", "Uncle", "Aunt"}
        if any(role in extended_roles for role in roles):
            return "Extended family"
        
        return "Other arrangement"
    
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
