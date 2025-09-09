"""UK Census household size classifier."""

from ....classifiers.base import HouseholdLevelClassifier
from typing import Dict
import pandas as pd


class UKHouseholdSizeClassifier(HouseholdLevelClassifier):
    """Household size classifier for UK Census data."""
    
    def get_name(self) -> str:
        """Get classifier name."""
        return 'uk_census'

    def classify_household(self, household_df: pd.DataFrame, **kwargs) -> str:
        """Classify a household by size.
        
        Args:
            household_df: DataFrame containing all members of one household
            **kwargs: Additional parameters (unused)
            
        Returns:
            String label for the household size category
        """
        size = len(household_df)
        # UK Census typically groups sizes 8+ together
        if size >= 8:
            return "8"
        return str(size)
    
    def get_label_map(self) -> Dict[str, str]:
        """Get mapping from internal labels to display labels.
        
        Returns:
            Dictionary mapping size numbers to descriptive labels
        """
        return {
            "1": "1 person",
            "2": "2 people", 
            "3": "3 people",
            "4": "4 people",
            "5": "5 people",
            "6": "6 people",
            "7": "7 people",
            "8": "8+ people"
        }
    
    def get_label_order(self) -> list:
        """Get ordered list of labels for consistent display.
        
        Returns:
            List of labels in logical order
        """
        return ["1", "2", "3", "4", "5", "6", "7", "8"]

    def compute_observed_distribution(self, synthetic_df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Compute household size distribution for UK Census categories.
        
        Args:
            synthetic_df: DataFrame with synthetic population data
            **kwargs: Additional parameters (unused)
            
        Returns:
            Dictionary mapping size categories to percentages
        """
        household_sizes = synthetic_df.groupby("household_id").size()
        
        # UK Census typically uses size categories 1-8, with 8+ grouped
        household_sizes = household_sizes.apply(lambda x: x if x <= 8 else 8)
        size_counts = household_sizes.value_counts().to_dict()
        total = sum(size_counts.values())
        
        # Return as percentages for sizes 1-8
        return {
            str(size): round((size_counts.get(size, 0) / total) * 100, 2) if total > 0 else 0.00
            for size in range(1, 9)
        }
