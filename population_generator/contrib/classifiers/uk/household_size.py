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
            "1": "1",
            "2": "2", 
            "3": "3",
            "4": "4",
            "5": "5",
            "6": "6",
            "7": "7",
            "8": "8+"
        }
    
    def get_label_order(self) -> list:
        """Get ordered list of labels for consistent display.
        
        Returns:
            List of labels in logical order
        """
        return ["1", "2", "3", "4", "5", "6", "7", "8+"]
