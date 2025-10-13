"""UN Population Division household size classifier."""

from ....classifiers.base import HouseholdLevelClassifier
from typing import Dict
import pandas as pd


class UNPDHouseholdSizeClassifier(HouseholdLevelClassifier):
    """Household size classifier for UN Population Division global data."""
    
    def get_name(self) -> str:
        """Get classifier name."""
        return 'unpd'

    def classify_household(self, household_df: pd.DataFrame, **kwargs) -> str:
        """Classify a household by size using global demographic patterns.
        
        Args:
            household_df: DataFrame containing all members of one household
            **kwargs: Additional parameters (unused)
            
        Returns:
            String label for the household size category
        """
        size = len(household_df)
        return self._bucket_size(size)
    
    def _bucket_size(self, size: int) -> str:
        """Bucket household size into global demographic categories.
        
        Args:
            size: Number of people in household
            
        Returns:
            Size bucket label
        """
        if size == 1:
            return "1"
        elif size <= 3:
            return "2-3"
        elif size <= 5:
            return "4-5"
        else:
            return "6+"
    
    def compute_observed_distribution(self, synthetic_df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Compute household size distribution from synthetic population data.
        
        Uses UN global demographic patterns with broader size categories
        than UK Census data.
        
        Args:
            synthetic_df: DataFrame with synthetic population data
            **kwargs: Additional parameters (unused)
            
        Returns:
            Dictionary mapping size buckets to percentages
        """
        if synthetic_df.empty:
            return {bucket: 0.0 for bucket in ["1", "2-3", "4-5", "6+"]}
            
        # Group by household to get household sizes
        household_sizes = synthetic_df.groupby("household_id").size()
        
        # Apply bucketing function
        size_buckets = household_sizes.apply(self._bucket_size)
        bucket_counts = size_buckets.value_counts().to_dict()
        total = sum(bucket_counts.values())
        
        # Ensure all buckets are represented
        buckets = ["1", "2-3", "4-5", "6+"]
        return {
            bucket: round((bucket_counts.get(bucket, 0) / total) * 100, 2) if total > 0 else 0.0
            for bucket in buckets
        }
    
    def get_label_map(self) -> Dict[str, str]:
        """Get mapping from internal labels to display labels.
        
        Returns:
            Dictionary mapping size buckets to descriptive labels
        """
        return {
            "1": "1",
            "2-3": "2-3",
            "4-5": "4-5",
            "6+": "6+"
        }
    
    def get_label_order(self) -> list:
        """Get ordered list of labels for consistent display.
        
        Returns:
            List of size buckets in logical order
        """
        return ["1", "2-3", "4-5", "6+"]