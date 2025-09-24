"""UK Census age classifier."""

from ....classifiers.base import IndividualLevelClassifier
from typing import Dict, List, Tuple
import pandas as pd


class UKAgeClassifier(IndividualLevelClassifier):
    """Age classifier for UK Census data."""
    
    def get_name(self) -> str:
        """Get classifier name."""
        return 'uk_census'

    def classify_individual(self, individual: pd.Series, **kwargs) -> str:
        """Classify an individual by age band.
        
        Args:
            individual: Series containing individual's data
            **kwargs: Additional parameters (unused)
            
        Returns:
            String label for the age category (e.g., "30-39")
        """
        age = individual['age']
        return self._assign_age_band(age)
    
    def _assign_age_band(self, age: int) -> str:
        """Assign age band to a single age value.
        
        Args:
            age: Age in years
            
        Returns:
            Age band label (e.g., "30-39")
        """
        bins, labels = self.get_age_band_labels()
        
        for i, upper_bound in enumerate(bins[1:]):
            if age < upper_bound:
                return labels[i]
        
        # If age is >= last bin, assign to last label
        return labels[-1]
    
    def assign_age_band(self, age_series: pd.Series) -> pd.Series:
        """Assigns age bands to a series of ages using predefined bins.
        
        Args:
            age_series: Series of ages
            
        Returns:
            Series with age band labels
        """
        bins, labels = self.get_age_band_labels()
        return pd.cut(age_series, bins=bins, labels=labels, right=False)
    
    def get_age_band_labels(self) -> Tuple[List[int], List[str]]:
        """Returns bin edges and labels for broad demographic age groups.
        
        Returns:
            Tuple of (bin_edges, age_band_labels)
        """
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, float("inf")]
        labels = [
            "0-9",
            "10-19", 
            "20-29",
            "30-39",
            "40-49",
            "50-59",
            "60-69",
            "70-79",
            "80+",
        ]
        return bins, labels
    
    def get_label_map(self) -> Dict[str, str]:
        """Get mapping from internal labels to display labels.
        
        Returns:
            Dictionary mapping age categories to descriptive labels
        """
        _, age_labels = self.get_age_band_labels()
        
        return {
            age_label: f"{age_label} years"
            for age_label in age_labels
        }
    
    def get_label_order(self) -> List[str]:
        """Get ordered list of labels for consistent display.
        
        Returns:
            List of labels ordered by age (youngest to oldest)
        """
        _, age_labels = self.get_age_band_labels()
        return age_labels
    
    def compute_observed_distribution(self, synthetic_df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Compute age distribution.
        
        Args:
            synthetic_df: DataFrame with synthetic population data
            **kwargs: Additional parameters (unused)
            
        Returns:
            Dictionary mapping age categories to percentages
        """
        if 'age' not in synthetic_df.columns:
            raise ValueError("DataFrame must contain 'age' column for age classification")
        
        # Create age bands
        df_copy = synthetic_df.copy()
        df_copy["age_band"] = self.assign_age_band(df_copy["age"])
        
        # Compute distribution
        distribution = df_copy["age_band"].value_counts(normalize=True) * 100
        distribution_dict = distribution.round(2).to_dict()
        
        # Ensure all expected categories are present (fill missing with 0.0)
        all_labels = self.get_label_order()
        for label in all_labels:
            if label not in distribution_dict:
                distribution_dict[label] = 0.0
                
        return distribution_dict
    
    def get_category_description(self, category: str) -> str:
        """Get human-readable description for a category.
        
        Args:
            category: Category label (e.g., "30-39")
            
        Returns:
            Description of the category
        """
        return f"People aged {category} years"
