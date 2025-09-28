"""Standard age/sex pyramid classifier that can be reused across regions."""

from ....classifiers.base import IndividualLevelClassifier
from typing import Dict, List, Tuple
import pandas as pd


class StandardAgeSexPyramidClassifier(IndividualLevelClassifier):
    """Standard age/sex pyramid classifier that can be reused across different regional implementations."""
    
    def __init__(self, region_name: str):
        """Initialize with region name for identification.
        
        Args:
            region_name: Name/identifier for the region (e.g., 'uk_census', 'unpd')
        """
        self.region_name = region_name
    
    def get_name(self) -> str:
        """Get classifier name."""
        return self.region_name

    def classify_individual(self, individual: pd.Series, **kwargs) -> str:
        """Classify an individual by age band and gender.
        
        Args:
            individual: Series containing individual's data
            **kwargs: Additional parameters (unused)
            
        Returns:
            String label for the age/sex category (e.g., "Male_30-39")
        """
        age = individual['age']
        gender = individual.get('gender', 'Male')
        
        age_band = self._assign_age_band(age)
        return f"{gender}_{age_band}"
    
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
        
        Can be overridden by subclasses for region-specific age bands.
        
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
    
    def get_gender_categories(self) -> List[str]:
        """Get list of gender categories.
        
        Returns:
            List of gender categories
        """
        return ['Male', 'Female']
    
    def compute_observed_distribution(self, synthetic_df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Compute age/sex distribution from synthetic population data.
        
        Args:
            synthetic_df: DataFrame with synthetic population data
            **kwargs: Additional parameters (unused)
            
        Returns:
            Dictionary mapping age/sex combinations to percentages
        """
        if synthetic_df.empty:
            return {}
            
        # Create age/sex combinations
        age_bands = synthetic_df['age'].apply(self._assign_age_band)
        age_sex_combinations = synthetic_df['gender'] + '_' + age_bands
        
        counts = age_sex_combinations.value_counts()
        total = len(synthetic_df)
        
        # Convert to percentages
        distribution = {combo: (count / total) * 100 for combo, count in counts.items()}
        
        # Ensure all combinations are represented
        _, age_labels = self.get_age_band_labels()
        gender_categories = self.get_gender_categories()
        
        for gender in gender_categories:
            for age_band in age_labels:
                combo = f"{gender}_{age_band}"
                if combo not in distribution:
                    distribution[combo] = 0.0
        
        return distribution
    
    def get_label_order(self) -> List[str]:
        """Get ordered list of age/sex combination labels.
        
        Returns:
            List of labels in logical order (by gender, then age)
        """
        _, age_labels = self.get_age_band_labels()
        gender_categories = self.get_gender_categories()
        
        labels = []
        for gender in gender_categories:
            for age_band in age_labels:
                labels.append(f"{gender}_{age_band}")
        
        return labels
    
    def get_label_map(self) -> Dict[str, str]:
        """Get mapping from internal labels to display labels.
        
        Returns:
            Dictionary mapping age/sex combinations to descriptive labels
        """
        _, age_labels = self.get_age_band_labels()
        gender_categories = self.get_gender_categories()
        
        label_map = {}
        for gender in gender_categories:
            for age_band in age_labels:
                combo = f"{gender}_{age_band}"
                label_map[combo] = f"{gender} {age_band}"
        
        return label_map