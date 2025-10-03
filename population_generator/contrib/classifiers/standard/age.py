"""Standard age classifier that can be reused across regions."""

from ....classifiers.base import IndividualLevelClassifier
from typing import Dict, List, Tuple
import pandas as pd


class StandardAgeClassifier(IndividualLevelClassifier):
    """Standard age classifier that can be reused across different regional implementations."""
    
    def get_name(self) -> str:
        """Get classifier name."""
        return "standard_age"

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
    
    def compute_observed_distribution(self, synthetic_df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Compute age distribution from synthetic population data.
        
        Args:
            synthetic_df: DataFrame with synthetic population data
            **kwargs: Additional parameters (unused)
            
        Returns:
            Dictionary mapping age bands to percentages
        """
        if synthetic_df.empty:
            return {}
            
        age_bands = synthetic_df['age'].apply(self._assign_age_band)
        counts = age_bands.value_counts()
        total = len(synthetic_df)
        
        # Convert to percentages
        distribution = {band: (count / total) * 100 for band, count in counts.items()}
        
        # Ensure all bands are represented
        _, labels = self.get_age_band_labels()
        for label in labels:
            if label not in distribution:
                distribution[label] = 0.0
        
        return distribution
    
    def get_label_order(self) -> List[str]:
        """Get ordered list of age band labels.
        
        Returns:
            List of age bands in logical order
        """
        _, labels = self.get_age_band_labels()
        return labels
    
    def get_label_map(self) -> Dict[str, str]:
        """Get mapping from internal labels to display labels.
        
        Returns:
            Dictionary mapping age bands to descriptive labels
        """
        _, labels = self.get_age_band_labels()
        return {label: label for label in labels}