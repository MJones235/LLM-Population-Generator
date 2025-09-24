"""UK Census age/sex pyramid classifier."""

from ....classifiers.base import IndividualLevelClassifier
from typing import Dict, List, Tuple
import pandas as pd


class UKAgeSexPyramidClassifier(IndividualLevelClassifier):
    """Age/sex pyramid classifier for UK Census data."""
    
    def get_name(self) -> str:
        """Get classifier name."""
        return 'uk_census'

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
        
        Returns:
            Tuple of (bin_edges, age_band_labels)
        """
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, float("inf")]
        labels = [
            "0–9",
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
            Dictionary mapping age/sex categories to descriptive labels
        """
        _, age_labels = self.get_age_band_labels()
        genders = ['Male', 'Female']
        
        label_map = {}
        for gender in genders:
            for age_label in age_labels:
                internal_label = f"{gender}_{age_label}"
                display_label = f"{gender} {age_label}"
                label_map[internal_label] = display_label
                
        return label_map
    
    def get_label_order(self) -> List[str]:
        """Get ordered list of labels for consistent display.
        
        Returns:
            List of labels ordered by gender then age (suitable for pyramid display)
        """
        _, age_labels = self.get_age_band_labels()
        genders = ['Male', 'Female']
        
        ordered_labels = []
        for gender in genders:
            for age_label in age_labels:
                ordered_labels.append(f"{gender}_{age_label}")
                
        return ordered_labels
    
    def compute_observed_distribution(self, synthetic_df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Compute age/sex pyramid distribution.
        
        Args:
            synthetic_df: DataFrame with synthetic population data
            **kwargs: Additional parameters (unused)
            
        Returns:
            Dictionary mapping age/sex categories to percentages
        """
        if 'age' not in synthetic_df.columns or 'gender' not in synthetic_df.columns:
            raise ValueError("DataFrame must contain 'age' and 'gender' columns for age/sex classification")
        
        # Create age bands
        df_copy = synthetic_df.copy()
        df_copy["age_band"] = self.assign_age_band(df_copy["age"])
        
        # Create age/sex categories
        df_copy["age_sex_category"] = df_copy["gender"] + "_" + df_copy["age_band"].astype(str)
        
        # Compute distribution
        distribution = df_copy["age_sex_category"].value_counts(normalize=True) * 100
        distribution_dict = distribution.round(2).to_dict()
        
        # Ensure all expected categories are present (fill missing with 0.0)
        all_labels = self.get_label_order()
        for label in all_labels:
            if label not in distribution_dict:
                distribution_dict[label] = 0.0
                
        return distribution_dict
    
    def compute_age_distribution_only(self, synthetic_df: pd.DataFrame) -> Dict[str, float]:
        """Compute age distribution without gender breakdown.
        
        Args:
            synthetic_df: DataFrame with synthetic population data
            
        Returns:
            Dictionary mapping age bands to percentages
        """
        if 'age' not in synthetic_df.columns:
            raise ValueError("DataFrame must contain 'age' column")
            
        df_copy = synthetic_df.copy()
        df_copy["age_band"] = self.assign_age_band(df_copy["age"])
        
        distribution = df_copy["age_band"].value_counts(normalize=True) * 100
        return distribution.round(2).to_dict()
    
    def get_category_description(self, category: str) -> str:
        """Get human-readable description for a category.
        
        Args:
            category: Category label (e.g., "Male_30-39")
            
        Returns:
            Description of the category
        """
        if '_' in category:
            gender, age_band = category.split('_', 1)
            return f"{gender} aged {age_band} years"
        return category
