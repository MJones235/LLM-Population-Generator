"""UK Census sex/gender classifier."""

from ....classifiers.base import IndividualLevelClassifier
from typing import Dict, List
import pandas as pd


class UKSexClassifier(IndividualLevelClassifier):
    """Sex/gender classifier for UK Census data."""
    
    def get_name(self) -> str:
        """Get classifier name."""
        return 'uk_census'

    def classify_individual(self, individual: pd.Series, **kwargs) -> str:
        """Classify an individual by gender/sex.
        
        Args:
            individual: Series containing individual's data
            **kwargs: Additional parameters (unused)
            
        Returns:
            String label for the gender/sex category
        """
        gender = individual.get('gender', 'Unknown')
        return self._normalize_gender(gender)
    
    def _normalize_gender(self, gender: str) -> str:
        """Normalize gender values to standard categories.
        
        Args:
            gender: Raw gender value
            
        Returns:
            Normalized gender category
        """
        if pd.isna(gender) or gender == '':
            return 'Unknown'
        
        gender_lower = str(gender).lower().strip()
        
        # Map common variations to standard categories
        if gender_lower in ['male', 'm', 'man', 'boy']:
            return 'Male'
        elif gender_lower in ['female', 'f', 'woman', 'girl']:
            return 'Female'
        else:
            return 'Unknown'
    
    def get_standard_categories(self) -> List[str]:
        """Get list of standard gender categories.
        
        Returns:
            List of standard gender categories
        """
        return ['Male', 'Female', 'Unknown']
    
    def get_label_map(self) -> Dict[str, str]:
        """Get mapping from internal labels to display labels.
        
        Returns:
            Dictionary mapping gender categories to descriptive labels
        """
        return {
            'Male': 'Male',
            'Female': 'Female', 
            'Unknown': 'Unknown/Not specified'
        }
    
    def get_label_order(self) -> List[str]:
        """Get ordered list of labels for consistent display.
        
        Returns:
            List of labels in logical order
        """
        return ['Male', 'Female', 'Unknown']
    
    def compute_observed_distribution(self, synthetic_df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Compute gender/sex distribution.
        
        Args:
            synthetic_df: DataFrame with synthetic population data
            **kwargs: Additional parameters (unused)
            
        Returns:
            Dictionary mapping gender categories to percentages
        """
        if 'gender' not in synthetic_df.columns:
            raise ValueError("DataFrame must contain 'gender' column for sex classification")
        
        # Normalize gender values
        df_copy = synthetic_df.copy()
        df_copy["normalized_gender"] = df_copy["gender"].apply(self._normalize_gender)
        
        # Compute distribution
        distribution = df_copy["normalized_gender"].value_counts(normalize=True) * 100
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
            category: Category label (e.g., "Male")
            
        Returns:
            Description of the category
        """
        descriptions = {
            'Male': 'Male individuals',
            'Female': 'Female individuals',
            'Unknown': 'Unknown or unspecified gender'
        }
        return descriptions.get(category, category)
    