"""Standard sex/gender classifier that can be reused across regions."""

from ....classifiers.base import IndividualLevelClassifier
from typing import Dict, List
import pandas as pd


class StandardSexClassifier(IndividualLevelClassifier):
    """Standard sex/gender classifier that can be reused across different regional implementations."""
    
    def get_name(self) -> str:
        """Get classifier name."""
        return "standard_sex"

    def classify_individual(self, individual: pd.Series, **kwargs) -> str:
        """Classify an individual by gender/sex.
        
        Args:
            individual: Series containing individual's data
            **kwargs: Additional parameters (unused)
            
        Returns:
            String label for the gender/sex category
        """
        return individual.get('gender', 'Male')
    
    def get_standard_categories(self) -> List[str]:
        """Get list of standard gender categories.
        
        Returns:
            List of standard gender categories
        """
        return ['Male', 'Female']
    
    def get_label_map(self) -> Dict[str, str]:
        """Get mapping from internal labels to display labels.
        
        Returns:
            Dictionary mapping gender categories to descriptive labels
        """
        return {
            'Male': 'Male',
            'Female': 'Female'
        }
    
    def get_label_order(self) -> List[str]:
        """Get ordered list of labels for consistent display.
        
        Returns:
            List of labels in logical order
        """
        return ['Male', 'Female']