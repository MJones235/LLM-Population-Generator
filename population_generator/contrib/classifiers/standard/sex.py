"""Standard sex/gender classifier that can be reused across regions."""

from ....classifiers.base import IndividualLevelClassifier
from typing import Dict, List
import pandas as pd


class StandardSexClassifier(IndividualLevelClassifier):
    """Standard sex/gender classifier that can be reused across different regional implementations."""
    
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