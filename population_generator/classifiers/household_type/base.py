"""Base classes for household composition classification."""

from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd


class HouseholdCompositionClassifier(ABC):
    """Abstract base class for household composition classification."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the classifier name."""
        pass

    @abstractmethod
    def compute_observed_distribution(self, synthetic_df: pd.DataFrame, relationship_col: str = "relationship") -> Dict[str, float]:
        """Compute the observed distribution from synthetic data.
        
        Args:
            synthetic_df: DataFrame with synthetic population data
            relationship_col: Column name containing relationship information
            
        Returns:
            Dictionary mapping composition types to percentages
        """
        pass

    @abstractmethod
    def get_label_order(self) -> List[str]:
        """Get the ordered list of composition labels.
        
        Returns:
            List of composition category labels in display order
        """
        pass

    @abstractmethod
    def classify_household_structure(self, group: pd.DataFrame, relationship_col: str = "relationship") -> str:
        """Classify a household's structure based on member relationships.
        
        Args:
            group: DataFrame containing household members
            relationship_col: Column name containing relationship information
            
        Returns:
            String label for household composition type
        """
        pass
