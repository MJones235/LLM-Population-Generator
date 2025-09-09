"""Base classes for household size classification."""

from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd


class HouseholdSizeClassifier(ABC):
    """Abstract base class for household size classification."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the classifier name."""
        pass

    @abstractmethod
    def compute_observed_distribution(self, synthetic_df: pd.DataFrame) -> Dict[str, float]:
        """Compute the observed distribution from synthetic data.
        
        Args:
            synthetic_df: DataFrame with synthetic population data
            
        Returns:
            Dictionary mapping size categories to percentages
        """
        pass

    def compute_average_household_size(self, synthetic_df: pd.DataFrame) -> float:
        """Compute the average household size from synthetic data.
        
        Args:
            synthetic_df: DataFrame with synthetic population data
            
        Returns:
            Average household size as float
        """
        household_sizes = synthetic_df.groupby("household_id").size()
        return household_sizes.mean() if not household_sizes.empty else 0.0
