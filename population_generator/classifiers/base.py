"""Base classes for demographic classification."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
import pandas as pd


class DemographicClassifier(ABC):
    """Abstract base class for demographic classification.
    
    This class provides a flexible framework for computing demographic
    distributions from synthetic population data. Subclasses implement
    the specific classification logic for their domain.
    """
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the classifier name/identifier."""
        pass

    @abstractmethod
    def compute_observed_distribution(self, synthetic_df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Compute the observed distribution from synthetic data.
        
        Args:
            synthetic_df: DataFrame with synthetic population data
            **kwargs: Additional parameters specific to the classifier
            
        Returns:
            Dictionary mapping category labels to percentages
        """
        pass

    def get_label_order(self) -> Optional[List[str]]:
        """Get ordered list of category labels for consistent display.
        
        Returns:
            Ordered list of labels, or None if no specific order is needed
        """
        return None
    
    def get_label_map(self) -> Optional[Dict[str, str]]:
        """Get mapping from internal labels to display labels.
        
        Returns:
            Dictionary mapping internal labels to display labels,
            or None if no mapping is needed
        """
        return None
    
    def get_category_description(self, category: str) -> Optional[str]:
        """Get human-readable description for a category.
        
        Args:
            category: Category label
            
        Returns:
            Description of the category, or None if not available
        """
        return None
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get classifier metadata.
        
        Returns:
            Dictionary with classifier information
        """
        return {
            "name": self.get_name(),
            "type": self.__class__.__name__,
            "has_label_order": self.get_label_order() is not None,
            "has_label_map": self.get_label_map() is not None
        }


class HouseholdLevelClassifier(DemographicClassifier):
    """Base class for classifiers that analyze households as units.
    
    This classifier operates on household-level data, grouping individuals
    by household_id and then classifying each household.
    """
    
    @abstractmethod
    def classify_household(self, household_df: pd.DataFrame, **kwargs) -> str:
        """Classify a single household.
        
        Args:
            household_df: DataFrame containing all members of one household
            **kwargs: Additional parameters for classification
            
        Returns:
            String label for the household classification
        """
        pass
    
    def compute_observed_distribution(self, synthetic_df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Compute household-level distribution.
        
        Args:
            synthetic_df: DataFrame with synthetic population data
            **kwargs: Additional parameters for classification
            
        Returns:
            Dictionary mapping household categories to percentages
        """
        if 'household_id' not in synthetic_df.columns:
            raise ValueError("DataFrame must contain 'household_id' column for household-level classification")
        
        # Classify each household
        household_labels = synthetic_df.groupby("household_id", group_keys=False).apply(
            lambda x: self.classify_household(x, **kwargs), include_groups=False
        )
        
        # Compute distribution
        label_counts = household_labels.value_counts(normalize=True) * 100
        distribution = label_counts.to_dict()
        
        # Apply label mapping if available
        label_map = self.get_label_map()
        if label_map:
            distribution = {
                label_map.get(label, label): percentage 
                for label, percentage in distribution.items()
            }
        
        return distribution


class IndividualLevelClassifier(DemographicClassifier):
    """Base class for classifiers that analyze individuals.
    
    This classifier operates on individual-level data, classifying
    each person in the synthetic population.
    """
    
    @abstractmethod
    def classify_individual(self, individual: pd.Series, **kwargs) -> str:
        """Classify a single individual.
        
        Args:
            individual: Series containing individual's data
            **kwargs: Additional parameters for classification
            
        Returns:
            String label for the individual classification
        """
        pass
    
    def compute_observed_distribution(self, synthetic_df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Compute individual-level distribution.
        
        Args:
            synthetic_df: DataFrame with synthetic population data
            **kwargs: Additional parameters for classification
            
        Returns:
            Dictionary mapping individual categories to percentages
        """
        # Classify each individual
        individual_labels = synthetic_df.apply(
            lambda x: self.classify_individual(x, **kwargs), axis=1
        )
        
        # Compute distribution
        label_counts = individual_labels.value_counts(normalize=True) * 100
        distribution = label_counts.to_dict()
        
        # Apply label mapping if available
        label_map = self.get_label_map()
        if label_map:
            distribution = {
                label_map.get(label, label): percentage 
                for label, percentage in distribution.items()
            }
        
        return distribution


class FunctionalClassifier(DemographicClassifier):
    """Classifier that uses a user-provided function for classification.
    
    This provides maximum flexibility by allowing users to define
    their own classification logic through functions.
    """
    
    def __init__(self, 
                 name: str,
                 classify_func: Callable[[pd.DataFrame], Dict[str, float]],
                 label_order: Optional[List[str]] = None,
                 label_map: Optional[Dict[str, str]] = None,
                 description: Optional[str] = None):
        """Initialize functional classifier.
        
        Args:
            name: Classifier name/identifier
            classify_func: Function that takes DataFrame and returns distribution dict
            label_order: Optional ordered list of labels
            label_map: Optional mapping from internal to display labels
            description: Optional description of the classifier
        """
        self.name = name
        self.classify_func = classify_func
        self.label_order = label_order
        self.label_map = label_map
        self.description = description
    
    def get_name(self) -> str:
        """Get classifier name."""
        return self.name
    
    def compute_observed_distribution(self, synthetic_df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Compute distribution using the provided function."""
        distribution = self.classify_func(synthetic_df, **kwargs)
        
        # Apply label mapping if available
        if self.label_map:
            distribution = {
                self.label_map.get(label, label): percentage 
                for label, percentage in distribution.items()
            }
        
        return distribution
    
    def get_label_order(self) -> Optional[List[str]]:
        """Get label order."""
        return self.label_order
    
    def get_label_map(self) -> Optional[Dict[str, str]]:
        """Get label mapping."""
        return self.label_map
    
    def get_category_description(self, category: str) -> Optional[str]:
        """Get category description."""
        return self.description
