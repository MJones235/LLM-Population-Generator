"""Statistic provider implementations."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable
import pandas as pd

from .core import StatisticResult
from ...classifiers.base import DemographicClassifier


class StatisticProvider(ABC):
    """Abstract base class for statistic providers."""
    
    @abstractmethod
    def compute_statistic(self, synthetic_df: pd.DataFrame, **kwargs) -> StatisticResult:
        """Compute the statistic from synthetic data.
        
        Args:
            synthetic_df: DataFrame with synthetic population data
            **kwargs: Additional parameters for computation
            
        Returns:
            StatisticResult with observed and optionally target distributions
        """
        pass
    
    @abstractmethod
    def get_statistic_name(self) -> str:
        """Get the name of this statistic."""
        pass


class ClassifierStatisticProvider(StatisticProvider):
    """Wrapper to use existing classifiers as statistic providers."""
    
    def __init__(self, classifier: DemographicClassifier, 
                 target_data: Optional[Dict[str, float]] = None):
        """Initialize with a classifier.
        
        Args:
            classifier: The classifier to wrap
            target_data: Optional target distribution data
        """
        self.classifier = classifier
        self.target_data = target_data
    
    def compute_statistic(self, synthetic_df: pd.DataFrame, **kwargs) -> StatisticResult:
        """Compute statistic using the wrapped classifier."""
        # Use the classifier's compute_observed_distribution method
        observed = self.classifier.compute_observed_distribution(synthetic_df, **kwargs)
        name = f"{self.classifier.__class__.__name__.lower()}_{self.classifier.get_name()}"
        
        # Get label order from classifier if available
        label_order = None
        if hasattr(self.classifier, 'get_label_order'):
            label_order = self.classifier.get_label_order()
            
        return StatisticResult(
            name=name,
            observed=observed,
            target=self.target_data,
            metadata={"classifier_type": type(self.classifier).__name__},
            label_order=label_order
        )
    
    def get_statistic_name(self) -> str:
        """Get the statistic name."""
        return f"{self.classifier.__class__.__name__.lower()}_{self.classifier.get_name()}"


class CustomStatisticProvider(StatisticProvider):
    """Provider for custom statistic functions."""
    
    def __init__(self, name: str, compute_func: Callable[[pd.DataFrame], Dict[str, float]],
                 target_data: Optional[Dict[str, float]] = None):
        """Initialize with a custom function.
        
        Args:
            name: Name of the statistic
            compute_func: Function that takes DataFrame and returns Dict[str, float]
            target_data: Optional target distribution data
        """
        self.name = name
        self.compute_func = compute_func
        self.target_data = target_data
    
    def compute_statistic(self, synthetic_df: pd.DataFrame, **kwargs) -> StatisticResult:
        """Compute statistic using the custom function."""
        observed = self.compute_func(synthetic_df)
        return StatisticResult(
            name=self.name,
            observed=observed,
            target=self.target_data,
            metadata={"provider_type": "custom_function"}
        )
    
    def get_statistic_name(self) -> str:
        """Get the statistic name."""
        return self.name