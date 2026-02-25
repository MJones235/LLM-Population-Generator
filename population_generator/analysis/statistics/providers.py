"""Statistic provider implementations."""

import logging
import traceback
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
                 target_data: Optional[Dict[str, float]] = None,
                 format_type: str = "comparison"):
        """Initialize with a classifier.
        
        Args:
            classifier: The classifier to wrap
            target_data: Optional target distribution data
            format_type: Format type for this classifier ("comparison", "observed", "target")
        """
        self.classifier = classifier
        self.target_data = target_data
        self.format_type = format_type
    
    def compute_statistic(self, synthetic_df: pd.DataFrame, **kwargs) -> StatisticResult:
        """Compute statistic using the wrapped classifier."""
        try:
            observed = self.classifier.compute_observed_distribution(synthetic_df, **kwargs)
            name = f"{self.classifier.__class__.__name__.lower()}_{self.classifier.get_name()}"
        except Exception as e:
            logging.error(f"Error in compute_statistic for classifier {self.classifier.__class__.__name__}:")
            logging.error(f"  Error: {str(e)}")
            logging.error(f"  synthetic_df shape: {synthetic_df.shape}")
            logging.error(f"  synthetic_df columns: {list(synthetic_df.columns)}")
            logging.error(f"  synthetic_df dtypes: {dict(synthetic_df.dtypes)}")
            logging.error(f"  Stack trace: {traceback.format_exc()}")
            raise  # Re-raise the error
        
        # Get label order from classifier if available
        label_order = None
        if hasattr(self.classifier, 'get_label_order'):
            label_order = self.classifier.get_label_order()
        
        # Build metadata with classifier info
        metadata = {"classifier_type": type(self.classifier).__name__}
        
        # Add data type if available (for FunctionalClassifier)
        if hasattr(self.classifier, 'data_type'):
            metadata['data_type'] = self.classifier.data_type
        else:
            metadata['data_type'] = 'percentage'  # Default for other classifiers
        
        # Add threshold if available (for any classifier with threshold attribute)
        if hasattr(self.classifier, 'threshold') and self.classifier.threshold is not None:
            metadata['threshold'] = self.classifier.threshold
            
        return StatisticResult(
            name=name,
            observed=observed,
            target=self.target_data,
            metadata=metadata,
            label_order=label_order
        )
    
    def get_statistic_name(self) -> str:
        """Get the statistic name."""
        return f"{self.classifier.__class__.__name__.lower()}_{self.classifier.get_name()}"


class CustomStatisticProvider(StatisticProvider):
    """Provider for custom statistic functions."""
    
    def __init__(self, name: str, compute_func: Callable[[pd.DataFrame], Dict[str, float]],
                 target_data: Optional[Dict[str, float]] = None,
                 format_type: str = "comparison"):
        """Initialize with a custom function.
        
        Args:
            name: Name of the statistic
            compute_func: Function that takes DataFrame and returns Dict[str, float]
            target_data: Optional target distribution data
            format_type: Format type for this statistic ("comparison", "observed", "target")
        """
        self.name = name
        self.compute_func = compute_func
        self.target_data = target_data
        self.format_type = format_type
    
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