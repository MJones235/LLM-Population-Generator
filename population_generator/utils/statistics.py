"""Statistics management for prompt placeholders."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass
import pandas as pd
from pathlib import Path

from ..classifiers.base import DemographicClassifier
from .data_loading import FlexibleDataManager


@dataclass
class StatisticResult:
    """Container for a computed statistic."""
    name: str
    observed: Dict[str, float]
    target: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    

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
            
        return StatisticResult(
            name=name,
            observed=observed,
            target=self.target_data,
            metadata={"classifier_type": type(self.classifier).__name__}
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


class StatisticsManager:
    """Manages statistics providers and placeholder replacements."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize statistics manager.
        
        Args:
            data_dir: Directory containing target data files
        """
        self.providers: Dict[str, StatisticProvider] = {}
        self.placeholder_mappings: Dict[str, str] = {}
        self.data_dir = Path(data_dir) if data_dir else None
        self.data_manager = FlexibleDataManager()
        
        # Default placeholder mappings
        self._setup_default_mappings()
    
    def _setup_default_mappings(self):
        """Set up default placeholder to statistic mappings."""
        self.placeholder_mappings.update({
            "HOUSEHOLD_SIZE_STATS": "household_size_stats",
            "HOUSEHOLD_COMPOSITION_STATS": "household_composition_stats",
            "AGE_STATS": "age_stats",
            "GENDER_STATS": "gender_stats"
        })
    
    def register_classifier(self, placeholder: str, 
                          classifier: DemographicClassifier,
                          target_data: Optional[Dict[str, float]] = None,
                          target_file: Optional[str] = None):
        """Register a classifier as a statistic provider.
        
        Args:
            placeholder: Placeholder name (e.g., "HOUSEHOLD_SIZE_STATS")
            classifier: The classifier instance
            target_data: Target distribution data
            target_file: Path to file containing target data (relative to data_dir)
        """
        # Load target data from file if specified
        if target_file and self.data_dir:
            target_data = self._load_target_data(target_file)
        
        provider = ClassifierStatisticProvider(classifier, target_data)
        statistic_name = provider.get_statistic_name()
        
        self.providers[statistic_name] = provider
        self.placeholder_mappings[placeholder] = statistic_name
    
    def register_custom_statistic(self, placeholder: str, name: str,
                                compute_func: Callable[[pd.DataFrame], Dict[str, float]],
                                target_data: Optional[Dict[str, float]] = None,
                                target_file: Optional[str] = None):
        """Register a custom statistic function.
        
        Args:
            placeholder: Placeholder name (e.g., "AGE_STATS")
            name: Internal name for the statistic
            compute_func: Function to compute the statistic
            target_data: Target distribution data
            target_file: Path to file containing target data
        """
        # Load target data from file if specified
        if target_file and self.data_dir:
            target_data = self._load_target_data(target_file)
            
        provider = CustomStatisticProvider(name, compute_func, target_data)
        
        self.providers[name] = provider
        self.placeholder_mappings[placeholder] = name
    
    def _load_target_data(self, filename: str) -> Optional[Dict[str, float]]:
        """Load target data from file using flexible data loading system.
        
        Args:
            filename: Name of the target data file
            
        Returns:
            Dictionary with target distribution or None if file not found
        """
        if not self.data_dir:
            return None
            
        file_path = self.data_dir / filename
        if not file_path.exists():
            print(f"Warning: Target data file not found: {file_path}")
            return None
            
        try:
            # Use flexible data manager to load any supported format
            data = self.data_manager.load_target_data(file_path)
            
            # Get metadata for logging
            metadata = self.data_manager.get_metadata(file_path)
            if metadata:
                print(f"Loaded target data: {metadata.name} ({metadata.source})")
                if metadata.description:
                    print(f"  Description: {metadata.description}")
            
            return data
            
        except Exception as e:
            print(f"Warning: Error loading target data from {filename}: {e}")
            return None
    
    def compute_all_statistics(self, synthetic_df: pd.DataFrame, **kwargs) -> Dict[str, StatisticResult]:
        """Compute all registered statistics.
        
        Args:
            synthetic_df: DataFrame with synthetic population data
            **kwargs: Additional parameters for statistic computation
            
        Returns:
            Dictionary mapping statistic names to results
        """
        results = {}
        for name, provider in self.providers.items():
            try:
                results[name] = provider.compute_statistic(synthetic_df, **kwargs)
            except Exception as e:
                print(f"Warning: Error computing statistic {name}: {e}")
        return results
    
    def format_statistic_text(self, result: StatisticResult, format_type: str = "comparison") -> str:
        """Format a statistic result as text.
        
        Args:
            result: The statistic result to format
            format_type: Format type ("comparison", "observed", "target")
            
        Returns:
            Formatted text representation
        """
        if format_type == "observed":
            return self._format_distribution(result.observed, "Current")
        elif format_type == "target" and result.target:
            return self._format_distribution(result.target, "Target")
        elif format_type == "comparison" and result.target:
            return self._format_comparison(result.observed, result.target)
        else:
            return self._format_distribution(result.observed, "Current")
    
    def _format_distribution(self, distribution: Dict[str, float], label: str) -> str:
        """Format a single distribution."""
        items = [f"{k}: {v}%" for k, v in distribution.items()]
        return f"{label} Distribution: {', '.join(items)}"
    
    def _format_comparison(self, observed: Dict[str, float], target: Dict[str, float]) -> str:
        """Format a comparison between observed and target distributions."""
        lines = ["Distribution Comparison:"]
        all_keys = set(observed.keys()) | set(target.keys())
        
        for key in sorted(all_keys):
            obs_val = observed.get(key, 0.0)
            tgt_val = target.get(key, 0.0)
            diff = obs_val - tgt_val
            sign = "+" if diff > 0 else ""
            lines.append(f"  {key}: Target {tgt_val}% → Current {obs_val}% ({sign}{diff:.1f}%)")
        
        return "\n".join(lines)
    
    def replace_placeholders_in_prompt(self, prompt: str, synthetic_df: Optional[pd.DataFrame] = None,
                                     format_type: str = "comparison", **kwargs) -> str:
        """Replace statistics placeholders in a prompt.
        
        Args:
            prompt: The prompt template with placeholders
            synthetic_df: DataFrame with synthetic data (None for first batch)
            format_type: How to format statistics ("comparison", "observed", "target")
            **kwargs: Additional parameters for statistic computation
            
        Returns:
            Prompt with placeholders replaced
        """
        if synthetic_df is None or len(synthetic_df) == 0:
            # For first batch, remove placeholders or replace with empty string
            for placeholder in self.placeholder_mappings.keys():
                prompt = prompt.replace(f"{{{placeholder}}}", "")
            return prompt
        
        # Compute all statistics
        results = self.compute_all_statistics(synthetic_df, **kwargs)
        
        # Replace each placeholder
        for placeholder, statistic_name in self.placeholder_mappings.items():
            if statistic_name in results:
                stat_text = self.format_statistic_text(results[statistic_name], format_type)
                prompt = prompt.replace(f"{{{placeholder}}}", stat_text)
            else:
                # Remove placeholder if no statistic available
                prompt = prompt.replace(f"{{{placeholder}}}", "")
        
        return prompt
    
    def clear_all(self):
        """Clear all registered statistic providers."""
        self.providers.clear()
