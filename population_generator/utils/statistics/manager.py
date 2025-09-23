"""Statistics management and orchestration."""

from typing import Dict, Optional, Callable, Any
from pathlib import Path
import pandas as pd

from .core import StatisticResult
from .providers import StatisticProvider, ClassifierStatisticProvider, CustomStatisticProvider
from .fit_metrics import DistributionalFitCalculator
from .formatters import StatisticFormatter
from .reporting import FitReporter
from ...classifiers.base import DemographicClassifier
from ..data_loading import FlexibleDataManager


class StatisticsManager:
    """Manages statistics providers and placeholder replacements."""
    
    def __init__(self, data_dir: Optional[str] = None, compute_fit_metrics: bool = True):
        """Initialize statistics manager.
        
        Args:
            data_dir: Directory containing target data files
            compute_fit_metrics: Whether to automatically compute fit metrics
        """
        self.providers: Dict[str, StatisticProvider] = {}
        self.placeholder_mappings: Dict[str, str] = {}
        self.data_dir = Path(data_dir) if data_dir else None
        self.data_manager = FlexibleDataManager()
        self.compute_fit_metrics = compute_fit_metrics
        self.fit_calculator = DistributionalFitCalculator()
        self.formatter = StatisticFormatter()
        self.reporter = FitReporter()
        
        # Default placeholder mappings
        self._setup_default_mappings()
    
    def _setup_default_mappings(self):
        """Set up default placeholder to statistic mappings."""
        self.placeholder_mappings.update({
            "HOUSEHOLD_SIZE_STATS": "household_size_stats",
            "HOUSEHOLD_COMPOSITION_STATS": "household_composition_stats",
            "AGE_STATS": "age_stats",
            "SEX_STATS": "sex_stats"
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
        sample_size = len(synthetic_df)
        
        for name, provider in self.providers.items():
            try:
                result = provider.compute_statistic(synthetic_df, **kwargs)
                
                # Compute fit metrics if target data available
                if self.compute_fit_metrics and result.target:
                    fit_metrics = self.fit_calculator.compute_all_metrics(
                        result.observed, result.target, sample_size
                    )
                    result.fit_metrics = fit_metrics
                
                results[name] = result
                
            except Exception as e:
                print(f"Warning: Error computing statistic {name}: {e}")
        
        return results
    
    def replace_placeholders_in_prompt(self, prompt: str, synthetic_df: Optional[pd.DataFrame] = None,
                                     format_type: str = "comparison", threshold: Optional[float] = 0.5, **kwargs) -> str:
        """Replace statistics placeholders in a prompt.
        
        Args:
            prompt: The prompt template with placeholders
            synthetic_df: DataFrame with synthetic data (None for first batch)
            format_type: How to format statistics ("comparison", "observed", "target")
            threshold: Threshold for showing guidance text (None to disable guidance)
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
                stat_text = self.formatter.format_statistic_text(results[statistic_name], format_type, threshold)
                prompt = prompt.replace(f"{{{placeholder}}}", stat_text)
            else:
                # Remove placeholder if no statistic available
                prompt = prompt.replace(f"{{{placeholder}}}", "")
        
        return prompt
    
    def clear_all(self):
        """Clear all registered statistic providers."""
        self.providers.clear()
    
    def get_overall_fit_summary(self, results: Dict[str, StatisticResult]) -> Dict[str, Any]:
        """Get summary of fit metrics across all statistics."""
        return self.reporter.get_overall_fit_summary(results)
    
    def format_fit_summary(self, summary: Dict[str, Any]) -> str:
        """Format fit summary as readable text."""
        return self.reporter.format_fit_summary(summary)