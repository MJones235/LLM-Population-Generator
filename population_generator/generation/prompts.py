"""Prompt management utilities."""

import re
import json
from typing import Any, Callable, Optional, Dict, List
import pandas as pd
from pathlib import Path

from ..engine.config import Config
from ..classifiers.base import DemographicClassifier
from ..analysis.statistics import StatisticsManager


class PromptManager:
    """Manages prompt templates and statistical feedback."""
    
    def __init__(self, config: Config):
        """Initialize prompt manager.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.statistics_manager = StatisticsManager(
            data_dir=config.get("data.data_dir", "data")
        )
    
    def load_prompt(self, filename: str, replacements: Optional[Dict[str, str]] = None) -> str:
        """Load prompt template from file.
        
        Args:
            filename: Prompt template filename
            replacements: Dictionary of replacements to apply
            
        Returns:
            Loaded prompt string
        """
        prompts_dir = self.config.get("data.prompts_dir", "prompts")
        path = Path(prompts_dir) / filename
        
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            prompt = f.read()
            
        if replacements:
            for key, value in replacements.items():
                prompt = prompt.replace(f"{{{key}}}", str(value))
                
        return prompt
    
    def load_schema(self, filename: str) -> Dict[str, Any]:
        """Load JSON schema from file.
        
        Args:
            filename: Schema filename
            
        Returns:
            Loaded schema as dictionary
        """
        schemas_dir = Path(self.config.get("data.data_dir", "data")) / "schemas"
        path = schemas_dir / filename
        
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_prompt_with_statistics(self, filename: str, 
                                  synthetic_df: Optional[pd.DataFrame] = None,
                                  replacements: Optional[Dict[str, str]] = None,
                                  statistics_format: str = "comparison",
                                  **kwargs) -> str:
        """Load prompt template and replace statistics placeholders.
        
        Args:
            filename: Prompt template filename
            synthetic_df: DataFrame with synthetic data for statistics computation
            replacements: Dictionary of basic replacements to apply
            statistics_format: Format for statistics ("comparison", "observed", "target")
            **kwargs: Additional parameters for statistics computation
            
        Returns:
            Loaded prompt string with statistics placeholders replaced
        """
        # Load base prompt
        prompt = self.load_prompt(filename, replacements)
        
        # Replace statistics placeholders
        prompt = self.statistics_manager.replace_placeholders_in_prompt(
            prompt, synthetic_df, statistics_format, **kwargs
        )
        
        return prompt
    

    
    def register_classifier(self, placeholder: str, classifier: DemographicClassifier,
                          target_data: Optional[Dict[str, float]] = None,
                          target_file: Optional[str] = None,
                          format_type: str = "comparison"):
        """Register any demographic classifier for a placeholder.
        
        Args:
            placeholder: Placeholder name (e.g., "HOUSEHOLD_SIZE_STATS", "CUSTOM_STATS")
            classifier: DemographicClassifier instance
            target_data: Target distribution data
            target_file: Path to target data file (relative to data directory)
            format_type: Format type for this classifier ("comparison", "observed", "target")
        """
        self.statistics_manager.register_classifier(
            placeholder, classifier, target_data, target_file, format_type
        )

    def register_custom_statistic(self, placeholder: str, name: str,
                                compute_func: Callable[[pd.DataFrame], Dict[str, float]],
                                target_data: Optional[Dict[str, float]] = None,
                                target_file: Optional[str] = None,
                                format_type: str = "comparison"):
        """Register a custom statistic for any placeholder.
        
        Args:
            placeholder: Placeholder name (e.g., "AGE_STATS", "GENDER_STATS")
            name: Internal name for the statistic
            compute_func: Function to compute statistic from DataFrame
            target_data: Target distribution data
            target_file: Path to target data file
            format_type: Format type for this statistic ("comparison", "observed", "target")
        """
        self.statistics_manager.register_custom_statistic(
            placeholder, name, compute_func, target_data, target_file, format_type
        )
    
    def clear_classifiers(self):
        """Clear all registered classifiers and statistics."""
        self.statistics_manager.clear_all()
    
    def list_registered_placeholders(self) -> List[str]:
        """Get list of all registered placeholders.
        
        Returns:
            List of placeholder names
        """
        return list(self.statistics_manager.providers.keys())
    
    def prepare_prompt_with_feedback(
        self,
        base_prompt: str,
        synthetic_df: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> str:
        """Prepare prompt with statistical feedback using registered classifiers.
        
        This method uses the statistics manager to replace any registered
        placeholders in the prompt with actual statistics computed from
        synthetic_df.
        
        Args:
            base_prompt: Base prompt template
            synthetic_df: DataFrame with generated data (for feedback)
            **kwargs: Additional parameters passed to statistics computation
            
        Returns:
            Enhanced prompt with statistical feedback
        """
        # Use the statistics manager to replace all registered placeholders
        # For first batch (empty data), it will show target distributions only
        return self.statistics_manager.replace_placeholders_in_prompt(
            base_prompt, synthetic_df, format_type="comparison", **kwargs
        )
    

