"""Prompt management utilities."""

import re
from typing import Any, Callable, Optional, Dict
import pandas as pd
from pathlib import Path

from ..core.config import Config
from ..classifiers.base import DemographicClassifier
from ..classifiers.household_size.base import HouseholdSizeClassifier
from ..classifiers.household_type.base import HouseholdCompositionClassifier
from .statistics import StatisticsManager


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
    
    def register_household_size_classifier(self, classifier: HouseholdSizeClassifier,
                                         target_data: Optional[Dict[str, float]] = None,
                                         target_file: Optional[str] = None):
        """Register a household size classifier for HOUSEHOLD_SIZE_STATS placeholder.
        
        Args:
            classifier: Household size classifier instance
            target_data: Target distribution data
            target_file: Path to target data file (relative to data directory)
        """
        self.statistics_manager.register_classifier(
            "HOUSEHOLD_SIZE_STATS", classifier, target_data, target_file
        )
    
    def register_household_composition_classifier(self, classifier: HouseholdCompositionClassifier,
                                                target_data: Optional[Dict[str, float]] = None,
                                                target_file: Optional[str] = None):
        """Register a household composition classifier for HOUSEHOLD_COMPOSITION_STATS placeholder.
        
        Args:
            classifier: Household composition classifier instance
            target_data: Target distribution data
            target_file: Path to target data file (relative to data directory)
        """
        self.statistics_manager.register_classifier(
            "HOUSEHOLD_COMPOSITION_STATS", classifier, target_data, target_file
        )
    
    def register_classifier(self, placeholder: str, classifier: DemographicClassifier,
                          target_data: Optional[Dict[str, float]] = None,
                          target_file: Optional[str] = None):
        """Register any demographic classifier for a placeholder.
        
        Args:
            placeholder: Placeholder name (e.g., "HOUSEHOLD_SIZE_STATS", "CUSTOM_STATS")
            classifier: DemographicClassifier instance
            target_data: Target distribution data
            target_file: Path to target data file (relative to data directory)
        """
        self.statistics_manager.register_classifier(
            placeholder, classifier, target_data, target_file
        )

    def register_custom_statistic(self, placeholder: str, name: str,
                                compute_func: Callable[[pd.DataFrame], Dict[str, float]],
                                target_data: Optional[Dict[str, float]] = None,
                                target_file: Optional[str] = None):
        """Register a custom statistic for any placeholder.
        
        Args:
            placeholder: Placeholder name (e.g., "AGE_STATS", "GENDER_STATS")
            name: Internal name for the statistic
            compute_func: Function to compute statistic from DataFrame
            target_data: Target distribution data
            target_file: Path to target data file
        """
        self.statistics_manager.register_custom_statistic(
            placeholder, name, compute_func, target_data, target_file
        )
    
    def prepare_prompt(
        self,
        base_prompt: str,
        synthetic_df: Optional[pd.DataFrame],
        location: str,
        n_households_generated: int,
        include_stats: bool = True,
        include_guidance: bool = True,
        include_target: bool = True,
        include_avg_household_size: bool = False,
        custom_guidance: Optional[str] = None,
        hh_type_classifier: Optional[HouseholdCompositionClassifier] = None,
        hh_size_classifier: Optional[HouseholdSizeClassifier] = None
    ) -> str:
        """Prepare prompt with statistical feedback and guidance.
        
        Args:
            base_prompt: Base prompt template
            synthetic_df: DataFrame with generated data (for feedback)
            location: Location name
            n_households_generated: Number of households generated so far
            include_stats: Whether to include statistical feedback
            include_guidance: Whether to include guidance
            include_target: Whether to include target distributions
            include_avg_household_size: Whether to include average household size
            custom_guidance: Custom guidance text
            hh_type_classifier: Household type classifier
            hh_size_classifier: Household size classifier
            
        Returns:
            Enhanced prompt with feedback and guidance
        """
        if not include_stats and not include_guidance:
            return base_prompt
            
        # For first batch, no synthetic data available
        if synthetic_df is None or len(synthetic_df) == 0:
            return base_prompt
            
        feedback_parts = []
        
        if include_stats:
            # Add statistical feedback based on generated vs target distributions
            if hh_size_classifier:
                size_feedback = self._generate_size_feedback(
                    synthetic_df, location, hh_size_classifier, include_target
                )
                if size_feedback:
                    feedback_parts.append(size_feedback)
            
            if hh_type_classifier:
                comp_feedback = self._generate_composition_feedback(
                    synthetic_df, location, hh_type_classifier, include_target
                )
                if comp_feedback:
                    feedback_parts.append(comp_feedback)
                    
        if include_guidance and custom_guidance:
            feedback_parts.append(f"Additional Guidance: {custom_guidance}")
            
        if feedback_parts:
            feedback_section = "\n\n" + "\n\n".join(feedback_parts)
            return base_prompt + feedback_section
            
        return base_prompt
    
    def _generate_size_feedback(
        self, 
        synthetic_df: pd.DataFrame, 
        location: str, 
        classifier: HouseholdSizeClassifier,
        include_target: bool = True
    ) -> str:
        """Generate household size distribution feedback."""
        try:
            observed = classifier.compute_observed_distribution(synthetic_df)
            
            if not include_target:
                return f"Household Size Distribution (current): {observed}"
                
            # Would need target data loading here
            # For now, return basic feedback
            return f"Household Size Distribution: {observed}"
            
        except Exception as e:
            print(f"Warning: Could not generate size feedback: {e}")
            return ""
    
    def _generate_composition_feedback(
        self, 
        synthetic_df: pd.DataFrame, 
        location: str, 
        classifier: HouseholdCompositionClassifier,
        include_target: bool = True
    ) -> str:
        """Generate household composition distribution feedback."""
        try:
            observed = classifier.compute_observed_distribution(synthetic_df)
            
            if not include_target:
                return f"Household Composition Distribution (current): {observed}"
                
            # Would need target data loading here
            # For now, return basic feedback
            return f"Household Composition Distribution: {observed}"
            
        except Exception as e:
            print(f"Warning: Could not generate composition feedback: {e}")
            return ""
