"""Batch processing for population generation."""

import logging
import traceback
from typing import Any, Dict, List, Optional
import pandas as pd

from ..engine.config import Config
from ..llm.base import BaseLLM
from ..data.validation import CustomValidator
from ..analysis.costs import TokenAnalyzer
from .prompts import PromptManager


class BatchProcessor:
    """Handles batch processing of population generation requests."""
    
    def __init__(self, config, prompt_manager: PromptManager, token_analyzer: Optional[TokenAnalyzer] = None):
        """Initialize batch processor.
        
        Args:
            config: Configuration object
            prompt_manager: Prompt manager instance
            token_analyzer: Optional token analyzer for cost tracking
        """
        self.config = config
        self.prompt_manager = prompt_manager
        self.token_analyzer = token_analyzer
    
    def prepare_batch_prompts(self, 
                             prompt_template: str, 
                             batch_count: int) -> List[str]:
        """Prepare prompts for a batch of households.
        
        Args:
            prompt_template: Template prompt to use
            batch_count: Number of households in this batch
            
        Returns:
            List of prompts for the batch
        """
        batch_prompts = []
        for i in range(batch_count):
            batch_prompts.append(prompt_template)
        return batch_prompts
    
    def run_batch(self, 
                  model: BaseLLM, 
                  prompts: List[str], 
                  schema: Dict[str, Any], 
                  custom_validator: Optional[CustomValidator] = None,
                  households_generated: int = 0, 
                  total_households: int = 0) -> List[Dict[str, Any]]:
        """Run a batch of prompts through the LLM.
        
        Args:
            model: LLM instance to use
            prompts: List of prompts to process
            schema: JSON schema for validation
            custom_validator: Optional custom validator
            households_generated: Number of households already generated
            total_households: Total target number of households
            
        Returns:
            List of generated household dictionaries
        """
        try:
            timeout = self.config.get("generation.default_timeout", 60)
            
            # Enable token tracking on the model if we have a token analyzer
            if self.token_analyzer:
                model.enable_token_tracking(True)
            
            results = []
            
            # Process each prompt individually to show progress
            for i, prompt in enumerate(prompts):
                household_num = households_generated + i + 1
                print(f"  Generating household {household_num}/{total_households}...", end=" ", flush=True)
                
                try:
                    result = model.generate_json(prompt, schema, custom_validator, timeout=timeout)
                    results.append(result)
                    print("✓")
                except Exception as e:
                    print(f"✗ (Failed: {str(e)})")
                    results.append({})  # Return empty dict on failure
            
            # Record token usage if tracking is enabled
            if self.token_analyzer and model.track_tokens and model.token_usage_history:
                self._record_token_usage(model, len(prompts))
            
            return results
        except Exception as e:
            print(f"[ERROR] Batch generation failed: {e}")
            return []
    
    def update_prompt_with_statistics(self, 
                                    base_prompt: str, 
                                    location: str, 
                                    households: List[Dict[str, Any]]) -> str:
        """Update prompt with current statistics vs targets.
        
        Args:
            base_prompt: Original base prompt template
            location: Location name for context
            households: Current list of generated households
            
        Returns:
            Updated prompt with statistics
        """
        # Handle both data formats: direct arrays and {"household": [...]} objects
        people_data = []
        for household_idx, household in enumerate(households):
            if isinstance(household, dict) and 'household' in household:
                # Format: {"household": [person1, person2, ...]}
                people = household['household']
            else:
                # Skip invalid household format
                continue
            
            for person in people:
                if isinstance(person, dict):
                    people_data.append(dict(**person, household_id=household_idx))
        
        synthetic_df = pd.DataFrame(people_data)
        
        # Update prompt with current statistics vs targets
        try:
            result = self.prompt_manager.prepare_prompt_with_feedback(
                base_prompt.replace("{LOCATION}", location),
                synthetic_df=synthetic_df,
                relationship_col="relationship"  # For household composition classifiers
            )
            return result
        except Exception as e:
            logging.error(f"Error in prepare_prompt_with_feedback:")
            logging.error(f"  Error: {str(e)}")
            logging.error(f"  synthetic_df shape: {synthetic_df.shape}")
            logging.error(f"  synthetic_df columns: {list(synthetic_df.columns)}")
            logging.error(f"  synthetic_df dtypes: {dict(synthetic_df.dtypes)}")
            logging.error(f"  Stack trace: {traceback.format_exc()}")
            raise  # Re-raise the error
    
    def _record_token_usage(self, model: BaseLLM, batch_size: int):
        """Record token usage for the batch.
        
        Args:
            model: LLM model instance
            batch_size: Size of the processed batch
        """
        # Get the most recent token usage entries for this batch
        recent_usage = model.token_usage_history[-batch_size:]
        for usage in recent_usage:
            self.token_analyzer.record_usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                prompt_type="batch_generation",
                batch_size=batch_size,
                metadata={"model": model.model_name}
            )