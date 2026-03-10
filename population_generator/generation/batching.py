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
                             batch_count: int,
                             multi_household_prompt: bool = False) -> List[str]:
        """Prepare prompts for a batch of households.

        Args:
            prompt_template: Template prompt to use
            batch_count: Number of households in this batch
            multi_household_prompt: If True, return a single prompt asking the LLM
                to generate all batch_count households at once (the batch_size
                parameter then controls how many households are produced per LLM
                call rather than how many calls are made before statistics are
                recalculated). If False (default), return one prompt per household.

        Returns:
            List of prompts for the batch
        """
        if not multi_household_prompt:
            # Current behaviour: one prompt per household
            return [prompt_template] * batch_count
        else:
            # Multi-household mode: append a count instruction and return a
            # single prompt that asks for all batch_count households at once.
            instruction = (
                f"\n\nGenerate exactly {batch_count} household(s). "
                f"Return a JSON array containing {batch_count} household object(s)."
            )
            return [prompt_template + instruction]
    
    def run_batch(self,
                  model: BaseLLM,
                  prompts: List[str],
                  schema: Dict[str, Any],
                  custom_validator: Optional[CustomValidator] = None,
                  households_generated: int = 0,
                  total_households: int = 0,
                  multi_household_prompt: bool = False,
                  batch_count: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run a batch of prompts through the LLM.

        Args:
            model: LLM instance to use
            prompts: List of prompts to process
            schema: JSON schema for validation
            custom_validator: Optional custom validator
            households_generated: Number of households already generated
            total_households: Total target number of households
            multi_household_prompt: If True, a single prompt is expected that
                asks the LLM for multiple households at once. The response is
                unpacked as a JSON array. If False (default), each prompt
                produces one household.
            batch_count: Number of households expected from a multi-household
                prompt (required when multi_household_prompt=True; ignored
                otherwise).

        Returns:
            List of generated household dictionaries
        """
        try:
            timeout = self.config.get("generation.default_timeout", 60)

            # Enable token tracking on the model if we have a token analyzer
            if self.token_analyzer:
                model.enable_token_tracking(True)

            # Snapshot history length before the batch so we can capture ALL
            # LLM calls made during this batch, including retry attempts.
            token_history_start = len(model.token_usage_history)

            results = []

            if not multi_household_prompt:
                # Current behaviour: one LLM call per household
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
            else:
                # Multi-household mode: one LLM call returns multiple households.
                # prompts contains exactly one entry produced by prepare_batch_prompts.
                n = batch_count if batch_count is not None else len(prompts)
                prompt = prompts[0]
                start_num = households_generated + 1
                end_num = households_generated + n
                print(f"  Generating households {start_num}–{end_num}/{total_households}...", end=" ", flush=True)

                array_schema = {
                    "type": "array",
                    "items": schema,
                    "minItems": n,
                    "maxItems": n
                }

                try:
                    # Do NOT pass custom_validator here: in multi-household mode
                    # the parsed response is a list, but custom validators expect
                    # individual {"household": [...]} dicts.  Per-household
                    # validation is applied below after unpacking the array.
                    result = model.generate_json(prompt, array_schema, None, timeout=timeout)
                    if isinstance(result, list):
                        # Validate each household individually
                        validated = []
                        for household in result:
                            if custom_validator:
                                errors = custom_validator.validate(household)
                                if errors:
                                    rule_names = ", ".join(e.rule_name for e in errors)
                                    print(f"\n    ⚠ Household failed custom validation ({rule_names}) — skipped", end=" ")
                                    validated.append({})
                                else:
                                    validated.append(household)
                            else:
                                validated.append(household)
                        results.extend(validated)
                        print("✓")
                    else:
                        print(f"✗ (Expected list, got {type(result).__name__})")
                        results.extend([{}] * n)
                except Exception as e:
                    print(f"✗ (Failed: {str(e)})")
                    results.extend([{}] * n)

            # Record ALL token usage entries added during this batch (including retries).
            if self.token_analyzer and model.track_tokens:
                self._record_token_usage(model, token_history_start)

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
    
    def _record_token_usage(self, model: BaseLLM, history_start: int):
        """Record token usage for ALL LLM calls made since history_start.

        Using a start index rather than a trailing slice ensures that retry
        attempts are included, not just one entry per prompt.

        Args:
            model: LLM model instance
            history_start: Index into model.token_usage_history at the start of
                           the batch; all entries from this index onward are recorded.
        """
        new_usage = model.token_usage_history[history_start:]
        for usage in new_usage:
            self.token_analyzer.record_usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                prompt_type="batch_generation",
                batch_size=len(new_usage),
                metadata={"model": model.model_name}
            )