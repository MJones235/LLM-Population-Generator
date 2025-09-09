"""Main population generator class."""

from typing import Any, Dict, List, Optional
import pandas as pd

from .config import Config
from ..llm.base import BaseLLM
from ..utils.prompts import PromptManager



class PopulationGenerator:
    """Main class for generating synthetic population data using LLMs."""
    
    def __init__(self, 
                 data_path: Optional[str] = None,
                 prompts_path: Optional[str] = None,
                 config_path: Optional[str] = None):
        """Initialize the population generator.
        
        Args:
            data_path: Base path for data files (census, etc.)
            prompts_path: Path to prompt templates
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        
        if data_path:
            self.config.set_data_paths(data_path)
            
        if prompts_path:
            self.config._config["data"]["prompts_dir"] = prompts_path
            
        self.prompt_manager = PromptManager(self.config)
    
    @property
    def data_loader(self):
        """Access to data loading functionality through prompt manager."""
        return self.prompt_manager
    
    def generate_households(
        self,
        n_households: int,
        model: BaseLLM,
        base_prompt: str,
        schema: str,
        location: str,
        batch_size: Optional[int] = None,
        n_run: int = 1,
        target_data_files: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Generate synthetic households using LLM with automatic statistics feedback.
        
        Args:
            n_households: Number of households to generate
            model: LLM instance to use for generation
            base_prompt: Base prompt template (can contain statistics placeholders)
            schema: JSON schema for validation
            location: Location name for context
            batch_size: Number of households per batch
            n_run: Run number for tracking
            target_data_files: List of target data files to load for comparison
            
        Returns:
            List of generated household dictionaries
            
        Note:
            Statistics placeholders in base_prompt (e.g., {HOUSEHOLD_SIZE_STATS}) 
            will be automatically replaced with current vs target distributions 
            after each batch if statistics providers are registered.
        """
        # Set defaults
        if batch_size is None:
            batch_size = self.config.get("generation.default_batch_size", 10)
        
        # Note: Target data files should be registered with classifiers using prompt_manager.register_classifier()
        # This parameter is kept for backwards compatibility but users should register
        # target data when adding classifiers for better control
        
        households = []
        
        # Prepare initial prompt (no statistics available yet)
        prompt = self.prompt_manager.statistics_manager.replace_placeholders_in_prompt(
            base_prompt.replace("{LOCATION}", location),
            synthetic_df=None,
            format_type="comparison"
        )

        for i in range(0, n_households, batch_size):
            batch_count = min(batch_size, n_households - i)
            is_last_batch = (i + batch_count) >= n_households

            print(f"\n--- Batch {i // batch_size + 1} ({batch_count} households), Run {n_run} ---")

            batch_prompts = self._prepare_batch_prompts(
                prompt,
                batch_count
            )

            print(f"Prompt (first): {batch_prompts[0][:200]}...")

            batch_results = self._run_batch(model, batch_prompts, schema)
            households.extend(batch_results)

            # Update prompt with statistics for next batch
            if not is_last_batch:
                synthetic_df = pd.DataFrame(
                    [dict(**person, household_id=household_idx) 
                     for household_idx, household in enumerate(households) 
                     for person in household]
                )

                # Update prompt with current statistics vs targets
                prompt = self.prompt_manager.prepare_prompt_with_feedback(
                    base_prompt.replace("{LOCATION}", location),
                    synthetic_df=synthetic_df,
                    relationship_col="relationship"  # For household composition classifiers
                )

        return households
    

    
    def _prepare_batch_prompts(self, 
                              prompt_template: str, 
                              batch_count: int) -> List[str]:
        """Prepare prompts for a batch of households."""
        batch_prompts = []
        for i in range(batch_count):
            batch_prompts.append(prompt_template)
        return batch_prompts
    
    def _run_batch(self, model: BaseLLM, prompts: List[str], schema: str) -> List[Dict[str, Any]]:
        """Run a batch of prompts through the LLM."""
        try:
            timeout = self.config.get("generation.default_timeout", 60)
            return model.generate_batch_json(prompts, schema, max_parallel=1, timeout=timeout)
        except Exception as e:
            print(f"[ERROR] Batch generation failed: {e}")
            return []
