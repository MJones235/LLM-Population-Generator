"""Main population generator class."""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd

from .config import Config
from ..llm.base import BaseLLM
from ..utils.prompts import PromptManager
from ..utils.token_analysis import TokenAnalyzer
from ..utils.data_export import PopulationDataSaver
from ..utils.validation import CustomValidator



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
        self.token_analyzer: Optional[TokenAnalyzer] = None
    
    @property
    def data_loader(self):
        """Access to data loading functionality through prompt manager."""
        return self.prompt_manager
    
    def enable_cost_tracking(self, 
                           model_name: str,
                           pricing: Dict[str, float]):
        """Enable comprehensive cost tracking.
        
        Args:
            model_name: Name of the model for pricing
            pricing: Pricing dict with 'input' and 'output' keys (cost per 1K tokens in USD)
                    Example: {"input": 0.005, "output": 0.015}
        """
        self.token_analyzer = TokenAnalyzer(model_name, pricing)
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost analysis summary."""
        if not self.token_analyzer:
            return {"error": "Cost tracking not enabled. Call enable_cost_tracking() first."}
        return self.token_analyzer.get_session_summary()
    
    def export_cost_log(self, filepath: str):
        """Export detailed cost log."""
        if self.token_analyzer:
            self.token_analyzer.export_detailed_log(filepath)
        else:
            print("Cost tracking not enabled. Call enable_cost_tracking() first.")
    
    def generate_households(
        self,
        n_households: int,
        model: BaseLLM,
        base_prompt: str,
        schema: Dict[str, Any],
        location: str,
        custom_validator: Optional[CustomValidator] = None,
        batch_size: Optional[int] = None,
        n_run: int = 1,
    ) -> List[Dict[str, Any]]:
        """Generate synthetic households using LLM with automatic statistics feedback.
        
        Args:
            n_households: Number of households to generate
            model: LLM instance to use for generation
            base_prompt: Base prompt template (can contain statistics placeholders)
            schema: JSON schema for validation
            location: Location name for context
            custom_validator: Optional custom validator for additional rule checking
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
            batch_num = i // batch_size + 1
            total_batches = (n_households + batch_size - 1) // batch_size

            print(f"\n--- Batch {batch_num}/{total_batches} ({batch_count} households), Run {n_run} ---")

            batch_prompts = self._prepare_batch_prompts(
                prompt,
                batch_count
            )

            print(f"Prompt: {batch_prompts[0]}...")

            batch_results = self._run_batch(model, batch_prompts, schema, custom_validator, 
                                          households_generated=len(households), 
                                          total_households=n_households)
            households.extend(batch_results)

            # Update prompt with statistics for next batch
            if not is_last_batch:
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
    
    def _run_batch(self, model: BaseLLM, prompts: List[str], schema: Dict[str, Any], 
                   custom_validator: Optional[CustomValidator] = None,
                   households_generated: int = 0, total_households: int = 0) -> List[Dict[str, Any]]:
        """Run a batch of prompts through the LLM."""
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
                # Get the most recent token usage entries for this batch
                recent_usage = model.token_usage_history[-len(prompts):]
                for usage in recent_usage:
                    self.token_analyzer.record_usage(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        prompt_type="batch_generation",
                        batch_size=len(prompts),
                        metadata={"model": model.model_name}
                    )
            
            return results
        except Exception as e:
            print(f"[ERROR] Batch generation failed: {e}")
            return []
    
    def save_population_data(
        self,
        households: List[Dict[str, Any]],
        model_info: Dict[str, Any],
        generation_parameters: Dict[str, Any],
        output_dir: Union[str, Path],
        output_name: str,
        llm_model: Optional[Any] = None
    ) -> Dict[str, str]:
        """Save population data with comprehensive metadata.
        
        This is a convenience method that creates a PopulationDataSaver and saves
        the generated population data with detailed metadata for future analysis.
        Population data is saved in CSV format and metadata in JSON format.
        
        Args:
            households: Generated household data
            model_info: Information about the LLM used (name, version, etc.)
            generation_parameters: Parameters used for generation (n_households, batch_size, etc.)
            output_dir: Directory to save files
            output_name: Base name for output files (without extension)
            llm_model: LLM model instance for extracting failure statistics
            
        Returns:
            Dictionary with paths to saved files (population_csv and metadata_json)
        """
        from ..utils.data_export import save_generation_results
        
        # Extract cost and failure information from token analyzer and LLM
        kwargs = {}
        if self.token_analyzer:
            session_summary = self.get_cost_summary()
            if not session_summary.get("error") and session_summary.get("estimated_cost"):
                estimated_cost = session_summary["estimated_cost"]
                kwargs.update({
                    'total_cost': estimated_cost.get("total", 0),
                    'cost_breakdown': {
                        'input': estimated_cost.get("input", 0),
                        'output': estimated_cost.get("output", 0)
                    },
                    'token_usage': {
                        'input_tokens': session_summary.get("total_input_tokens", 0),
                        'output_tokens': session_summary.get("total_output_tokens", 0),
                        'total_tokens': session_summary.get("total_tokens", 0)
                    }
                })
        
        if llm_model and hasattr(llm_model, 'failure_tracker'):
            failure_summary = llm_model.failure_tracker.get_academic_summary()
            kwargs.update({
                'total_requests': failure_summary.get("total_attempts", 0),
                'failed_requests': failure_summary.get("total_failures", 0)
            })
        
        # Extract registered classifiers for statistical analysis
        classifiers = []
        statistics_manager = None
        if hasattr(self.prompt_manager, 'statistics_manager'):
            statistics_manager = self.prompt_manager.statistics_manager
            for provider in self.prompt_manager.statistics_manager.providers.values():
                if hasattr(provider, 'classifier'):
                    classifiers.append(provider.classifier)
        
        return save_generation_results(
            households=households,
            model_info=model_info,
            generation_parameters=generation_parameters,
            output_dir=output_dir,
            output_name=output_name,
            classifiers=classifiers if classifiers else None,
            statistics_manager=statistics_manager,
            **kwargs
        )
