"""Main population generator class."""

import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .config import Config
from ..generation.checkpoints import CheckpointManager
from ..generation.batching import BatchProcessor
from ..generation.sessions import GenerationSession
from ..analysis.cost_tracker import CostTracker
from ..llm.base import BaseLLM
from ..generation.prompts import PromptManager
from ..data.validation import CustomValidator



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
        
        # Initialize components
        self.cost_tracker = CostTracker()
        self.checkpoint_manager = CheckpointManager()
        self.generation_session = GenerationSession()
        # Initialize batch processor without token_analyzer - will be passed during generation
        self.batch_processor = BatchProcessor(
            self.config, 
            self.prompt_manager, 
            None  # Will be updated when cost tracking is enabled
        )
    
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
        self.cost_tracker.enable_cost_tracking(model_name, pricing)
        # Update batch processor with the enabled token analyzer
        self.batch_processor.token_analyzer = self.cost_tracker.token_analyzer
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost analysis summary."""
        return self.cost_tracker.get_cost_summary()
    
    def export_cost_log(self, filepath: str):
        """Export detailed cost log."""
        self.cost_tracker.export_log(filepath)
    
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
        enable_progressive_saving: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_name: Optional[str] = None,
        resume_from_checkpoint: bool = False,
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
            enable_progressive_saving: Enable saving after each batch to prevent data loss
            output_dir: Output directory for final results (when provided, checkpoints default to output_dir/checkpoints)
            checkpoint_dir: Directory to save checkpoints (defaults to output_dir/checkpoints if output_dir provided, else './checkpoints')
            checkpoint_name: Name for checkpoint files (defaults to timestamp-based name)
            resume_from_checkpoint: Whether to resume from an existing checkpoint
            
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
        
        # Auto-configure checkpoint directory if progressive saving is enabled
        if enable_progressive_saving and checkpoint_dir is None:
            if output_dir is not None:
                # Default to output_dir/checkpoints when output_dir is provided
                checkpoint_dir = Path(output_dir) / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
            else:
                # Fall back to ./checkpoints if no output_dir provided
                checkpoint_dir = "./checkpoints"
        
        # Initialize session
        session = GenerationSession(
            n_households=n_households,
            batch_size=batch_size,
            n_run=n_run,
            enable_progressive_saving=enable_progressive_saving,
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
            checkpoint_name=checkpoint_name
        )
        
        # Set up checkpoint manager if progressive saving is enabled
        checkpoint_path = None
        if enable_progressive_saving:
            if checkpoint_dir:
                self.checkpoint_manager = CheckpointManager(checkpoint_dir)
            checkpoint_path = self.checkpoint_manager.create_checkpoint_path(checkpoint_name, n_run)
            print(f"Progressive saving enabled. Checkpoint file: {checkpoint_path}")
        
        households = []
        start_batch = 0
        
        # Resume from checkpoint if requested
        if resume_from_checkpoint and checkpoint_path:
            households, start_batch = self.checkpoint_manager.resume_from_checkpoint(checkpoint_path)
            
            # Check if generation is already complete
            if len(households) >= n_households:
                print(f"Generation already completed ({len(households)} >= {n_households} households). Returning existing data.")
                return households[:n_households]
        
        # Prepare initial prompt (no statistics available yet)
        prompt = self.prompt_manager.statistics_manager.replace_placeholders_in_prompt(
            base_prompt.replace("{LOCATION}", location),
            synthetic_df=None,
            format_type="comparison"
        )

        # Generate households in batches
        for i in range(start_batch * batch_size, n_households, batch_size):
            batch_info = session.get_batch_info(i)
            
            print(f"\n--- Batch {batch_info['batch_num']}/{batch_info['total_batches']} ({batch_info['batch_count']} households), Run {n_run} ---")

            # Prepare and run batch
            batch_prompts = self.batch_processor.prepare_batch_prompts(prompt, batch_info['batch_count'])
            print(f"Prompt: {batch_prompts[0]}...")

            batch_results = self.batch_processor.run_batch(
                model, batch_prompts, schema, custom_validator, 
                households_generated=len(households), 
                total_households=n_households
            )
            households.extend(batch_results)
            
            # Save checkpoint if enabled
            if checkpoint_path and session.should_save_checkpoint():
                self.checkpoint_manager.save_checkpoint(
                    checkpoint_path, households, batch_info['batch_num'], 
                    batch_info['total_batches'], n_households, model, base_prompt, schema, location
                )

            # Update prompt with statistics for next batch
            if not batch_info['is_last_batch']:
                prompt = self.batch_processor.update_prompt_with_statistics(
                    base_prompt, location, households
                )

        return households
    

    
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
        from ..data.export import save_generation_results
        
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
            cost_tracker=self.cost_tracker,  # Pass the actual tracker object
            failure_tracker=llm_model.failure_tracker if llm_model and hasattr(llm_model, 'failure_tracker') else None
        )

    def list_checkpoints(self, checkpoint_dir: Union[str, Path] = "./checkpoints") -> List[Dict[str, Any]]:
        """List available checkpoint files with summary information.
        
        Args:
            checkpoint_dir: Directory to search for checkpoint files
            
        Returns:
            List of dictionaries containing checkpoint metadata
        """
        checkpoint_manager = CheckpointManager(checkpoint_dir)
        return checkpoint_manager.list_checkpoints()
    
    def load_checkpoint_data(self, checkpoint_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Load checkpoint data from file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Dictionary containing checkpoint data or None if loading fails
        """
        return self.checkpoint_manager.load_checkpoint(checkpoint_path)
