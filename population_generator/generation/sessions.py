"""Session management for population generation."""

from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path

from ..analysis.costs import TokenAnalyzer


class GenerationSession:
    """Manages state and metadata for a population generation session."""
    
    def __init__(self, 
                 n_households: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 n_run: int = 1,
                 enable_progressive_saving: bool = False,
                 checkpoint_dir: Optional[Path] = None,
                 checkpoint_name: Optional[str] = None):
        """Initialize a generation session.
        
        Args:
            n_households: Target number of households to generate
            batch_size: Number of households per batch
            n_run: Run number for tracking
            enable_progressive_saving: Whether to enable progressive saving
            checkpoint_dir: Directory for checkpoint files
            checkpoint_name: Base name for checkpoint files
        """
        self.n_households = n_households
        self.batch_size = batch_size
        self.n_run = n_run
        self.enable_progressive_saving = enable_progressive_saving
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.households_generated = 0
        self.current_batch = 0
        self.start_time = datetime.now()
        self.metadata = {}
        
    @property
    def total_batches(self) -> int:
        """Calculate total number of batches needed."""
        if not self.n_households or not self.batch_size:
            return 0
        import math
        return math.ceil(self.n_households / self.batch_size)
        
    @property
    def is_complete(self) -> bool:
        """Check if generation is complete."""
        return self.households_generated >= self.n_households
    
    @property
    def progress_percent(self) -> float:
        """Get current progress as percentage."""
        if self.n_households == 0:
            return 100.0
        return (self.households_generated / self.n_households) * 100
    
    def get_progress_percent(self) -> float:
        """Calculate progress percentage (method version)."""
        return self.progress_percent
    
    def get_batch_info(self, batch_start_idx: int) -> Dict[str, Any]:
        """Get information about the current batch.
        
        Args:
            batch_start_idx: Starting index for the current batch
            
        Returns:
            Dictionary with batch information
        """
        batch_count = min(self.batch_size, self.n_households - batch_start_idx)
        batch_num = batch_start_idx // self.batch_size + 1
        is_last_batch = (batch_start_idx + batch_count) >= self.n_households
        
        return {
            'batch_count': batch_count,
            'batch_num': batch_num,
            'total_batches': self.total_batches,
            'is_last_batch': is_last_batch,
            'start_idx': batch_start_idx,
            'end_idx': batch_start_idx + batch_count
        }
    
    def update_progress(self, households_added: int):
        """Update session progress.
        
        Args:
            households_added: Number of households added in latest batch
        """
        self.households_generated += households_added
        self.current_batch += 1
    
    def should_save_checkpoint(self) -> bool:
        """Check if checkpoint should be saved."""
        return self.enable_progressive_saving and not self.is_complete


class CostTracker:
    """Wrapper for TokenAnalyzer to provide cleaner interface."""
    
    def __init__(self):
        """Initialize cost tracker."""
        self.token_analyzer: Optional[TokenAnalyzer] = None
    
    def enable(self, model_name: str, pricing: Dict[str, float]):
        """Enable cost tracking.
        
        Args:
            model_name: Name of the model for pricing
            pricing: Pricing dict with 'input' and 'output' keys (cost per 1K tokens in USD)
        """
        self.token_analyzer = TokenAnalyzer(model_name, pricing)
    
    def is_enabled(self) -> bool:
        """Check if cost tracking is enabled."""
        return self.token_analyzer is not None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost analysis summary."""
        if not self.token_analyzer:
            return {"error": "Cost tracking not enabled."}
        return self.token_analyzer.get_session_summary()
    
    def export_log(self, filepath: str):
        """Export detailed cost log."""
        if self.token_analyzer:
            self.token_analyzer.export_detailed_log(filepath)
        else:
            print("Cost tracking not enabled.")
    
    def get_cost_metadata(self) -> Dict[str, Any]:
        """Get cost metadata for export."""
        if not self.token_analyzer:
            return {}
            
        session_summary = self.get_summary()
        if session_summary.get("error") or not session_summary.get("estimated_cost"):
            return {}
            
        estimated_cost = session_summary["estimated_cost"]
        return {
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
        }