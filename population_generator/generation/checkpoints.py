"""Checkpoint management for population generation."""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
from datetime import datetime

from ..llm.base import BaseLLM


class CheckpointManager:
    """Manages checkpoint save/load/resume operations for population generation."""
    
    def __init__(self, checkpoint_dir: Union[str, Path] = "./checkpoints"):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def create_checkpoint_path(self, checkpoint_name: Optional[str] = None, n_run: int = 1) -> Path:
        """Create a checkpoint file path.
        
        Args:
            checkpoint_name: Name for checkpoint files (defaults to timestamp-based name)
            n_run: Run number for tracking
            
        Returns:
            Path to checkpoint file
        """
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"population_gen_{timestamp}_run{n_run}"
        
        return self.checkpoint_dir / f"{checkpoint_name}.json"
    
    def save_checkpoint(self, 
                       checkpoint_path: Path, 
                       households: List[Dict[str, Any]], 
                       batch_num: int, 
                       total_batches: int, 
                       n_households: int,
                       model: BaseLLM,
                       base_prompt: str,
                       schema: Dict[str, Any],
                       location: str) -> None:
        """Save checkpoint data to allow resuming generation after interruption.
        
        Args:
            checkpoint_path: Path to save the checkpoint file
            households: Current list of generated households
            batch_num: Current batch number (1-indexed)
            total_batches: Total number of batches
            n_households: Target number of households
            model: LLM model instance (for metadata)
            base_prompt: Original base prompt
            schema: JSON schema used for validation
            location: Location name for context
        """
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'completed_batches': batch_num,
            'total_batches': total_batches,
            'households_generated': len(households),
            'target_households': n_households,
            'progress_percent': (len(households) / n_households) * 100,
            'households': households,
            'generation_metadata': {
                'model_name': getattr(model, 'model_name', 'unknown'),
                'base_prompt': base_prompt,
                'schema': schema,
                'location': location
            }
        }
        
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            print(f"  ✓ Checkpoint saved: {len(households)}/{n_households} households ({checkpoint_data['progress_percent']:.1f}%)")
        except Exception as e:
            print(f"  ⚠ Warning: Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Load checkpoint data from file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Dictionary containing checkpoint data or None if loading fails
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Checkpoint file not found: {checkpoint_path}")
            return None
            
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
    
    def resume_from_checkpoint(self, checkpoint_path: Path) -> tuple[List[Dict[str, Any]], int]:
        """Resume generation from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Tuple of (households list, start_batch)
        """
        if not checkpoint_path.exists():
            return [], 0
            
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint_data = self.load_checkpoint(checkpoint_path)
        
        if not checkpoint_data:
            return [], 0
            
        households = checkpoint_data.get('households', [])
        start_batch = checkpoint_data.get('completed_batches', 0)
        print(f"Loaded {len(households)} households from checkpoint. Starting from batch {start_batch + 1}")
        
        return households, start_batch
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoint files with summary information.
        
        Returns:
            List of dictionaries containing checkpoint metadata
        """
        if not self.checkpoint_dir.exists():
            return []
            
        checkpoints = []
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                    
                checkpoint_info = {
                    'file': str(checkpoint_file),
                    'name': checkpoint_file.stem,
                    'timestamp': data.get('timestamp', 'unknown'),
                    'progress': f"{data.get('households_generated', 0)}/{data.get('target_households', 0)}",
                    'progress_percent': data.get('progress_percent', 0),
                    'completed_batches': data.get('completed_batches', 0),
                    'total_batches': data.get('total_batches', 0),
                    'model_name': data.get('generation_metadata', {}).get('model_name', 'unknown')
                }
                checkpoints.append(checkpoint_info)
            except Exception as e:
                print(f"Warning: Could not read checkpoint {checkpoint_file}: {e}")
                
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        return checkpoints