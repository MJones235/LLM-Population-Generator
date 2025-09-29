"""Cost tracking functionality for population generation."""

from typing import Any, Dict, Optional
from .costs import TokenAnalyzer


class CostTracker:
    """Manages cost tracking and analysis for LLM population generation."""
    
    def __init__(self):
        """Initialize the cost tracker."""
        self.token_analyzer: Optional[TokenAnalyzer] = None
    
    def enable_cost_tracking(self, 
                           model_name: str,
                           pricing: Dict[str, float]) -> None:
        """Enable comprehensive cost tracking.
        
        Args:
            model_name: Name of the model for pricing
            pricing: Pricing dict with 'input' and 'output' keys (cost per 1K tokens in USD)
                    Example: {"input": 0.005, "output": 0.015}
        """
        self.token_analyzer = TokenAnalyzer(model_name, pricing)
    
    def is_enabled(self) -> bool:
        """Check if cost tracking is enabled."""
        return self.token_analyzer is not None
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost analysis summary."""
        if not self.token_analyzer:
            return {"error": "Cost tracking not enabled. Call enable_cost_tracking() first."}
        return self.token_analyzer.get_session_summary()
    
    def export_cost_log(self, filepath: str) -> None:
        """Export detailed cost log."""
        if self.token_analyzer:
            self.token_analyzer.export_detailed_log(filepath)
        else:
            print("Cost tracking not enabled. Call enable_cost_tracking() first.")
    
    def record_batch_usage(self, model, prompts: list) -> None:
        """Record token usage for a batch of prompts."""
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
    
    def get_cost_metadata(self) -> Dict[str, Any]:
        """Get cost metadata for saving to files."""
        metadata = {}
        if self.token_analyzer:
            session_summary = self.get_cost_summary()
            if not session_summary.get("error") and session_summary.get("estimated_cost"):
                estimated_cost = session_summary["estimated_cost"]
                metadata.update({
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
        return metadata