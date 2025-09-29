"""Utilities for token usage analysis and cost estimation.

Note: This module does not include default pricing information. Users must provide
current pricing from their LLM provider. Pricing changes frequently and varies by
provider, region, and service tier.

For Azure OpenAI pricing, see:
https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/

For OpenAI API pricing, see:
https://openai.com/pricing
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CostEstimate:
    """Cost estimation for token usage."""
    input_cost: float
    output_cost: float
    total_cost: float
    currency: str = "GBP"


class TokenAnalyzer:
    """Analyzer for token usage patterns and cost estimation."""
    
    def __init__(self, model_name: str, pricing: Dict[str, float]):
        """Initialize token analyzer.
        
        Args:
            model_name: Name of the model for cost estimation
            pricing: Pricing dict with 'input' and 'output' keys (cost per 1K tokens in USD)
                    Example: {"input": 0.005, "output": 0.015}
        
        Raises:
            ValueError: If pricing dict is missing required keys or has invalid values
        """
        if not isinstance(pricing, dict):
            raise ValueError("Pricing must be a dictionary")
        
        required_keys = {"input", "output"}
        if not required_keys.issubset(pricing.keys()):
            raise ValueError(f"Pricing must contain keys: {required_keys}")
        
        if not all(isinstance(v, (int, float)) and v >= 0 for v in pricing.values()):
            raise ValueError("Pricing values must be non-negative numbers")
        
        self.model_name = model_name
        self.pricing = pricing
        self.session_data: List[Dict[str, Any]] = []
    
    def record_usage(self, 
                    input_tokens: int, 
                    output_tokens: int,
                    prompt_type: str = "generation",
                    batch_size: int = 1,
                    metadata: Optional[Dict[str, Any]] = None):
        """Record token usage for analysis.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            prompt_type: Type of prompt (e.g., 'generation', 'retry', 'batch')
            batch_size: Size of the batch this usage represents
            metadata: Additional metadata for analysis
        """
        record = {
            "timestamp": datetime.now(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "prompt_type": prompt_type,
            "batch_size": batch_size,
            "cost": self.estimate_cost(input_tokens, output_tokens),
            **(metadata or {})
        }
        self.session_data.append(record)
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> CostEstimate:
        """Estimate cost for given token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            CostEstimate object
        """
        input_cost = (input_tokens / 1000) * self.pricing["input"]
        output_cost = (output_tokens / 1000) * self.pricing["output"]
        
        return CostEstimate(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost
        )
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the current session.
        
        Returns:
            Dictionary with session statistics and cost analysis
        """
        if not self.session_data:
            return {"error": "No usage data recorded"}
        
        df = pd.DataFrame(self.session_data)
        
        total_input = df["input_tokens"].sum()
        total_output = df["output_tokens"].sum()
        total_cost = sum(record["cost"].total_cost for record in self.session_data)
        
        return {
            "model": self.model_name,
            "session_duration": (df["timestamp"].max() - df["timestamp"].min()).total_seconds(),
            "total_requests": len(df),
            "total_input_tokens": int(total_input),
            "total_output_tokens": int(total_output),
            "total_tokens": int(total_input + total_output),
            "estimated_cost": {
                "total": round(total_cost, 6),  # Increased precision for small costs
                "input": round(sum(record["cost"].input_cost for record in self.session_data), 6),
                "output": round(sum(record["cost"].output_cost for record in self.session_data), 6),
                "currency": "GBP"
            },
            "averages": {
                "input_tokens_per_request": round(total_input / len(df), 1),
                "output_tokens_per_request": round(total_output / len(df), 1),
                "cost_per_request": round(total_cost / len(df), 6)  # Increased precision
            },
            "by_prompt_type": df.groupby("prompt_type").agg({
                "input_tokens": ["sum", "mean"],
                "output_tokens": ["sum", "mean"],
                "total_tokens": ["sum", "mean"]
            }).round(1).to_dict()
        }
    
    def export_detailed_log(self, filepath: str):
        """Export detailed usage log to CSV.
        
        Args:
            filepath: Path to save the CSV file
        """
        if not self.session_data:
            return
        
        # Flatten cost data for CSV export
        export_data = []
        for record in self.session_data:
            flat_record = {
                **{k: v for k, v in record.items() if k != "cost"},
                "input_cost": record["cost"].input_cost,
                "output_cost": record["cost"].output_cost,
                "total_cost": record["cost"].total_cost
            }
            export_data.append(flat_record)
        
        df = pd.DataFrame(export_data)
        df.to_csv(filepath, index=False)
        print(f"Token usage log exported to {filepath}")
    
    def reset_session(self):
        """Reset session data."""
        self.session_data.clear()


def create_pricing_config(input_cost_per_1k: float, output_cost_per_1k: float) -> Dict[str, float]:
    """Create a pricing configuration dictionary.
    
    Args:
        input_cost_per_1k: Cost per 1,000 input tokens in USD
        output_cost_per_1k: Cost per 1,000 output tokens in USD
    
    Returns:
        Pricing dictionary ready for use with TokenAnalyzer
    
    Example:
        # Azure OpenAI GPT-4o-mini pricing (as of Sept 2025)
        pricing = create_pricing_config(0.00015, 0.0006)
        analyzer = TokenAnalyzer("gpt-4o-mini", pricing)
    """
    return {"input": input_cost_per_1k, "output": output_cost_per_1k}


class TokenTrackingMixin:
    """Mixin to add token tracking to population generators."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_analyzer: Optional[TokenAnalyzer] = None
    
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
            return {"error": "Cost tracking not enabled"}
        return self.token_analyzer.get_session_summary()
    
    def export_cost_log(self, filepath: str):
        """Export detailed cost log."""
        if self.token_analyzer:
            self.token_analyzer.export_detailed_log(filepath)
