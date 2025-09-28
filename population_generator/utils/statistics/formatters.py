"""Text formatting for statistics."""

from typing import Dict, Optional

from .core import StatisticResult


class StatisticFormatter:
    """Handles formatting of statistic results as text."""
    
    def format_statistic_text(self, result: StatisticResult, format_type: str = "comparison", 
                             threshold: Optional[float] = 0.5) -> str:
        """Format a statistic result as text.
        
        Args:
            result: The statistic result to format
            format_type: Format type ("comparison", "observed", "target")
            threshold: Threshold for showing guidance text (None to disable guidance)
            
        Returns:
            Formatted text representation
        """
        if format_type == "observed":
            return self._format_distribution(result.observed, "Current")
        elif format_type == "target" and result.target:
            return self._format_distribution(result.target, "Target")
        elif format_type == "comparison" and result.target:
            return self._format_comparison(result.observed, result.target, threshold)
        else:
            return self._format_distribution(result.observed, "Current")
    
    def _format_distribution(self, distribution: Dict[str, float], label: str) -> str:
        """Format a single distribution."""
        items = [f"{k}: {v:.1f}%" for k, v in distribution.items()]
        return f"{label} Distribution: {', '.join(items)}"
    
    def _format_comparison(self, observed: Dict[str, float], target: Dict[str, float], 
                           threshold: Optional[float] = 0.5) -> str:
        """Format a comparison between observed and target distributions.
        
        Args:
            observed: Observed distribution
            target: Target distribution  
            threshold: Threshold for showing guidance text (None to disable guidance)
        """
        lines = []
        all_keys = set(observed.keys()) | set(target.keys())
        
        for key in sorted(all_keys):
            obs_val = observed.get(key, 0.0)
            tgt_val = target.get(key, 0.0)
            
            # Generate guidance text if threshold is set and difference exceeds threshold
            guidance_text = ""
            if threshold is not None and abs(obs_val - tgt_val) >= threshold:
                underrepresented = obs_val < tgt_val
                guidance_text = " (under-represented)" if underrepresented else " (over-represented)"
            
            lines.append(f"- {key}: current = {obs_val:.1f}%, target = {tgt_val:.1f}%{guidance_text}")
        
        return "\n".join(lines)