"""Text formatting for statistics."""

from typing import Dict, Optional, List

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
            return self._format_distribution(result.observed, "Current", result.label_order)
        elif format_type == "target" and result.target:
            return self._format_distribution(result.target, "Target", result.label_order)
        elif format_type == "comparison" and result.target:
            return self._format_comparison(result.observed, result.target, threshold, result.label_order)
        else:
            return self._format_distribution(result.observed, "Current", result.label_order)
    
    def _format_distribution(self, distribution: Dict[str, float], label: str, 
                           label_order: Optional[List[str]] = None) -> str:
        """Format a single distribution.
        
        Args:
            distribution: Dictionary of category -> percentage
            label: Label for the distribution (e.g., "Target", "Current")
            label_order: Optional ordered list of labels for consistent ordering
        """
        if label_order:
            # Use provided label order, only including categories that exist in distribution
            ordered_keys = [k for k in label_order if k in distribution]
            # Add any missing keys not in label_order (shouldn't happen but defensive)
            missing_keys = [k for k in distribution.keys() if k not in ordered_keys]
            ordered_keys.extend(sorted(missing_keys))
        else:
            # Fall back to alphabetical sorting
            ordered_keys = sorted(distribution.keys())
            
        items = [f"{k}: {distribution[k]:.1f}%" for k in ordered_keys]
        return f"{label} Distribution: {', '.join(items)}"
    
    def _format_comparison(self, observed: Dict[str, float], target: Dict[str, float], 
                           threshold: Optional[float] = 0.5, 
                           label_order: Optional[List[str]] = None) -> str:
        """Format a comparison between observed and target distributions.
        
        Args:
            observed: Observed distribution
            target: Target distribution  
            threshold: Threshold for showing guidance text (None to disable guidance)
            label_order: Optional ordered list of labels for consistent ordering
        """
        lines = []
        all_keys = set(observed.keys()) | set(target.keys())
        
        if label_order:
            # Use provided label order, only including categories that exist
            ordered_keys = [k for k in label_order if k in all_keys]
            # Add any missing keys not in label_order (shouldn't happen but defensive)
            missing_keys = [k for k in all_keys if k not in ordered_keys]
            ordered_keys.extend(sorted(missing_keys))
        else:
            # Fall back to alphabetical sorting
            ordered_keys = sorted(all_keys)
        
        for key in ordered_keys:
            obs_val = observed.get(key, 0.0)
            tgt_val = target.get(key, 0.0)
            
            # Generate guidance text if threshold is set and difference exceeds threshold
            guidance_text = ""
            if threshold is not None and abs(obs_val - tgt_val) >= threshold:
                underrepresented = obs_val < tgt_val
                guidance_text = " (under-represented)" if underrepresented else " (over-represented)"
            
            lines.append(f"- {key}: current = {obs_val:.1f}%, target = {tgt_val:.1f}%{guidance_text}")
        
        return "\n".join(lines)