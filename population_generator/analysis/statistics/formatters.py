"""Text formatting for statistics."""

import logging
import traceback
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
            threshold: Default threshold for showing guidance text (can be overridden by classifier)
            
        Returns:
            Formatted text representation
        """
        # Use classifier-specific threshold if available, otherwise use provided threshold
        effective_threshold = threshold
        if result.metadata and 'threshold' in result.metadata:
            effective_threshold = result.metadata['threshold']
        
        if format_type == "observed":
            return self._format_distribution(result.observed, "Current", result.label_order, result)
        elif format_type == "target" and result.target:
            return self._format_distribution(result.target, "Target", result.label_order, result)
        elif format_type == "comparison" and result.target:
            return self._format_comparison(result.observed, result.target, effective_threshold, result.label_order, result)
        else:
            return self._format_distribution(result.observed, "Current", result.label_order, result)
    
    def _format_distribution(self, distribution: Dict[str, float], label: str, 
                           label_order: Optional[List[str]] = None,
                           result: Optional[StatisticResult] = None) -> str:
        """Format a single distribution.
        
        Args:
            distribution: Dictionary of category -> percentage
            label: Label for the distribution (e.g., "Target", "Current")
            label_order: Optional ordered list of labels for consistent ordering
            result: Optional StatisticResult for metadata access
        """
        try:
            if label_order:
                # Use provided label order, but only include labels with non-zero values
                # or labels that actually exist in the distribution
                ordered_keys = []
                for k in label_order:
                    if k in distribution and distribution[k] > 0:
                        ordered_keys.append(k)
                # Add any missing keys not in label_order (shouldn't happen but defensive)
                missing_keys = [k for k in distribution.keys() if k not in ordered_keys and distribution[k] > 0]
                ordered_keys.extend(sorted(missing_keys))
            else:
                # Fall back to alphabetical sorting, only including non-zero values
                ordered_keys = sorted([k for k in distribution.keys() if distribution[k] > 0])
        except TypeError as e:
            logging.error(f"TypeError in _format_distribution when processing distribution:")
            logging.error(f"  distribution: {distribution}")
            logging.error(f"  label_order: {label_order}")
            logging.error(f"  result.metadata: {result.metadata if result else 'No result'}")
            logging.error(f"  Stack trace: {traceback.format_exc()}")
            raise  # Re-raise the error so we get the full stack trace
        
        # Check data type from metadata
        is_value_data = False
        if result and result.metadata and 'data_type' in result.metadata:
            is_value_data = result.metadata['data_type'] == 'value'
        
        # Format items based on data type
        display_items = []
        for k in ordered_keys:
            value = distribution[k]  # We know this exists since we filtered above
            if is_value_data:
                # Format as values (no % sign)
                display_items.append(f"{k}: {value:.1f}")
            else:
                # Format as percentages
                display_items.append(f"{k}: {value:.1f}%")
            
        return f"{label}:\n{'\n'.join(display_items)}"
    
    def _format_comparison(self, observed: Dict[str, float], target: Dict[str, float], 
                           threshold: Optional[float] = 0.5, 
                           label_order: Optional[List[str]] = None,
                           result: Optional[StatisticResult] = None) -> str:
        """Format a comparison between observed and target distributions.
        
        Args:
            observed: Observed distribution
            target: Target distribution  
            threshold: Threshold for showing guidance text (None to disable guidance)
            label_order: Optional ordered list of labels for consistent ordering
            result: Optional StatisticResult for metadata access
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
        
        # Check data type from metadata
        is_value_data = False
        if result and result.metadata and 'data_type' in result.metadata:
            is_value_data = result.metadata['data_type'] == 'value'
        
        for key in ordered_keys:
            obs_val = observed.get(key, 0.0)
            tgt_val = target.get(key, 0.0)
            
            # Generate guidance text if threshold is set and difference exceeds threshold
            guidance_text = ""
            try:
                if threshold is not None and abs(obs_val - tgt_val) >= threshold:
                    if is_value_data:
                        # For metrics, use "too high/too low"
                        guidance_text = " (too low)" if obs_val < tgt_val else " (too high)"
                    else:
                        # For distributions, use "under/over-represented"
                        guidance_text = " (under-represented)" if obs_val < tgt_val else " (over-represented)"
            except TypeError as e:
                logging.error(f"TypeError in _format_comparison for key '{key}':")
                logging.error(f"  obs_val: {obs_val} (type: {type(obs_val)})")
                logging.error(f"  tgt_val: {tgt_val} (type: {type(tgt_val)})")
                logging.error(f"  threshold: {threshold} (type: {type(threshold)})")
                logging.error(f"  observed dict: {observed}")
                logging.error(f"  target dict: {target}")
                logging.error(f"  result.metadata: {result.metadata if result else 'No result'}")
                logging.error(f"  Stack trace: {traceback.format_exc()}")
                raise  # Re-raise the error so we get the full stack trace
            
            if is_value_data:
                # Format as values (no % sign)
                lines.append(f"- {key}: current = {obs_val:.1f}, target = {tgt_val:.1f}{guidance_text}")
            else:
                # Format as percentages
                lines.append(f"- {key}: current = {obs_val:.1f}%, target = {tgt_val:.1f}%{guidance_text}")
        
        return "\n".join(lines)