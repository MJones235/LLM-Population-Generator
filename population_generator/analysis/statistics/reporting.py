"""Reporting and summary functionality for fit metrics."""

from typing import Dict, Any
import numpy as np

from .core import StatisticResult


class FitReporter:
    """Handles reporting and summarization of fit metrics."""
    
    def get_overall_fit_summary(self, results: Dict[str, StatisticResult]) -> Dict[str, Any]:
        """Get summary of fit metrics across all statistics.
        
        Args:
            results: Results from compute_all_statistics
            
        Returns:
            Summary dictionary with aggregate fit metrics
        """
        summary = {
            'num_distributions': 0,
            'distributions_with_targets': 0,
            'avg_jensen_shannon': None,
            'avg_total_variation': None,
            'worst_jensen_shannon': None,
            'best_jensen_shannon': None,
            'distributions_summary': []
        }
        
        js_values = []
        tv_values = []
        
        for name, result in results.items():
            summary['num_distributions'] += 1
            
            if result.target and result.fit_metrics:
                summary['distributions_with_targets'] += 1
                
                js = result.fit_metrics.get('jensen_shannon_divergence')
                tv = result.fit_metrics.get('total_variation_distance')
                
                if js is not None:
                    js_values.append(js)
                if tv is not None:
                    tv_values.append(tv)
                
                summary['distributions_summary'].append({
                    'name': name,
                    'jensen_shannon': js,
                    'total_variation': tv
                })
        
        # Compute aggregates
        if js_values:
            summary['avg_jensen_shannon'] = np.mean(js_values)
            summary['worst_jensen_shannon'] = max(js_values)
            summary['best_jensen_shannon'] = min(js_values)
        
        if tv_values:
            summary['avg_total_variation'] = np.mean(tv_values)
        
        return summary
    
    def format_fit_summary(self, summary: Dict[str, Any]) -> str:
        """Format fit summary as readable text."""
        lines = ["=== Distributional Fit Summary ==="]
        
        lines.append(f"Distributions analyzed: {summary['num_distributions']}")
        lines.append(f"With target data: {summary['distributions_with_targets']}")
        
        if summary['avg_jensen_shannon'] is not None:
            lines.append(f"Average JS Divergence: {summary['avg_jensen_shannon']:.4f}")
            lines.append(f"Best JS Divergence: {summary['best_jensen_shannon']:.4f}")
            lines.append(f"Worst JS Divergence: {summary['worst_jensen_shannon']:.4f}")
        
        if summary['avg_total_variation'] is not None:
            lines.append(f"Average TV Distance: {summary['avg_total_variation']:.4f}")
        
        lines.append("\nPer-distribution details:")
        for dist in summary['distributions_summary']:
            js = dist['jensen_shannon']
            js_str = f"{js:.4f}" if js is not None else "N/A"
            lines.append(f"  {dist['name']}: JS = {js_str}")
        
        return "\n".join(lines)