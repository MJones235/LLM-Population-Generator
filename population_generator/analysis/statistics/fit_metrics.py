"""Distributional fit metrics calculations."""

from typing import Dict
import numpy as np
import warnings


class DistributionalFitCalculator:
    """Calculator for distributional fit metrics."""
    
    @staticmethod
    def jensen_shannon_divergence(observed: Dict[str, float], 
                                target: Dict[str, float]) -> float:
        """Compute Jensen-Shannon divergence between two distributions.
        
        Args:
            observed: Observed distribution as percentages
            target: Target distribution as percentages
            
        Returns:
            Jensen-Shannon divergence (0 = identical, 1 = maximally different)
        """
        # Get all categories
        all_keys = set(observed.keys()) | set(target.keys())
        
        # Convert to probability arrays
        p = np.array([observed.get(k, 0.0) / 100.0 for k in sorted(all_keys)])
        q = np.array([target.get(k, 0.0) / 100.0 for k in sorted(all_keys)])
        
        # Normalize to ensure they sum to 1
        p = p / p.sum() if p.sum() > 0 else p
        q = q / q.sum() if q.sum() > 0 else q
        
        # Compute JS divergence using scipy if available, otherwise manual calculation
        try:
            from scipy.spatial.distance import jensenshannon
            return jensenshannon(p, q)
        except ImportError:
            # Manual calculation if scipy not available
            # JS(P,Q) = 0.5 * KL(P,M) + 0.5 * KL(Q,M) where M = 0.5*(P+Q)
            m = 0.5 * (p + q)
            
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            p_safe = np.maximum(p, eps)
            q_safe = np.maximum(q, eps)
            m_safe = np.maximum(m, eps)
            
            # KL divergence calculations
            kl_pm = np.sum(p * np.log(p_safe / m_safe))
            kl_qm = np.sum(q * np.log(q_safe / m_safe))
            
            js_div = 0.5 * kl_pm + 0.5 * kl_qm
            return np.sqrt(js_div)  # Return JS distance, not divergence
    
    @staticmethod
    def total_variation_distance(observed: Dict[str, float], 
                               target: Dict[str, float]) -> float:
        """Compute Total Variation distance between distributions.
        
        Returns:
            Total variation distance (0 = identical, 1 = maximally different)
        """
        all_keys = set(observed.keys()) | set(target.keys())
        
        distance = 0.0
        for key in all_keys:
            obs_val = observed.get(key, 0.0) / 100.0
            tgt_val = target.get(key, 0.0) / 100.0
            distance += abs(obs_val - tgt_val)
        
        return distance / 2.0
    
    @staticmethod
    def chi_squared_statistic(observed: Dict[str, float], 
                            target: Dict[str, float],
                            sample_size: int = 1000) -> float:
        """Compute Chi-squared statistic (assumes sample size for observed counts).
        
        Args:
            observed: Observed distribution as percentages
            target: Target distribution as percentages  
            sample_size: Assumed sample size for observed data
            
        Returns:
            Chi-squared statistic
        """
        all_keys = set(observed.keys()) | set(target.keys())
        
        chi_sq = 0.0
        for key in all_keys:
            obs_count = (observed.get(key, 0.0) / 100.0) * sample_size
            exp_count = (target.get(key, 0.0) / 100.0) * sample_size
            
            if exp_count > 0:
                chi_sq += ((obs_count - exp_count) ** 2) / exp_count
        
        return chi_sq
    
    @classmethod
    def compute_all_metrics(cls, observed: Dict[str, float], 
                          target: Dict[str, float],
                          sample_size: int = 1000) -> Dict[str, float]:
        """Compute all available fit metrics.
        
        Args:
            observed: Observed distribution as percentages
            target: Target distribution as percentages
            sample_size: Sample size for observed data
            
        Returns:
            Dictionary of metric name -> value
        """
        metrics = {}
        
        try:
            metrics['jensen_shannon_divergence'] = cls.jensen_shannon_divergence(observed, target)
        except Exception as e:
            warnings.warn(f"Could not compute JS divergence: {e}")
        
        try:
            metrics['total_variation_distance'] = cls.total_variation_distance(observed, target)
        except Exception as e:
            warnings.warn(f"Could not compute TV distance: {e}")
            
        try:
            metrics['chi_squared'] = cls.chi_squared_statistic(observed, target, sample_size)
        except Exception as e:
            warnings.warn(f"Could not compute chi-squared: {e}")
        
        return metrics