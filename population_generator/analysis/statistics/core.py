"""Core data structures for statistics management."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class StatisticResult:
    """Container for a computed statistic."""
    name: str
    observed: Dict[str, float]
    target: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    fit_metrics: Optional[Dict[str, float]] = None
    label_order: Optional[List[str]] = None