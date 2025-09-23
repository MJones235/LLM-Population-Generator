"""Statistics management for prompt placeholders.

This module has been refactored into a package structure. 
Import the required classes from the statistics package.
"""

# Re-export all classes from the new package structure for backward compatibility
from .statistics import (
    StatisticResult,
    DistributionalFitCalculator,
    StatisticProvider,
    ClassifierStatisticProvider,
    CustomStatisticProvider,
    StatisticsManager,
    StatisticFormatter,
    FitReporter
)

# Keep the old imports working
__all__ = [
    'StatisticResult',
    'DistributionalFitCalculator',
    'StatisticProvider',
    'ClassifierStatisticProvider',
    'CustomStatisticProvider',
    'StatisticsManager',
    'StatisticFormatter',
    'FitReporter'
]
