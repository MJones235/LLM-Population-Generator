"""Statistics package for population generation."""

from .core import StatisticResult
from .fit_metrics import DistributionalFitCalculator
from .providers import (
    StatisticProvider, 
    ClassifierStatisticProvider, 
    CustomStatisticProvider
)
from .manager import StatisticsManager
from .formatters import StatisticFormatter
from .reporting import FitReporter

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