"""Core classification system for demographic analysis.

This module provides the abstract base classes for implementing
custom classifiers. For reference implementations, see:
population_generator.contrib.classifiers
"""

from .household_size import HouseholdSizeClassifier
from .household_type import HouseholdCompositionClassifier

__all__ = [
    "HouseholdSizeClassifier", 
    "HouseholdCompositionClassifier"
]
