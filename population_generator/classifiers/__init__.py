"""Core classification system for demographic analysis.

This module provides the abstract base classes for implementing
custom classifiers. For reference implementations, see:
population_generator.contrib.classifiers
"""

from .base import DemographicClassifier, HouseholdLevelClassifier, IndividualLevelClassifier

__all__ = [
    "DemographicClassifier",
    "HouseholdLevelClassifier", 
    "IndividualLevelClassifier"
]
