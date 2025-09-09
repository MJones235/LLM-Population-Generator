"""Household size classification system.

This module provides the abstract base class for household size classifiers.
For reference implementations, see population_generator.contrib.classifiers.
"""

from .base import HouseholdSizeClassifier

__all__ = ["HouseholdSizeClassifier"]
