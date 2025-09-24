"""UK Census data preprocessors.

These preprocessors convert UK Census data from the standard ONS format
to the standardized CSV format expected by the core library.

The preprocessors ensure consistency with the corresponding UK classifiers
by using the same category mappings and validation logic.
"""

from .census_preprocessor import UKCensusPreprocessor
from .age import UKAgePreprocessor
from .sex import UKSexPreprocessor
from .household_size import UKHouseholdSizePreprocessor
from .household_composition import UKHouseholdCompositionPreprocessor

__all__ = [
    "UKCensusPreprocessor",
    "UKAgePreprocessor", 
    "UKSexPreprocessor",
    "UKHouseholdSizePreprocessor",
    "UKHouseholdCompositionPreprocessor"
]