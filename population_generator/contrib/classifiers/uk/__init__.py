"""UK demographic classifiers based on UK Census data.

These classifiers implement household size, composition, age/sex pyramid,
age, and sex classification logic based on UK Census demographic patterns.
"""

from .household_size import UKHouseholdSizeClassifier
from .household_composition import UKHouseholdCompositionClassifier
from .age_sex_pyramid import UKAgeSexPyramidClassifier
from .age import UKAgeClassifier
from .sex import UKSexClassifier

__all__ = [
    "UKHouseholdSizeClassifier",
    "UKHouseholdCompositionClassifier",
    "UKAgeSexPyramidClassifier",
    "UKAgeClassifier",
    "UKSexClassifier"
]
