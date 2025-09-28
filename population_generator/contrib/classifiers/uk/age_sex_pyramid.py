"""UK Census age/sex pyramid classifier using standard implementation."""

from ..standard.age_sex_pyramid import StandardAgeSexPyramidClassifier


class UKAgeSexPyramidClassifier(StandardAgeSexPyramidClassifier):
    """Age/sex pyramid classifier for UK Census data."""
    
    def __init__(self):
        """Initialize with UK Census region identifier."""
        super().__init__('uk_census')