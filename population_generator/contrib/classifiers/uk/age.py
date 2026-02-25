"""UK Census age classifier using standard implementation."""

from ..standard.age import StandardAgeClassifier


class UKAgeClassifier(StandardAgeClassifier):
    """Age classifier for UK Census data."""
    
    def __init__(self, threshold=None):
        super().__init__(threshold=threshold)