"""UK Census sex/gender classifier using standard implementation."""

from ..standard.sex import StandardSexClassifier


class UKSexClassifier(StandardSexClassifier):
    """Sex/gender classifier for UK Census data."""
    
    def __init__(self, threshold=None):
        super().__init__(threshold=threshold) 