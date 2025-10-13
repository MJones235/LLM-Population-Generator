"""UK Census sex/gender classifier using standard implementation."""

from ..standard.sex import StandardSexClassifier


class UKSexClassifier(StandardSexClassifier):
    """Sex/gender classifier for UK Census data."""
    
    def __init__(self):
        """Initialize with UK Census region identifier."""
        super().__init__(threshold=None)  # Don't pass 'uk_census' as threshold