"""UN Population Division sex classifier."""

from ..standard.sex import StandardSexClassifier


class UNPDSexClassifier(StandardSexClassifier):
    """Sex/gender classifier for UN Population Division data.
    
    Inherits standard sex classification logic but can be customized
    for UNPD-specific requirements if needed.
    """
    
    def __init__(self):
        """Initialize with UNPD region identifier."""
        super().__init__('unpd')
    
    # The standard implementation is suitable for UNPD data
    # Override methods below if UNPD-specific customization is needed
    
    # def get_standard_categories(self) -> List[str]:
    #     """Override for UNPD-specific gender categories if needed."""
    #     # Custom UNPD gender categories could be defined here
    #     return super().get_standard_categories()