"""UK Census household composition data preprocessor.

This preprocessor converts UK Census household composition data from the standard ONS format
to the standardized CSV format expected by the library.

It uses the same household composition categories as the UKHouseholdCompositionClassifier to ensure consistency.
"""

from typing import Dict, List
from .census_preprocessor import UKCensusPreprocessor
from ...classifiers.uk.household_composition import UKHouseholdCompositionClassifier


class UKHouseholdCompositionPreprocessor(UKCensusPreprocessor):
    """Preprocessor for UK Census household composition data.
    
    This handles the standard UK Census household composition data format.
    """
    
    def __init__(self):
        """Initialize with household composition classifier for consistent mappings."""
        self._household_composition_classifier = UKHouseholdCompositionClassifier()
        super().__init__()
    
    def _build_category_mapping(self) -> Dict[str, str]:
        """Build mapping from UK Census household composition categories to standardized categories.
        
        Returns:
            Dictionary mapping census household composition descriptions to standard labels
        """
        # Use the classifier's label mapping to ensure consistency
        label_map = self._household_composition_classifier.get_label_map()
        
        # Add None mapping for "Does not apply" category (which has 0 observations)
        category_mapping = label_map.copy()
        category_mapping["Does not apply"] = None
        
        return category_mapping
    
    def _get_expected_columns(self) -> List[str]:
        """Get expected column names for UK Census household composition data.
        
        Returns:
            List of expected column names
        """
        return [
            "Upper tier local authorities Code",
            "Upper tier local authorities",
            "Household composition (8 categories) Code",
            "Household composition (8 categories)",
            "Observation"
        ]
    
    def get_classifier_categories(self) -> List[str]:
        """Get the household composition categories used by the corresponding classifier.
        
        Returns:
            List of household composition categories from the classifier
        """
        return self._household_composition_classifier.get_label_order()
    
    def validate_with_classifier(self, data: Dict[str, float]):
        """Validate processed data against the household composition classifier categories."""
        is_valid, errors = super().validate_with_classifier(data)
        
        # Additional household composition-specific validations
        classifier_categories = set(self.get_classifier_categories())
        data_categories = set(data.keys())
        
        if data_categories != classifier_categories:
            missing = classifier_categories - data_categories
            extra = data_categories - classifier_categories
            
            if missing:
                errors.append(f"Missing household composition categories needed by classifier: {missing}")
            if extra:
                errors.append(f"Extra household composition categories not expected by classifier: {extra}")
            
            is_valid = False
        
        return is_valid, errors