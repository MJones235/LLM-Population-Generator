"""UK Census household size data preprocessor.

This preprocessor converts UK Census household size data from the standard ONS format
to the standardized CSV format expected by the library.

It uses the same household size categories as the UKHouseholdSizeClassifier to ensure consistency.
"""

from typing import Dict, List
from .census_preprocessor import UKCensusPreprocessor
from ...classifiers.uk.household_size import UKHouseholdSizeClassifier


class UKHouseholdSizePreprocessor(UKCensusPreprocessor):
    """Preprocessor for UK Census household size data.
    
    This handles the standard UK Census household size data format.
    """
    
    def __init__(self):
        """Initialize with household size classifier for consistent mappings."""
        self._household_size_classifier = UKHouseholdSizeClassifier()
        super().__init__()
    
    def _build_category_mapping(self) -> Dict[str, str]:
        """Build mapping from UK Census household size categories to standardized categories.
        
        Returns:
            Dictionary mapping census household size descriptions to standard labels
        """
        # Use the classifier's label order to ensure consistency
        classifier_labels = self._household_size_classifier.get_label_order()
        
        # Map UK Census 9 household size categories to classifier labels
        return {
            "0 people in household": None,  # Skip this category
            "1 person in household": classifier_labels[0],    # "1"
            "2 people in household": classifier_labels[1],    # "2"
            "3 people in household": classifier_labels[2],    # "3"
            "4 people in household": classifier_labels[3],    # "4"
            "5 people in household": classifier_labels[4],    # "5"
            "6 people in household": classifier_labels[5],    # "6"
            "7 people in household": classifier_labels[6],    # "7"
            "8 or more people in household": classifier_labels[7]  # "8"
        }
    
    def _get_expected_columns(self) -> List[str]:
        """Get expected column names for UK Census household size data.
        
        Returns:
            List of expected column names
        """
        return [
            "Upper tier local authorities Code",
            "Upper tier local authorities",
            "Household size (9 categories) Code",
            "Household size (9 categories)",
            "Observation"
        ]
    
    def get_classifier_categories(self) -> List[str]:
        """Get the household size categories used by the corresponding classifier.
        
        Returns:
            List of household size categories from the classifier
        """
        return self._household_size_classifier.get_label_order()
    
    def validate_with_classifier(self, data: Dict[str, float]):
        """Validate processed data against the household size classifier categories."""
        is_valid, errors = super().validate_with_classifier(data)
        
        # Additional household size-specific validations
        classifier_categories = set(self.get_classifier_categories())
        data_categories = set(data.keys())
        
        if data_categories != classifier_categories:
            missing = classifier_categories - data_categories
            extra = data_categories - classifier_categories
            
            if missing:
                errors.append(f"Missing household size categories needed by classifier: {missing}")
            if extra:
                errors.append(f"Extra household size categories not expected by classifier: {extra}")
            
            is_valid = False
        
        return is_valid, errors