"""UK Census sex/gender data preprocessor.

This preprocessor converts UK Census sex data from the standard ONS format
to the standardized CSV format expected by the library.

It uses the same gender categories as the UKSexClassifier to ensure consistency.
"""

from typing import Dict, List
from .census_preprocessor import UKCensusPreprocessor
from ...classifiers.uk.sex import UKSexClassifier


class UKSexPreprocessor(UKCensusPreprocessor):
    """Preprocessor for UK Census sex/gender data.
    
    This handles the standard UK Census sex data format with 2 categories.
    """
    
    def __init__(self):
        """Initialize with sex classifier for consistent mappings."""
        self._sex_classifier = UKSexClassifier()
        super().__init__()
    
    def _build_category_mapping(self) -> Dict[str, str]:
        """Build mapping from UK Census sex categories to standardized categories.
        
        Returns:
            Dictionary mapping census sex descriptions to standard labels
        """
        # Use the classifier's label mapping to ensure consistency
        # For sex, UK Census categories map directly to classifier categories
        return self._sex_classifier.get_label_map()
    
    def _get_expected_columns(self) -> List[str]:
        """Get expected column names for UK Census sex data.
        
        Returns:
            List of expected column names
        """
        return [
            "Upper tier local authorities Code",
            "Upper tier local authorities",
            "Sex (2 categories) Code", 
            "Sex (2 categories)",
            "Observation"
        ]
    
    def get_classifier_categories(self) -> List[str]:
        """Get the sex categories used by the corresponding classifier.
        
        Returns:
            List of sex categories from the classifier
        """
        return self._sex_classifier.get_label_order()
    
    def validate_with_classifier(self, data: Dict[str, float]):
        """Validate processed data against the sex classifier categories.
        
        Uses the parent method but also checks sex-specific constraints.
        """
        is_valid, errors = super().validate_with_classifier(data)
        
        # Additional sex-specific validations
        classifier_categories = set(self.get_classifier_categories())
        data_categories = set(data.keys())
        
        if data_categories != classifier_categories:
            missing = classifier_categories - data_categories
            extra = data_categories - classifier_categories
            
            if missing:
                errors.append(f"Missing sex categories needed by classifier: {missing}")
            if extra:
                errors.append(f"Extra sex categories not expected by classifier: {extra}")
            
            is_valid = False
        
        return is_valid, errors