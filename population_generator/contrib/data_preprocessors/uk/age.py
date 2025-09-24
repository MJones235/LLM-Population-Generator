"""UK Census age data preprocessor.

This preprocessor converts UK Census age data from the standard ONS format
to the standardized CSV format expected by the library.

It uses the same age band mappings as the UKAgeClassifier to ensure consistency.
"""

from typing import Dict, List
from .census_preprocessor import UKCensusPreprocessor
from ...classifiers.uk.age import UKAgeClassifier


class UKAgePreprocessor(UKCensusPreprocessor):
    """Preprocessor for UK Census age data.
    
    This handles the standard UK Census age data format with 18 age categories.
    It maps these to the broader age bands used by the UKAgeClassifier.
    """
    
    def __init__(self):
        """Initialize with age classifier for consistent mappings."""
        self._age_classifier = UKAgeClassifier()
        super().__init__()
    
    def _build_category_mapping(self) -> Dict[str, str]:
        """Build mapping from UK Census age categories to standardized categories.
        
        Returns:
            Dictionary mapping census age descriptions to age band labels
        """
        # Use the classifier's age band labels to ensure consistency
        age_band_labels = self._age_classifier.get_label_order()
        
        # Map UK Census 18 age categories to the classifier's age bands
        # Updated to match actual UK Census category names
        return {
            "Aged 4 years and under": age_band_labels[0],    # "0-9"
            "Aged 5 to 9 years": age_band_labels[0],         # "0-9"
            "Aged 10 to 15 years": age_band_labels[1],       # "10-19"
            "Aged 16 to 19 years": age_band_labels[1],       # "10-19" 
            "Aged 20 to 24 years": age_band_labels[2],       # "20-29"
            "Aged 25 to 29 years": age_band_labels[2],       # "20-29"
            "Aged 30 to 34 years": age_band_labels[3],       # "30-39"
            "Aged 35 to 39 years": age_band_labels[3],       # "30-39"
            "Aged 40 to 44 years": age_band_labels[4],       # "40-49"
            "Aged 45 to 49 years": age_band_labels[4],       # "40-49"
            "Aged 50 to 54 years": age_band_labels[5],       # "50-59"
            "Aged 55 to 59 years": age_band_labels[5],       # "50-59"
            "Aged 60 to 64 years": age_band_labels[6],       # "60-69"
            "Aged 65 to 69 years": age_band_labels[6],       # "60-69"
            "Aged 70 to 74 years": age_band_labels[7],       # "70-79"
            "Aged 75 to 79 years": age_band_labels[7],       # "70-79"
            "Aged 80 to 84 years": age_band_labels[8],       # "80+"
            "Aged 85 years and over": age_band_labels[8]     # "80+"
        }
    
    def _get_expected_columns(self) -> List[str]:
        """Get expected column names for UK Census age data.
        
        Returns:
            List of expected column names
        """
        return [
            "Upper tier local authorities Code",
            "Upper tier local authorities", 
            "Age (18 categories) Code",
            "Age (18 categories)",
            "Observation"
        ]
    
    def get_classifier_categories(self) -> List[str]:
        """Get the age categories used by the corresponding classifier.
        
        Returns:
            List of age band labels from the classifier
        """
        return self._age_classifier.get_label_order()
    
    def validate_with_classifier(self, data: Dict[str, float]):
        """Validate processed data against the age classifier categories.
        
        Uses the parent method but also checks age-specific constraints.
        """
        is_valid, errors = super().validate_with_classifier(data)
        
        # Additional age-specific validations
        classifier_categories = set(self.get_classifier_categories())
        data_categories = set(data.keys())
        
        if data_categories != classifier_categories:
            missing = classifier_categories - data_categories
            extra = data_categories - classifier_categories
            
            if missing:
                errors.append(f"Missing age categories needed by classifier: {missing}")
            if extra:
                errors.append(f"Extra age categories not expected by classifier: {extra}")
            
            is_valid = False
        
        return is_valid, errors