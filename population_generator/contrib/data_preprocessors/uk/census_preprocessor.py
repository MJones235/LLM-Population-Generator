"""Base UK Census data preprocessor.

This module provides the core functionality for preprocessing UK Census data
from the standard ONS CSV format to the standardized format expected by the library.
"""

from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
from abc import ABC, abstractmethod


class UKCensusPreprocessor(ABC):
    """Base class for UK Census data preprocessors.
    
    This handles the common structure of UK Census data files which have:
    - Area codes and names
    - Category codes and descriptions
    - Observation counts
    
    Subclasses implement specific logic for different demographic variables.
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self._category_mapping = self._build_category_mapping()
    
    @abstractmethod
    def _build_category_mapping(self) -> Dict[str, str]:
        """Build mapping from census categories to standardized categories.
        
        Returns:
            Dictionary mapping census category descriptions to standardized categories
        """
        pass
    
    @abstractmethod
    def _get_expected_columns(self) -> List[str]:
        """Get the expected column names for this data type.
        
        Returns:
            List of expected column names in the census data
        """
        pass
    
    def preprocess_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        area_code: Optional[str] = None,
        area_name: Optional[str] = None
    ) -> Dict[str, float]:
        """Preprocess a UK Census data file to standardized format.
        
        Args:
            input_path: Path to input CSV file in ONS format
            output_path: Path for output standardized CSV file
            area_code: Optional area code to filter by
            area_name: Optional area name to filter by
            
        Returns:
            Dictionary of category percentages
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If file format is invalid
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Load and validate data
        df = pd.read_csv(input_path)
        self._validate_columns(df)
        
        # Filter by area if specified
        if area_code or area_name:
            df = self._filter_by_area(df, area_code, area_name)
        
        # Process the data
        processed_data = self._process_data(df)
        
        # Save standardized output
        self._save_standardized_data(processed_data, output_path)
        
        return processed_data
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns are present.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        expected_cols = self._get_expected_columns()
        missing_cols = [col for col in expected_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def _filter_by_area(
        self, 
        df: pd.DataFrame, 
        area_code: Optional[str] = None,
        area_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Filter data by area code or name.
        
        Args:
            df: Input DataFrame
            area_code: Area code to filter by
            area_name: Area name to filter by
            
        Returns:
            Filtered DataFrame
        """
        if area_code:
            # Try different possible area code column names
            area_code_cols = [col for col in df.columns if 'Code' in col and 'area' in col.lower()]
            if area_code_cols:
                df = df[df[area_code_cols[0]] == area_code]
        
        if area_name:
            # Try different possible area name column names  
            area_name_cols = [col for col in df.columns if 'area' in col.lower() and 'Code' not in col]
            if area_name_cols:
                df = df[df[area_name_cols[0]] == area_name]
        
        return df
    
    def _process_data(self, df: pd.DataFrame) -> Dict[str, float]:
        """Process the census data to standardized format.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping categories to percentages
        """
        # Get the category and observation columns
        category_col = self._get_category_column(df)
        observation_col = 'Observation'
        
        # Group by category and sum observations
        grouped = df.groupby(category_col)[observation_col].sum()
        
        # Map categories to standardized categories and sum by standardized category
        standardized_counts = {}
        total_mapped_count = 0
        
        for category, count in grouped.items():
            std_category = self._map_category(category)
            if std_category:  # Only include mappable categories
                if std_category in standardized_counts:
                    standardized_counts[std_category] += count
                else:
                    standardized_counts[std_category] = count
                total_mapped_count += count
        
        # Convert to percentages using only the total of mapped categories
        percentages = {}
        for std_category, count in standardized_counts.items():
            percentage = (count / total_mapped_count) * 100
            percentages[std_category] = percentage
        
        return percentages
    
    def _get_category_column(self, df: pd.DataFrame) -> str:
        """Get the category column name from the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Name of the category column
            
        Raises:
            ValueError: If category column not found
        """
        # Look for columns that contain category descriptions (not codes)
        category_cols = [col for col in df.columns 
                        if 'categories' in col and 'Code' not in col]
        
        if not category_cols:
            raise ValueError("Could not find category column in data")
        
        return category_cols[0]
    
    def _map_category(self, category: str) -> Optional[str]:
        """Map census category to standardized category.
        
        Args:
            category: Census category description
            
        Returns:
            Standardized category or None if not mappable
        """
        return self._category_mapping.get(category)
    
    def _save_standardized_data(self, data: Dict[str, float], output_path: Path) -> None:
        """Save processed data in standardized CSV format.
        
        The output format matches the DataLoader expected format:
        category,percentage
        1,30.0
        2,35.0
        ...
        
        Args:
            data: Processed data dictionary
            output_path: Output file path
        """
        # Create DataFrame in standardized format expected by DataLoader
        df = pd.DataFrame([
            {'category': category, 'percentage': percentage}
            for category, percentage in data.items()
        ])
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV with exact format expected by DataLoader
        df.to_csv(output_path, index=False)
    
    def validate_with_classifier(self, data: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Validate processed data against corresponding classifier categories.
        
        Args:
            data: Processed data dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check that we have all expected categories
        expected_categories = set(self._category_mapping.values())
        actual_categories = set(data.keys())
        
        missing = expected_categories - actual_categories
        if missing:
            errors.append(f"Missing categories: {missing}")
        
        unexpected = actual_categories - expected_categories
        if unexpected:
            errors.append(f"Unexpected categories: {unexpected}")
        
        # Check percentages sum to ~100%
        total = sum(data.values())
        if abs(total - 100.0) > 1.0:  # Allow 1% tolerance
            errors.append(f"Percentages sum to {total:.1f}%, expected ~100%")
        
        return len(errors) == 0, errors