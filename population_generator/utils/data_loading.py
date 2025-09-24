"""Simple data loading system for demographic target data.

This module loads standardized demographic data from CSV files with 'category' and 'percentage' columns.

Users are responsible for preprocessing their raw data into this standard CSV format.
This keeps the library simple while providing maximum flexibility for different data sources.
"""

from typing import Dict, Union
from pathlib import Path
import pandas as pd


class DataLoader:
    """Loads standardized demographic data from CSV files.
    
    Expected CSV format:
        category,percentage
        1,30.0
        2,35.0
        3,15.0
        ...
    
    Users must preprocess raw data into this format.
    """
    
    def __init__(self):
        self._cache = {}
    
    def load_target_data(self, file_path: Union[str, Path], force_reload: bool = False) -> Dict[str, float]:
        """Load target data from standardized CSV file.
        
        Args:
            file_path: Path to CSV file with category,percentage columns
            force_reload: If True, bypass cache and reload from file
            
        Returns:
            Dictionary mapping categories to percentages
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if file_path.suffix.lower() != '.csv':
            raise ValueError(f"Only CSV files are supported. Got: {file_path.suffix}")
        
        # Check cache
        cache_key = str(file_path.absolute())
        if not force_reload and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Load CSV data
        data = self._load_csv(file_path)
        
        # Cache and return
        self._cache[cache_key] = data
        return data
    
    def _load_csv(self, file_path: Path) -> Dict[str, float]:
        """Load CSV file with category,percentage columns."""
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file {file_path}: {e}")
        
        # Find category and percentage columns
        category_col = self._find_category_column(df)
        percentage_col = self._find_percentage_column(df)
        
        if not category_col:
            raise ValueError(f"No 'category' column found in {file_path}. Expected columns: category, percentage")
        if not percentage_col:
            raise ValueError(f"No 'percentage' column found in {file_path}. Expected columns: category, percentage")
        
        # Convert to dictionary
        distribution = {}
        for _, row in df.iterrows():
            category = row[category_col]
            
            # Handle different category types
            if pd.isna(category):
                continue
            if isinstance(category, (int, float)) and not pd.isna(category):
                category = str(int(category)) if isinstance(category, float) and category.is_integer() else str(category)
            else:
                category = str(category).strip()
            
            if not category or category.lower() in ['nan', 'null', '']:
                continue
                
            try:
                percentage = float(row[percentage_col])
                if percentage > 0:  # Only include positive percentages
                    distribution[category] = percentage
            except (ValueError, TypeError):
                raise ValueError(f"Invalid percentage value in {file_path}: {row[percentage_col]}")
        
        if not distribution:
            raise ValueError(f"No valid data found in {file_path}")
            
        return distribution
    
    def _find_category_column(self, df: pd.DataFrame) -> str:
        """Find the category column."""
        for col in df.columns:
            if col.lower().strip() in ['category', 'categories', 'group', 'groups', 'label', 'labels']:
                return col
        return None
    
    def _find_percentage_column(self, df: pd.DataFrame) -> str:
        """Find the percentage column."""
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ['percentage', 'percent', '%', 'proportion', 'prop', 'rate']:
                return col
        return None
    
    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
