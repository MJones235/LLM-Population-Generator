"""Flexible data loading system for demographic target data.

This module provides utilities to load demographic data from various formats:
- Raw UK Census CSV files (with codes and descriptions)
- Preprocessed percentage files
- JSON format data
- Custom data structures

The system automatically detects format and converts to standardized classifier format.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import pandas as pd
import json
import re
from dataclasses import dataclass


@dataclass
class DatasetMetadata:
    """Metadata about a demographic dataset."""
    name: str
    source: str
    year: Optional[int] = None
    region: Optional[str] = None
    classification_type: Optional[str] = None
    description: Optional[str] = None


class DataLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """Check if this loader can handle the given file."""
        pass
    
    @abstractmethod
    def load(self, file_path: Path, **kwargs) -> Dict[str, float]:
        """Load data and return as percentage distribution."""
        pass
    
    @abstractmethod
    def get_metadata(self, file_path: Path) -> Optional[DatasetMetadata]:
        """Extract metadata from the file if available."""
        pass


class UKCensusRawLoader(DataLoader):
    """Loader for raw UK Census CSV files with codes and descriptions."""
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if file appears to be raw UK Census format."""
        if not file_path.suffix.lower() == '.csv':
            return False
            
        try:
            # Read first few lines to check structure
            df = pd.read_csv(file_path, nrows=5)
            
            # Check for typical UK Census columns
            required_patterns = [
                r'.*[Cc]ode.*',  # Some kind of code column
                r'.*[Oo]bservation.*|.*[Cc]ount.*|.*[Nn]umber.*',  # Observation/count column
            ]
            
            columns = df.columns.tolist()
            
            # Must have at least 3 columns and match patterns
            if len(columns) < 3:
                return False
                
            matches = 0
            for pattern in required_patterns:
                if any(re.match(pattern, col) for col in columns):
                    matches += 1
                    
            return matches >= 1 and 'Observation' in columns
            
        except Exception:
            return False
    
    def load(self, file_path: Path, **kwargs) -> Dict[str, float]:
        """Load raw UK Census data and convert to percentages."""
        df = pd.read_csv(file_path)
        
        # Auto-detect key columns
        observation_col = self._find_column(df, [r'.*[Oo]bservation.*', r'.*[Cc]ount.*'])
        category_col = self._find_category_column(df)
        
        if not observation_col or not category_col:
            raise ValueError(f"Cannot identify observation and category columns in {file_path}")
        
        # Filter out zero or invalid observations
        df_clean = df[df[observation_col] > 0].copy()
        
        # Extract category labels and values
        categories = {}
        for _, row in df_clean.iterrows():
            category = self._extract_category_label(row, category_col)
            if category:
                categories[category] = categories.get(category, 0) + row[observation_col]
        
        # Convert to percentages
        total = sum(categories.values())
        if total == 0:
            return {}
            
        return {
            category: round((count / total) * 100, 2)
            for category, count in categories.items()
        }
    
    def _find_column(self, df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
        """Find column matching any of the given regex patterns."""
        for pattern in patterns:
            for col in df.columns:
                if re.match(pattern, col, re.IGNORECASE):
                    return col
        return None
    
    def _find_category_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the column that contains category information."""
        # Look for columns that contain category descriptions, not codes or observations
        exclude_patterns = [
            r'.*[Cc]ode.*',
            r'.*[Oo]bservation.*',
            r'.*[Cc]ount.*',
            r'.*[Nn]umber.*',
            r'.*local authorities.*'  # Exclude geographic identifiers
        ]
        
        # Priority patterns for category columns
        priority_patterns = [
            r'.*size.*categories.*',  # "Household size (9 categories)"
            r'.*categories.*',
            r'.*type.*',
            r'.*composition.*'
        ]
        
        # First, try to find high-priority category columns
        for pattern in priority_patterns:
            for col in df.columns:
                if re.match(pattern, col, re.IGNORECASE) and df[col].dtype == 'object':
                    return col
        
        # Fallback: find any non-excluded text column
        for col in df.columns:
            is_excluded = any(re.match(pattern, col, re.IGNORECASE) for pattern in exclude_patterns)
            if not is_excluded and df[col].dtype == 'object':
                return col
                
        return None
    
    def _extract_category_label(self, row: pd.Series, category_col: str) -> Optional[str]:
        """Extract a clean category label from the row."""
        category = str(row[category_col]).strip()
        
        # Handle household size patterns
        if 'people in household' in category.lower():
            # Extract number from patterns like "1 person in household", "8 or more people in household"
            if 'or more' in category.lower():
                # Handle "8 or more people" -> "8+"
                match = re.search(r'(\d+)\s+or\s+more', category)
                if match:
                    return f"{match.group(1)}+"
            else:
                # Extract number from "N people in household"
                match = re.search(r'^(\d+)\s+people?\s+in\s+household', category)
                if match:
                    return match.group(1)
        
        # General number extraction as fallback
        if any(word in category.lower() for word in ['people', 'household', 'size']):
            match = re.search(r'(\d+)', category)
            if match:
                num = match.group(1)
                if 'or more' in category.lower():
                    return f"{num}+"
                return num
        
        # Remove common prefixes/patterns for other cases
        category = re.sub(r'^\d+\s+', '', category)  # Remove leading numbers
        category = re.sub(r'^people?\s+in\s+household\s*:?\s*', '', category, flags=re.IGNORECASE)
        category = re.sub(r'^household\s+size\s*:?\s*', '', category, flags=re.IGNORECASE)
        
        return category if category and category != 'nan' else None
    
    def get_metadata(self, file_path: Path) -> Optional[DatasetMetadata]:
        """Extract metadata from UK Census file."""
        try:
            df = pd.read_csv(file_path, nrows=10)
            
            # Try to extract region from data
            region = None
            for col in df.columns:
                if 'local authorities' in col.lower() or 'area' in col.lower():
                    region = df[col].iloc[0] if len(df) > 0 else None
                    break
            
            # Try to infer classification type from filename or columns
            classification_type = None
            filename = file_path.stem.lower()
            if 'household' in filename and 'size' in filename:
                classification_type = 'household_size'
            elif 'household' in filename and ('composition' in filename or 'type' in filename):
                classification_type = 'household_composition'
            elif 'age' in filename:
                classification_type = 'age'
            elif 'sex' in filename or 'gender' in filename:
                classification_type = 'sex'
            
            return DatasetMetadata(
                name=file_path.stem,
                source="UK Census",
                region=region,
                classification_type=classification_type,
                description=f"Raw UK Census data from {file_path.name}"
            )
        except Exception:
            return None


class PreprocessedLoader(DataLoader):
    """Loader for preprocessed CSV files with Category,Percentage format."""
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if file is preprocessed format."""
        if not file_path.suffix.lower() == '.csv':
            return False
            
        try:
            df = pd.read_csv(file_path, nrows=3)
            columns = [col.lower().strip() for col in df.columns]
            
            # Check for category/percentage pattern
            has_category = any('category' in col or 'label' in col or col.startswith('category_') for col in columns)
            has_percentage = any('percentage' in col or 'percent' in col or col == '%' for col in columns)
            
            return has_category and has_percentage and len(df.columns) <= 3
            
        except Exception:
            return False
    
    def load(self, file_path: Path, **kwargs) -> Dict[str, float]:
        """Load preprocessed CSV data with Category,Percentage columns."""
        df = pd.read_csv(file_path)
        
        # Expected format: Category, Percentage columns
        category_col = None
        percentage_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'category' in col_lower or 'group' in col_lower:
                category_col = col
            elif 'percentage' in col_lower or 'percent' in col_lower or 'prop' in col_lower:
                percentage_col = col
        
        if not category_col or not percentage_col:
            raise ValueError(f"Could not find category and percentage columns in {file_path}")
        
        # Create distribution dictionary
        distribution = {}
        for _, row in df.iterrows():
            category = row[category_col]
            # Convert to string, but handle integers nicely
            if isinstance(category, float) and category.is_integer():
                category = str(int(category))
            else:
                category = str(category)
            
            percentage = float(row[percentage_col])
            if percentage > 0:  # Only include non-zero percentages
                distribution[category] = percentage
        
        return distribution
    
    def get_metadata(self, file_path: Path) -> Optional[DatasetMetadata]:
        """Extract metadata from preprocessed file."""
        return DatasetMetadata(
            name=file_path.stem,
            source="Preprocessed",
            description=f"Preprocessed demographic data from {file_path.name}"
        )


class JSONLoader(DataLoader):
    """Loader for JSON format data."""
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if file is JSON format."""
        return file_path.suffix.lower() == '.json'
    
    def load(self, file_path: Path, **kwargs) -> Dict[str, float]:
        """Load JSON data."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            if 'data' in data and isinstance(data['data'], dict):
                # {"data": {"category1": 25.0, "category2": 30.0}}
                return {str(k): float(v) for k, v in data['data'].items()}
            elif all(isinstance(v, (int, float)) for v in data.values()):
                # {"category1": 25.0, "category2": 30.0}
                return {str(k): float(v) for k, v in data.items()}
        
        raise ValueError(f"Unsupported JSON structure in {file_path}")
    
    def get_metadata(self, file_path: Path) -> Optional[DatasetMetadata]:
        """Extract metadata from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'metadata' in data:
                meta = data['metadata']
                return DatasetMetadata(
                    name=meta.get('name', file_path.stem),
                    source=meta.get('source', 'JSON'),
                    year=meta.get('year'),
                    region=meta.get('region'),
                    classification_type=meta.get('classification_type'),
                    description=meta.get('description')
                )
        except Exception:
            pass
            
        return DatasetMetadata(
            name=file_path.stem,
            source="JSON",
            description=f"JSON demographic data from {file_path.name}"
        )


class FlexibleDataManager:
    """Manager for loading demographic data from various formats."""
    
    def __init__(self):
        self.loaders = [
            UKCensusRawLoader(),
            PreprocessedLoader(), 
            JSONLoader()
        ]
        self._cache = {}
    
    def load_target_data(self, file_path: Union[str, Path], force_reload: bool = False) -> Dict[str, float]:
        """Load target data from file, auto-detecting format.
        
        Args:
            file_path: Path to data file
            force_reload: If True, bypass cache and reload from file
            
        Returns:
            Dictionary mapping categories to percentages
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Check cache
        cache_key = str(file_path.absolute())
        if not force_reload and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try each loader
        for loader in self.loaders:
            if loader.can_handle(file_path):
                try:
                    data = loader.load(file_path)
                    self._cache[cache_key] = data
                    return data
                except Exception as e:
                    print(f"Warning: {loader.__class__.__name__} failed to load {file_path}: {e}")
                    continue
        
        raise ValueError(f"No suitable loader found for {file_path}")
    
    def get_metadata(self, file_path: Union[str, Path]) -> Optional[DatasetMetadata]:
        """Get metadata for a data file."""
        file_path = Path(file_path)
        
        for loader in self.loaders:
            if loader.can_handle(file_path):
                return loader.get_metadata(file_path)
        
        return None
    
    def list_supported_formats(self) -> List[str]:
        """List supported data formats."""
        return [
            "Raw UK Census CSV (with codes, descriptions, observations)",
            "Preprocessed CSV (Category, Percentage columns)",
            "JSON (simple key-value or structured with metadata)"
        ]
    
    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
    
    def register_loader(self, loader: DataLoader):
        """Register a custom data loader."""
        self.loaders.insert(0, loader)  # Insert at beginning for priority
