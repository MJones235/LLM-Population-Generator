"""CSV export functionality for population data."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional


class CSVExporter:
    """Handles CSV export of population data in normalized format."""
    
    def __init__(self, output_dir: Path):
        """Initialize CSV exporter.
        
        Args:
            output_dir: Directory where CSV files will be saved
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_normalized_csv(self, households: List[Dict[str, Any]], output_name: str) -> Optional[Path]:
        """Export household data to normalized CSV format.
        
        Args:
            households: List of household dictionaries
            output_name: Base name for output file (without extension)
            
        Returns:
            Path to saved CSV file, or None if no data
        """
        if not households:
            print("Warning: No household data to export")
            return None
        
        # Convert to normalized format
        people_data = []
        for household_idx, household in enumerate(households, 1):
            # Handle {"household": [...]} format
            if isinstance(household, dict) and 'household' in household:
                household_people = household['household']
            else:
                continue
                
            for person_idx, person in enumerate(household_people, 1):
                if isinstance(person, dict):
                    record = {
                        'household_id': household_idx,
                        'person_id': person_idx,
                        **person  # Include all person attributes
                    }
                    people_data.append(record)
        
        if not people_data:
            print("Warning: No individual person data found in households")
            return None
        
        # Create DataFrame and save
        df = pd.DataFrame(people_data)
        output_path = self.output_dir / f"{output_name}.csv"
        df.to_csv(output_path, index=False)
        return output_path