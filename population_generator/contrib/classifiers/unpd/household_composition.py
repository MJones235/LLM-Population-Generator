"""UN Population Division household composition classifier."""

from ....classifiers.base import HouseholdLevelClassifier
from typing import Dict, List
import pandas as pd


class UNPDHouseholdCompositionClassifier(HouseholdLevelClassifier):
    """Household composition classifier for UN Population Division global data."""
    
    def get_name(self) -> str:
        """Get classifier name."""
        return 'unpd'

    def classify_household(self, household_df: pd.DataFrame, relationship_col: str = "relationship", **kwargs) -> str:
        """Classify household structure based on relationships using global demographic patterns.
        
        Args:
            household_df: DataFrame containing household members
            relationship_col: Column name containing relationship information
            **kwargs: Additional parameters (unused)
            
        Returns:
            String label for household composition type
        """
        return self.classify_household_structure(household_df, relationship_col)
    
    def classify_household_structure(self, group: pd.DataFrame, relationship_col: str = "relationship") -> str:
        """Classify household structure using UN global demographic categories.
        
        Args:
            group: DataFrame containing household members
            relationship_col: Column name containing relationship information
            
        Returns:
            String label for household composition type
        """
        roles = group[relationship_col].tolist()
        n = len(roles)

        if n == 1:
            return "One-person"

        has_partner = any(r in {"Spouse", "Partner"} for r in roles)
        has_child = any(r == "Child" for r in roles)

        # Couple only
        if n == 2 and has_partner:
            return "Couple only"

        # Nuclear couple with children — no other roles
        if has_partner and has_child and all(r in {"Head", "Spouse", "Partner", "Child"} for r in roles):
            return "Couple with children"

        # Single parent with children only
        if not has_partner and has_child and all(r in {"Head", "Child"} for r in roles):
            return "Single parent with children"

        # Reverse nuclear families (e.g., adult child is Head, living with parents and possibly siblings)
        if not has_partner and not has_child:
            reverse_roles = {"Head", "Parent", "Sibling"}
            if all(r in reverse_roles for r in roles):
                n_parents = sum(r == "Parent" for r in roles)
                if n_parents == 2:
                    return "Couple with children"
                elif n_parents == 1:
                    return "Single parent with children"

        # Extended family: all members are relatives
        all_relatives = all(
            r in {
                "Head", "Spouse", "Partner", "Child", "Child-in-law",
                "Parent", "Sibling", "Sibling-in-law",
                "Grandchild", "Grandparent", "Aunt", "Uncle",
                "Nephew", "Niece", "Cousin"
            }
            for r in roles
        )
        if all_relatives:
            return "Extended family"

        return "Non-relatives"
    
    def compute_observed_distribution(self, synthetic_df: pd.DataFrame, relationship_col: str = "relationship", **kwargs) -> Dict[str, float]:
        """Compute household composition distribution from synthetic population data.
        
        Args:
            synthetic_df: DataFrame with synthetic population data
            relationship_col: Column name containing relationship information
            **kwargs: Additional parameters (unused)
            
        Returns:
            Dictionary mapping composition types to percentages
        """
        if synthetic_df.empty:
            return {comp: 0.0 for comp in self.get_label_order()}
            
        household_labels = synthetic_df.groupby("household_id").apply(
            lambda x: self.classify_household_structure(x, relationship_col)
        )
        counts = household_labels.value_counts(normalize=True) * 100
        
        # Ensure all categories are represented
        result = counts.to_dict()
        for comp in self.get_label_order():
            if comp not in result:
                result[comp] = 0.0
                
        return result
    
    def get_label_order(self) -> List[str]:
        """Get ordered list of household composition labels.
        
        Returns:
            List of composition types in logical order
        """
        return [
            "One-person",
            "Single parent with children",
            "Couple only",
            "Couple with children",
            "Extended family",
            "Non-relatives"
        ]
    
    def get_label_map(self) -> Dict[str, str]:
        """Get mapping from internal labels to display labels.
        
        Returns:
            Dictionary mapping composition types to descriptive labels
        """
        return {
            "One-person": "One-person household",
            "Single parent with children": "Single parent with children",
            "Couple only": "Couple without children",
            "Couple with children": "Couple with children",
            "Extended family": "Extended family household",
            "Non-relatives": "Non-relatives household"
        }