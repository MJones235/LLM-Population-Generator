"""UK Census household composition classifier."""

from ....classifiers.base import HouseholdLevelClassifier
from typing import Dict, List
import pandas as pd


class UKHouseholdCompositionClassifier(HouseholdLevelClassifier):
    """Household composition classifier for UK Census data."""
    
    def get_name(self) -> str:
        """Get classifier name."""
        return 'uk_census'

    def classify_household(self, household_df: pd.DataFrame, relationship_col: str = "relationship", **kwargs) -> str:
        """Classify household structure based on relationships.
        
        Args:
            household_df: DataFrame containing household members
            relationship_col: Column name containing relationship information
            
        Returns:
            String label for household composition type
        """
        n = len(household_df)
        roles = set(household_df[relationship_col])
        head = household_df[household_df[relationship_col] == "Head"]
        children = household_df[household_df[relationship_col] == "Child"]
        parents = household_df[household_df[relationship_col] == "Parent"]
        siblings = household_df[household_df[relationship_col] == "Sibling"]

        if head.empty:
            return "No head of household"

        if n == 1:
            return (
                "One-person household: Aged 66 years and over"
                if head.iloc[0]["age"] >= 66
                else "One-person household: Other"
            )

        allowed_family_roles = {"Head", "Partner", "Spouse", "Child"}
        if roles.issubset(allowed_family_roles):
            if "Partner" in roles or "Spouse" in roles:
                if children.empty:
                    return "Single family household: Couple family household: No children"
                elif any(children["age"] < 18):
                    return "Single family household: Couple family household: Dependent children"
                else:
                    return "Single family household: Couple family household: All children non-dependent"
            else:
                return "Single family household: Lone parent household"

        # Handle reverse family structures
        allowed_reverse_family_roles = {"Head", "Parent", "Sibling"}
        if roles.issubset(allowed_reverse_family_roles):
            parent_count = len(parents)
            head_age = head.iloc[0]["age"]
            sibling_ages = siblings["age"].tolist()

            if parent_count == 1:
                return "Single family household: Lone parent household"
            elif parent_count == 2:
                all_ages = [head_age] + sibling_ages
                if any(age < 18 for age in all_ages):
                    return "Single family household: Couple family household: Dependent children"
                else:
                    return "Single family household: Couple family household: All children non-dependent"
            else:
                return "Other household types"

        return "Other household types"

    def get_label_order(self) -> List[str]:
        """Get ordered list of composition labels."""
        return [
            "One-person aged <66 years",
            "One-person aged 66+ years",
            "Lone parent",
            "Couple",
            "Couple with dependent children",
            "Couple with non-dependent children",
            "Other",
        ]

    def get_label_map(self) -> Dict[str, str]:
        """Map internal labels to display labels."""
        return {
            "One-person household: Aged 66 years and over": "One-person aged 66+ years",
            "One-person household: Other": "One-person aged <66 years",
            "Single family household: Lone parent household": "Lone parent",
            "Single family household: Couple family household: No children": "Couple",
            "Single family household: Couple family household: Dependent children": "Couple with dependent children",
            "Single family household: Couple family household: All children non-dependent": "Couple with non-dependent children",
            "Other household types": "Other",
        }
