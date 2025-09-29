"""
Standard household validation rules.

This module provides common validation rules for household population data.
These are example implementations that can be used as-is or as templates
for domain-specific validation logic.
"""

from typing import List, Optional
from ....data.validation import ValidationRule, FunctionValidationRule, CustomValidator


class HouseholdValidationRules:
    """Common validation rules for household population data."""
    
    @staticmethod
    def exactly_one_head_rule() -> ValidationRule:
        """Rule: Each household must have exactly one person with relationship 'Head'."""
        
        def validate_one_head(data) -> str:
            if not isinstance(data, list):
                return "Data must be a list of household members"
            
            heads = [person for person in data if 
                    isinstance(person, dict) and 
                    person.get("relationship", "").lower() == "head"]
            
            if len(heads) == 0:
                return "Household must have exactly one person with relationship 'Head' (found 0)"
            elif len(heads) > 1:
                return f"Household must have exactly one person with relationship 'Head' (found {len(heads)})"
            
            return ""  # Valid
        
        return FunctionValidationRule(
            name="exactly_one_head",
            validation_function=validate_one_head,
            description="Each household must have exactly one person with relationship 'Head'",
            severity="error"
        )
    
    @staticmethod
    def no_minors_living_alone_rule() -> ValidationRule:
        """Rule: No one under 18 should be able to live alone."""
        
        def validate_no_minors_alone(data) -> str:
            if not isinstance(data, list):
                return "Data must be a list of household members"
            
            if len(data) == 1:
                person = data[0]
                if isinstance(person, dict):
                    age = person.get("age")
                    if isinstance(age, (int, float)) and age < 18:
                        return f"Person under 18 (age {age}) cannot live alone"
            
            return ""  # Valid
        
        return FunctionValidationRule(
            name="no_minors_living_alone",
            validation_function=validate_no_minors_alone,
            description="No one under 18 should be able to live alone",
            severity="error"
        )
    
    @staticmethod
    def valid_relationships_rule(valid_relationships: Optional[List[str]] = None) -> ValidationRule:
        """Rule: All relationships must be from a valid set."""
        
        if valid_relationships is None:
            valid_relationships = ["Head", "Partner", "Child", "Parent", "Sibling", "Other", "Relative"]
        
        valid_lower = [rel.lower() for rel in valid_relationships]
        
        def validate_relationships(data) -> List[str]:
            if not isinstance(data, list):
                return ["Data must be a list of household members"]
            
            errors = []
            for i, person in enumerate(data):
                if isinstance(person, dict):
                    relationship = person.get("relationship", "")
                    if relationship and relationship.lower() not in valid_lower:
                        errors.append(f"Person {i+1} has invalid relationship '{relationship}'. Valid options: {', '.join(valid_relationships)}")
            
            return errors
        
        return FunctionValidationRule(
            name="valid_relationships",
            validation_function=validate_relationships,
            description=f"All relationships must be one of: {', '.join(valid_relationships)}",
            severity="error"
        )
    
    @staticmethod
    def age_relationship_consistency_rule() -> ValidationRule:
        """Rule: Age and relationship should be consistent (e.g., children should be younger than parents)."""
        
        def validate_age_relationships(data) -> List[str]:
            if not isinstance(data, list):
                return ["Data must be a list of household members"]
            
            errors = []
            
            # Extract people by relationship
            heads = [p for p in data if isinstance(p, dict) and p.get("relationship", "").lower() == "head"]
            partners = [p for p in data if isinstance(p, dict) and p.get("relationship", "").lower() == "partner"]
            children = [p for p in data if isinstance(p, dict) and p.get("relationship", "").lower() == "child"]
            parents = [p for p in data if isinstance(p, dict) and p.get("relationship", "").lower() == "parent"]
            
            # Get ages safely
            def get_age(person):
                age = person.get("age")
                return age if isinstance(age, (int, float)) else None
            
            # Check children are younger than heads/partners/parents
            adult_ages = []
            for adult_group in [heads, partners, parents]:
                for adult in adult_group:
                    age = get_age(adult)
                    if age is not None:
                        adult_ages.append(age)
            
            if adult_ages:
                min_adult_age = min(adult_ages)
                for child in children:
                    child_age = get_age(child)
                    if child_age is not None and child_age >= min_adult_age:
                        errors.append(f"Child (age {child_age}) should be younger than adults in household (youngest adult: {min_adult_age})")
            
            return errors
        
        return FunctionValidationRule(
            name="age_relationship_consistency",
            validation_function=validate_age_relationships,
            description="Age and relationship should be consistent (children younger than adults)",
            severity="warning"
        )
    
    @staticmethod
    def get_default_household_rules() -> List[ValidationRule]:
        """Get a default set of household validation rules."""
        return [
            HouseholdValidationRules.exactly_one_head_rule(),
            HouseholdValidationRules.no_minors_living_alone_rule(),
            HouseholdValidationRules.valid_relationships_rule(),
            HouseholdValidationRules.age_relationship_consistency_rule()
        ]


def create_custom_validator_for_households(
    additional_rules: Optional[List[ValidationRule]] = None,
    include_defaults: bool = True
) -> CustomValidator:
    """Create a validator with household-specific rules.
    
    Args:
        additional_rules: Additional custom rules to include
        include_defaults: Whether to include default household rules
        
    Returns:
        Configured CustomValidator
    """
    validator = CustomValidator()
    
    if include_defaults:
        default_rules = HouseholdValidationRules.get_default_household_rules()
        for rule in default_rules:
            validator.add_rule(rule)
    
    if additional_rules:
        for rule in additional_rules:
            validator.add_rule(rule)
    
    return validator