"""
Custom validation rules for LLM-generated data.

This module provides a framework for defining and applying custom validation rules
that go beyond JSON schema validation. Rules can be domain-specific and complex,
with detailed error tracking for academic research.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass


@dataclass
class ValidationError:
    """Details of a custom validation error."""
    rule_name: str
    error_message: str
    data_path: Optional[str] = None  # JSONPath to the problematic data
    rule_description: Optional[str] = None
    severity: str = "error"  # "error", "warning", "info"


class ValidationRule(ABC):
    """Abstract base class for custom validation rules."""
    
    def __init__(self, name: str, description: str = "", severity: str = "error"):
        """Initialize the validation rule.
        
        Args:
            name: Unique name for this rule
            description: Human-readable description of what this rule checks
            severity: Severity level ("error", "warning", "info")
        """
        self.name = name
        self.description = description
        self.severity = severity
    
    @abstractmethod
    def validate(self, data: Any) -> List[ValidationError]:
        """Validate the data against this rule.
        
        Args:
            data: The data to validate (usually parsed JSON)
            
        Returns:
            List of validation errors (empty if validation passes)
        """
        pass


class FunctionValidationRule(ValidationRule):
    """Validation rule that wraps a simple function."""
    
    def __init__(
        self, 
        name: str, 
        validation_function: Callable[[Any], Union[bool, str, List[str]]], 
        description: str = "",
        severity: str = "error"
    ):
        """Initialize with a validation function.
        
        Args:
            name: Unique name for this rule
            validation_function: Function that takes data and returns:
                - bool: True if valid, False if invalid
                - str: Error message if invalid, None/empty if valid
                - List[str]: List of error messages (empty if valid)
            description: Human-readable description
            severity: Severity level
        """
        super().__init__(name, description, severity)
        self.validation_function = validation_function
    
    def validate(self, data: Any) -> List[ValidationError]:
        """Validate using the wrapped function."""
        try:
            result = self.validation_function(data)
            
            if isinstance(result, bool):
                if not result:
                    return [ValidationError(
                        rule_name=self.name,
                        error_message=f"Validation rule '{self.name}' failed",
                        rule_description=self.description,
                        severity=self.severity
                    )]
                return []
            
            elif isinstance(result, str):
                if result:  # Non-empty string means error
                    return [ValidationError(
                        rule_name=self.name,
                        error_message=result,
                        rule_description=self.description,
                        severity=self.severity
                    )]
                return []
            
            elif isinstance(result, list):
                return [ValidationError(
                    rule_name=self.name,
                    error_message=error_msg,
                    rule_description=self.description,
                    severity=self.severity
                ) for error_msg in result if error_msg]
            
            else:
                return [ValidationError(
                    rule_name=self.name,
                    error_message=f"Validation function returned unexpected type: {type(result)}",
                    rule_description=self.description,
                    severity="error"
                )]
                
        except Exception as e:
            return [ValidationError(
                rule_name=self.name,
                error_message=f"Validation function raised exception: {str(e)}",
                rule_description=self.description,
                severity="error"
            )]


class CustomValidator:
    """Manages and applies multiple custom validation rules."""
    
    def __init__(self, rules: Optional[List[ValidationRule]] = None):
        """Initialize with a list of validation rules.
        
        Args:
            rules: List of validation rules to apply
        """
        self.rules = rules or []
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule.
        
        Args:
            rule: Validation rule to add
        """
        self.rules.append(rule)
    
    def add_function_rule(
        self, 
        name: str, 
        validation_function: Callable[[Any], Union[bool, str, List[str]]], 
        description: str = "",
        severity: str = "error"
    ) -> None:
        """Add a function-based validation rule.
        
        Args:
            name: Unique name for this rule
            validation_function: Validation function
            description: Human-readable description
            severity: Severity level
        """
        rule = FunctionValidationRule(name, validation_function, description, severity)
        self.add_rule(rule)
    
    def validate(self, data: Any) -> List[ValidationError]:
        """Apply all validation rules to the data.
        
        Args:
            data: Data to validate
            
        Returns:
            List of all validation errors found
        """
        all_errors = []
        for rule in self.rules:
            errors = rule.validate(data)
            all_errors.extend(errors)
        return all_errors
    
    def validate_with_summary(self, data: Any) -> Dict[str, Any]:
        """Validate and return a summary.
        
        Args:
            data: Data to validate
            
        Returns:
            Dictionary with validation results and summary
        """
        errors = self.validate(data)
        
        error_count = len([e for e in errors if e.severity == "error"])
        warning_count = len([e for e in errors if e.severity == "warning"])
        info_count = len([e for e in errors if e.severity == "info"])
        
        return {
            "valid": error_count == 0,
            "total_issues": len(errors),
            "errors": error_count,
            "warnings": warning_count,
            "info": info_count,
            "validation_errors": [
                {
                    "rule_name": error.rule_name,
                    "message": error.error_message,
                    "data_path": error.data_path,
                    "description": error.rule_description,
                    "severity": error.severity
                }
                for error in errors
            ]
        }


# Pre-defined validation rules for household data
class HouseholdValidationRules:
    """Common validation rules for household population data."""
    
    @staticmethod
    def exactly_one_head_rule() -> ValidationRule:
        """Rule: Each household must have exactly one person with relationship 'Head'."""
        
        def validate_one_head(data: Any) -> str:
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
        
        def validate_no_minors_alone(data: Any) -> str:
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
        
        def validate_relationships(data: Any) -> List[str]:
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
        
        def validate_age_relationships(data: Any) -> List[str]:
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