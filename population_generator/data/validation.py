"""
Fundamental validation framework for LLM-generated data.

This module provides the core validation infrastructure that can be used
across different domains and data types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass


@dataclass
class ValidationError:
    """Details of a validation error."""
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


def create_custom_validator(
    rules: Optional[List[ValidationRule]] = None
) -> CustomValidator:
    """Create a validator with specified rules.
    
    Args:
        rules: List of validation rules to include
        
    Returns:
        Configured CustomValidator
    """
    return CustomValidator(rules)