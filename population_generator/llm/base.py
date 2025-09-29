"""Base class for LLM interfaces."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Optional, NamedTuple
import json
import jsonschema
import time

from ..analysis.failures import GenerationFailureTracker
from ..data.validation import CustomValidator, ValidationError


class TokenUsage(NamedTuple):
    """Token usage statistics for a request."""
    input_tokens: int
    output_tokens: int
    total_tokens: int


class LLMResponse(NamedTuple):
    """Response from LLM with optional metadata."""
    content: Union[str, List[str]]
    token_usage: Optional[TokenUsage] = None
    model_info: Optional[Dict[str, Any]] = None


class BaseLLM(ABC):
    """Abstract base class for LLM interfaces.
    
    Provides a standard interface for local or API-based LLMs,
    including JSON generation and validation.
    """

    model_name: str
    temperature: float
    
    def __init__(self):
        self.track_tokens = False
        self.token_usage_history: List[TokenUsage] = []
        self.failure_tracker = GenerationFailureTracker()

    @abstractmethod
    def generate_text(self, prompt: Union[str, List[str]], timeout: int = 30) -> Union[str, List[str]]:
        """Generate text response from LLM.
        
        Args:
            prompt: Single prompt string or list of prompts
            timeout: Maximum time to wait for response
            
        Returns:
            Generated text response(s)
        """
        raise NotImplementedError
    
    @abstractmethod
    def generate_text_with_metadata(self, prompt: Union[str, List[str]], timeout: int = 30) -> LLMResponse:
        """Generate text response with metadata (including token usage).
        
        Args:
            prompt: Single prompt string or list of prompts
            timeout: Maximum time to wait for response
            
        Returns:
            LLMResponse with content and optional metadata
        """
        raise NotImplementedError
    
    def enable_failure_tracking(self, enabled: bool = True):
        """Enable or disable failure tracking.
        
        Args:
            enabled: Whether to track generation failures
        """
        if not enabled:
            self.failure_tracker.reset_session()
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get comprehensive failure statistics for academic research.
        
        Returns:
            Dictionary with detailed failure analysis suitable for research
        """
        return self.failure_tracker.get_academic_summary()
    
    def enable_token_tracking(self, enabled: bool = True):
        """Enable or disable token usage tracking.
        
        Args:
            enabled: Whether to track token usage
        """
        self.track_tokens = enabled
        if not enabled:
            self.token_usage_history.clear()
    
    def get_token_usage_summary(self) -> Dict[str, Any]:
        """Get summary of token usage across all requests.
        
        Returns:
            Dictionary with token usage statistics
        """
        if not self.token_usage_history:
            return {
                "total_requests": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "average_input_tokens": 0,
                "average_output_tokens": 0
            }
            
        total_input = sum(usage.input_tokens for usage in self.token_usage_history)
        total_output = sum(usage.output_tokens for usage in self.token_usage_history)
        total_requests = len(self.token_usage_history)
        
        return {
            "total_requests": total_requests,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "average_input_tokens": total_input / total_requests,
            "average_output_tokens": total_output / total_requests
        }
        
    @abstractmethod
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get model metadata information.
        
        Returns:
            Dictionary containing model information
        """
        raise NotImplementedError

    def generate_json(
        self,
        prompt: str,
        json_schema: Dict[str, Any],
        custom_validator: Optional[CustomValidator] = None,
        n_attempts: int = 3,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Generate and validate JSON response with comprehensive failure tracking.
        
        Args:
            prompt: Input prompt
            json_schema: JSON schema for validation
            custom_validator: Optional custom validator for additional rule checking
            n_attempts: Number of retry attempts
            timeout: Timeout per attempt
            
        Returns:
            Validated JSON response
            
        Raises:
            Exception: If all attempts fail
        """
        # Start tracking failures for this prompt
        prompt_hash = self.failure_tracker.start_prompt_tracking(prompt)
        
        attempts = 0
        current_prompt = prompt

        while attempts < n_attempts:
            attempts += 1
            start_time = time.time()
            
            try:
                if self.track_tokens:
                    llm_response = self.generate_text_with_metadata(current_prompt, timeout)
                    raw_response = llm_response.content
                    if llm_response.token_usage:
                        self.token_usage_history.append(llm_response.token_usage)
                    model_metadata = llm_response.model_info
                else:
                    raw_response = self.generate_text(current_prompt, timeout)
                    model_metadata = self.get_model_metadata()
                    
                if isinstance(raw_response, list):
                    raw_response = raw_response[0]
                raw_response = raw_response.strip()
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                self.failure_tracker.record_attempt_failure(
                    attempt_number=attempts,
                    failure_type='model_error',
                    error_message=str(e),
                    raw_response=None,
                    response_time_ms=response_time,
                    model_metadata=self.get_model_metadata()
                )
                print(f"Failed to generate response (attempt {attempts}): {str(e)}. Retrying...")
                continue

            response_time = (time.time() - start_time) * 1000
            
            try:
                data = json.loads(raw_response)
                jsonschema.validate(data, json_schema)
                
                # Apply custom validation if provided
                if custom_validator:
                    validation_errors = custom_validator.validate(data)
                    if validation_errors:
                        # Find the first error (we'll only report one per attempt)
                        first_error = validation_errors[0]
                        error_details = f"Custom validation failed - {first_error.rule_name}: {first_error.error_message}"
                        
                        self.failure_tracker.record_attempt_failure(
                            attempt_number=attempts,
                            failure_type='custom_validation',
                            error_message=error_details,
                            raw_response=raw_response,
                            response_time_ms=response_time,
                            model_metadata=model_metadata,
                            validation_rule_name=first_error.rule_name
                        )
                        print(f"Custom validation failed (attempt {attempts}): {error_details}. Retrying...")
                        
                        # Create an improved prompt with validation guidance
                        rule_descriptions = [f"- {error.rule_name}: {error.error_message}" for error in validation_errors[:3]]
                        guidance = "\n".join(rule_descriptions)
                        current_prompt = f"{prompt}\n\nPlease ensure your JSON response follows these rules:\n{guidance}"
                        continue
                
                # All validations passed!
                self.failure_tracker.record_prompt_success(attempts, response_time)
                return data
                
            except json.JSONDecodeError as e:
                self.failure_tracker.record_attempt_failure(
                    attempt_number=attempts,
                    failure_type='json_parse',
                    error_message=str(e),
                    raw_response=raw_response,
                    response_time_ms=response_time,
                    model_metadata=model_metadata
                )
                print(f"Invalid JSON response (attempt {attempts}): {str(e)}. Retrying...")
                current_prompt = f"{prompt}\n\nPlease ensure your response is valid JSON."
                
            except jsonschema.ValidationError as e:
                self.failure_tracker.record_attempt_failure(
                    attempt_number=attempts,
                    failure_type='schema_validation',
                    error_message=str(e),
                    raw_response=raw_response,
                    response_time_ms=response_time,
                    model_metadata=model_metadata
                )
                print(f"JSON validation failed (attempt {attempts}): {str(e)}. Retrying...")
                current_prompt = f"{prompt}\n\nPlease ensure your JSON response matches the required schema."

        # All attempts failed
        self.failure_tracker.record_prompt_final_failure()
        raise Exception(f"Failed to generate valid JSON after {n_attempts} attempts")

    def generate_batch_json(
        self,
        prompts: List[str],
        json_schema: Dict[str, Any],
        custom_validator: Optional[CustomValidator] = None,
        max_parallel: int = 5,
        timeout: int = 30
    ) -> List[Dict[str, Any]]:
        """Generate JSON responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            json_schema: JSON schema for validation
            custom_validator: Optional custom validator for additional rule checking
            max_parallel: Maximum parallel requests (for API models)
            timeout: Timeout per request
            
        Returns:
            List of validated JSON responses
        """
        results = []
        
        # Simple sequential processing for now
        # Subclasses can override for parallel processing
        for prompt in prompts:
            try:
                result = self.generate_json(prompt, json_schema, custom_validator, timeout=timeout)
                results.append(result)
            except Exception as e:
                print(f"Failed to generate JSON for prompt: {str(e)}")
                results.append({})  # Return empty dict on failure
                
        return results