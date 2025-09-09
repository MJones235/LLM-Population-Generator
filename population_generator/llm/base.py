"""Base class for LLM interfaces."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import json
import jsonschema
import time


class BaseLLM(ABC):
    """Abstract base class for LLM interfaces.
    
    Provides a standard interface for local or API-based LLMs,
    including JSON generation and validation.
    """

    is_local: bool = False
    model_name: str
    temperature: float

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
        n_attempts: int = 3,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Generate and validate JSON response.
        
        Args:
            prompt: Input prompt
            json_schema: JSON schema for validation
            n_attempts: Number of retry attempts
            timeout: Timeout per attempt
            
        Returns:
            Validated JSON response
            
        Raises:
            Exception: If all attempts fail
        """
        attempts = 0
        current_prompt = prompt

        while attempts < n_attempts:
            try:
                raw_response = self.generate_text(current_prompt, timeout)
                if isinstance(raw_response, list):
                    raw_response = raw_response[0]
                raw_response = raw_response.strip()
            except Exception as e:
                attempts += 1
                print(f"Failed to generate response: {str(e)}. Retrying...")
                continue

            try:
                data = json.loads(raw_response)
                jsonschema.validate(data, json_schema)
                return data
            except json.JSONDecodeError as e:
                attempts += 1
                print(f"Invalid JSON response: {str(e)}. Retrying...")
                current_prompt = f"{prompt}\n\nPlease ensure your response is valid JSON."
            except jsonschema.ValidationError as e:
                attempts += 1
                print(f"JSON validation failed: {str(e)}. Retrying...")
                current_prompt = f"{prompt}\n\nPlease ensure your JSON response matches the required schema."

        raise Exception(f"Failed to generate valid JSON after {n_attempts} attempts")

    def generate_batch_json(
        self,
        prompts: List[str],
        json_schema: Dict[str, Any],
        max_parallel: int = 5,
        timeout: int = 30
    ) -> List[Dict[str, Any]]:
        """Generate JSON responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            json_schema: JSON schema for validation
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
                result = self.generate_json(prompt, json_schema, timeout=timeout)
                results.append(result)
            except Exception as e:
                print(f"Failed to generate JSON for prompt: {str(e)}")
                results.append({})  # Return empty dict on failure
                
        return results
