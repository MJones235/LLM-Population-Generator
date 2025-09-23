"""Example implementation of Azure OpenAI LLM interface."""

import os
from openai import AzureOpenAI
from typing import Dict, Any, List, Union, Optional
from .base import BaseLLM, LLMResponse, TokenUsage


class OpenAIModel(BaseLLM):
    """Azure OpenAI GPT model implementation."""
    
    def __init__(self, 
                 api_key: str,
                 model_name: str = "gpt-4o",
                 azure_endpoint: Optional[str] = None,
                 api_version: str = "2024-08-01-preview",
                 temperature: float = 0.7,
                 top_p: float = 0.85,
                 top_k: int = 100):
        """Initialize Azure OpenAI model.
        
        Args:
            api_key: Azure OpenAI API key
            model_name: Deployment name (e.g., 'gpt-4o', 'gpt-35-turbo')
            azure_endpoint: Azure OpenAI endpoint URL. If None, reads from AZURE_OPENAI_ENDPOINT env var
            api_version: Azure OpenAI API version
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter (note: Azure OpenAI doesn't use top_k)
        """
        super().__init__()
        
        # Get Azure endpoint from parameter or environment variable
        if azure_endpoint is None:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not azure_endpoint:
                raise ValueError("Azure endpoint must be provided either as parameter or AZURE_OPENAI_ENDPOINT environment variable")
        
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version
        )
        self.model_name = model_name
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k  # Stored but not used by Azure OpenAI
        self.is_local = False
    
    def generate_text(self, prompt: Union[str, List[str]], timeout: int = 30) -> Union[str, List[str]]:
        """Generate text using Azure OpenAI API.
        
        Args:
            prompt: Single prompt or list of prompts
            timeout: Request timeout in seconds
            
        Returns:
            Generated text response(s)
        """
        if isinstance(prompt, list):
            # Handle batch requests
            responses = []
            for p in prompt:
                response = self._single_request(p, timeout)
                responses.append(response)
            return responses
        else:
            return self._single_request(prompt, timeout)
    
    def generate_text_with_metadata(self, prompt: Union[str, List[str]], timeout: int = 30) -> LLMResponse:
        """Generate text with token usage metadata."""
        if isinstance(prompt, list):
            # Handle batch requests - sum up token usage
            responses = []
            total_input = 0
            total_output = 0
            
            for p in prompt:
                response = self._single_request_with_metadata(p, timeout)
                responses.append(response.choices[0].message.content)
                if response.usage:
                    total_input += response.usage.prompt_tokens
                    total_output += response.usage.completion_tokens
            
            token_usage = TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output
            ) if total_input > 0 else None
            
            return LLMResponse(content=responses, token_usage=token_usage)
        else:
            response = self._single_request_with_metadata(prompt, timeout)
            content = response.choices[0].message.content
            
            token_usage = None
            if response.usage:
                token_usage = TokenUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
            
            return LLMResponse(content=content, token_usage=token_usage)

    def _single_request(self, prompt: str, timeout: int) -> str:
        """Make a single request to Azure OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,  # This should be the deployment name in Azure
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
                timeout=timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Azure OpenAI API request failed: {str(e)}")
    
    def _single_request_with_metadata(self, prompt: str, timeout: int):
        """Make a single request to Azure OpenAI API with full response metadata."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
                timeout=timeout
            )
            return response
        except Exception as e:
            raise Exception(f"Azure OpenAI API request failed: {str(e)}")
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "provider": "Azure OpenAI",
            "model": self.model_name,
            "azure_endpoint": self.azure_endpoint,
            "api_version": self.api_version,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "is_local": self.is_local
        }
