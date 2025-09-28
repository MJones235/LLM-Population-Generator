"""Azure AI Foundry model implementation."""

import re
from typing import Dict, Any, List, Union
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from .base import BaseLLM, LLMResponse, TokenUsage


class FoundryModel(BaseLLM):
    """Azure AI Foundry model implementation."""
    
    def __init__(self, 
                 model_name: str = "DeepSeek-R1-0528", 
                 api_key: str = None, 
                 endpoint: str = "https://population-generator-resource.services.ai.azure.com/models",
                 api_version: str = "2024-05-01-preview",
                 temperature: float = 0.7, 
                 top_p: float = 0.95, 
                 top_k: int = 40, 
                 **kwargs):
        """Initialize Azure AI Foundry model.
        
        Args:
            model_name: Name of the model to use
            api_key: Azure AI Foundry API key
            endpoint: Azure AI Foundry endpoint URL
            api_version: API version to use
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        
        self.model_name = model_name
        self.client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
            api_version=api_version
        )
        self.endpoint = endpoint
        self.api_version = api_version
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.kwargs = kwargs

    def generate_text(self, prompt: Union[str, List[str]], timeout: int = 30) -> Union[str, List[str]]:
        """Generate text using Azure AI Foundry API.
        
        Args:
            prompt: Single prompt or list of prompts
            timeout: Request timeout in seconds
            
        Returns:
            Generated text response(s)
        """
        if isinstance(prompt, list):
            return [self._call_foundry(p) for p in prompt]
        return self._call_foundry(prompt)

    def generate_text_with_metadata(self, prompt: Union[str, List[str]], timeout: int = 30) -> LLMResponse:
        """Generate text with token usage metadata."""
        if isinstance(prompt, list):
            # Handle batch requests - sum up token usage
            responses = []
            total_input = 0
            total_output = 0
            
            for p in prompt:
                response = self._call_foundry_with_metadata(p)
                responses.append(response["content"])
                if response["usage"]:
                    total_input += response["usage"]["prompt_tokens"]
                    total_output += response["usage"]["completion_tokens"]
            
            token_usage = TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output
            ) if total_input > 0 else None
            
            return LLMResponse(content=responses, token_usage=token_usage)
        else:
            response = self._call_foundry_with_metadata(prompt)
            content = response["content"]
            
            token_usage = None
            if response["usage"]:
                usage = response["usage"]
                token_usage = TokenUsage(
                    input_tokens=usage["prompt_tokens"],
                    output_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"]
                )
            
            return LLMResponse(content=content, token_usage=token_usage)

    def _call_foundry(self, prompt: str) -> str:
        """Make a single request to Azure AI Foundry API."""
        try:
            response = self.client.complete(
                model=self.model_name,
                messages=[
                    SystemMessage("You are an expert demographic modeller generating realistic synthetic households for population simulation. Your goal is to produce one new household at a time, ensuring that the characteristics of each household and its members are plausible and reflect the statistical context provided."),
                    UserMessage(prompt)
                ],
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            raw_output = response.choices[0].message.content.strip()
            return re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
        except Exception as e:
            raise Exception(f"Azure AI Foundry API request failed: {str(e)}")

    def _call_foundry_with_metadata(self, prompt: str) -> Dict[str, Any]:
        """Make a single request to Azure AI Foundry API with full response metadata."""
        try:
            response = self.client.complete(
                model=self.model_name,
                messages=[
                    SystemMessage("You are an expert demographic modeller generating realistic synthetic households for population simulation. Your goal is to produce one new household at a time, ensuring that the characteristics of each household and its members are plausible and reflect the statistical context provided."),
                    UserMessage(prompt)
                ],
                temperature=self.temperature,
                top_p=self.top_p
            )

            raw_output = response.choices[0].message.content.strip()
            content = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
            
            usage_info = None
            if hasattr(response, 'usage') and response.usage:
                usage_info = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            return {
                "content": content,
                "usage": usage_info
            }
        except Exception as e:
            raise Exception(f"Azure AI Foundry API request failed: {str(e)}")

    def get_model_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "provider": "Azure AI Foundry",
            "model": self.model_name,
            "endpoint": self.endpoint,
            "api_version": self.api_version,
        }