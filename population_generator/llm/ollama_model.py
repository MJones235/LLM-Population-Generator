"""Ollama local LLM interface implementation using LangChain."""

import subprocess
import multiprocessing
from typing import Dict, Any, List, Union, Optional

try:
    from langchain_ollama import OllamaLLM

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    OllamaLLM = None

from .base import BaseLLM, LLMResponse, TokenUsage


class OllamaModel(BaseLLM):
    """Ollama local LLM model implementation using LangChain."""

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        temperature: float = 0.7,
        top_p: float = 0.85,
        top_k: int = 40,
        **kwargs,
    ):
        """Initialize Ollama model using LangChain.

        Args:
            model_name: Name of the Ollama model (e.g., 'llama3.2:3b')
            temperature: Controls randomness in generation (0.0-1.0)
            top_p: Nucleus sampling parameter (0.0-1.0)
            top_k: Top-k sampling parameter
            **kwargs: Additional parameters for OllamaLLM
        """
        super().__init__()

        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain Ollama dependencies not available. "
                "Install with: pip install langchain-ollama"
            )

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self._extra_kwargs = kwargs

        # Ensure model is available
        self._ensure_model_available()

        # Initialize LangChain Ollama LLM
        self.llm = OllamaLLM(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs,
        )

    def generate_text(
        self, prompt: Union[str, List[str]], timeout: int = 30
    ) -> Union[str, List[str]]:
        """Generate text using LangChain Ollama.

        Args:
            prompt: Single prompt or list of prompts
            timeout: Request timeout in seconds

        Returns:
            Generated text response(s)
        """

        def call_llm(queue):
            """Execute LLM request in separate process for timeout control."""
            try:
                if isinstance(prompt, str):
                    result = self.llm.invoke(prompt)
                else:
                    result = self.llm.batch(prompt)
                queue.put(result)
            except Exception as e:
                queue.put(e)

        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=call_llm, args=(queue,))
        process.start()
        process.join(timeout)

        if process.is_alive():
            print(
                f"[TIMEOUT] LLM call timed out after {timeout} seconds. Terminating..."
            )
            process.terminate()
            process.join()
            raise TimeoutError(f"LLM call timed out after {timeout} seconds.")

        if not queue.empty():
            response = queue.get()
            if isinstance(response, Exception):
                raise response
            return response

        raise TimeoutError("LLM call did not return a response.")

    def get_model_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "provider": "LangChain Ollama",
            "model": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

    def generate_text_with_metadata(
        self, prompt: Union[str, List[str]], timeout: int = 30
    ) -> LLMResponse:
        """Generate text with metadata (estimated token usage)."""
        # Use the main generate_text method which handles timeout properly
        response = self.generate_text(prompt, timeout)

        # Estimate token usage based on character count
        if isinstance(prompt, list):
            total_estimated_input = sum(len(p) // 4 for p in prompt)
            total_estimated_output = (
                sum(len(r) // 4 for r in response) if isinstance(response, list) else 0
            )
        else:
            total_estimated_input = len(prompt) // 4
            total_estimated_output = (
                len(response) // 4 if isinstance(response, str) else 0
            )

        token_usage = TokenUsage(
            input_tokens=total_estimated_input,
            output_tokens=total_estimated_output,
            total_tokens=total_estimated_input + total_estimated_output,
        )

        model_info = self.get_model_metadata()
        return LLMResponse(
            content=response, token_usage=token_usage, model_info=model_info
        )

    def _ensure_model_available(self):
        """Ensure the model is available, attempt to pull if not."""
        available_models = self._get_available_models()
        if self.model_name not in available_models:
            print(f"Model '{self.model_name}' not found. Attempting to pull...")
            self._pull_model()

    def _get_available_models(self) -> List[str]:
        """Get list of available models using Ollama CLI."""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if result.returncode != 0:
                return []
            return [
                line.split()[0]
                for line in result.stdout.splitlines()[1:]
                if line.strip()
            ]
        except Exception:
            return []

    def _pull_model(self):
        """Pull the model from Ollama registry."""
        try:
            result = subprocess.run(
                ["ollama", "pull", self.model_name], capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"Failed to pull model: {result.stderr}")
        except Exception as e:
            print(f"Failed to pull model: {e}")


def create_ollama_model(
    model_name: str = "llama3.2:3b", temperature: float = 0.7, **kwargs
) -> OllamaModel:
    """Convenience function to create an Ollama model instance.

    Args:
        model_name: Name of the Ollama model
        temperature: Temperature for text generation
        **kwargs: Additional parameters passed to OllamaModel constructor

    Returns:
        Configured OllamaModel instance
    """
    return OllamaModel(model_name=model_name, temperature=temperature, **kwargs)
