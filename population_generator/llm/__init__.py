"""LLM interface implementations."""

from .base import BaseLLM
from .openai_model import OpenAIModel

__all__ = ["BaseLLM", "OpenAIModel"]
