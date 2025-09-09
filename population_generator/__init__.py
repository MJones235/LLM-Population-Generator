"""
LLM Population Generator

A Python package for generating synthetic population data using Large Language Models.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.generator import PopulationGenerator
from .core.config import Config

__all__ = ["PopulationGenerator", "Config"]
