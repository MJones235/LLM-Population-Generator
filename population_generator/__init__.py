"""
LLM Population Generator

A Python package for generating synthetic population data using Large Language Models.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .engine.generator import PopulationGenerator
from .engine.config import Config
from .data.export import PopulationDataSaver, save_generation_results

__all__ = ["PopulationGenerator", "Config", "PopulationDataSaver", "save_generation_results"]
