# LLM Population Generator

A Python package for generating synthetic population data using Large Language Models (LLMs). This library provides a flexible, extensible framework for creating realistic household and demographic data that matches target statistical distributions.

## Features

- **Multi-LLM Support**: Works with Azure OpenAI (primary), Ollama, and other LLM providers
- **Customizable Classifiers**: Extensible household size and composition classification systems
- **Flexible Data Sources**: Support for custom census data and statistical targets
- **Batch Processing**: Efficient generation of large populations with configurable batch sizes
- **Statistical Feedback**: Dynamic prompt adjustment based on generated vs. target distributions
- **Multiple Regions**: Pre-built classifiers for UK Census, UN Global, and custom regions

## Installation

```bash
pip install llm-population-generator
```

## Quick Start

```python
from population_generator import PopulationGenerator
from population_generator.llm import OpenAIModel
from population_generator.classifiers import UKHouseholdTypeClassifier

# Initialize the generator
generator = PopulationGenerator(
    data_path="./data",
    prompts_path="./prompts"
)

# Configure Azure OpenAI LLM
llm = OpenAIModel(
    api_key="your-azure-openai-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    model_name="gpt-4o"  # Your deployment name in Azure
)

# Generate households
households = generator.generate_households(
    n_households=1000,
    model=llm,
    location="London",
    region="E12000007",
    batch_size=10,
    include_stats=True
)
```

## Project Structure

```
population_generator/
├── core/              # Core generation logic
├── classifiers/       # Household classification systems
├── llm/              # LLM interface implementations
├── utils/            # Utility functions and data loaders
└── examples/         # Usage examples and sample data
```

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-population-generator.git
cd llm-population-generator

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

## License

MIT License - see LICENSE file for details.
