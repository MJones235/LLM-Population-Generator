# LLM Population Generator

A Python package for generating synthetic household/population data with Large Language Models (LLMs), including statistical feedback loops, progressive checkpointing, and export metadata.

## Features

- Multi-LLM support (`OpenAIModel`, `FoundryModel`, `OllamaModel`)
- Extensible demographic classifiers (UK, UNPD, and custom)
- Statistical prompt feedback via placeholder replacement
- Progressive saving and resume-from-checkpoint workflows
- Cost tracking and failure analysis support
- Structured CSV + metadata export for downstream analysis

## Installation

From source (recommended for this repository):

```bash
git clone git@github.com:MJones235/LLM-Population-Generator.git
cd LLM-Population-Generator
pip install -r requirements.txt
pip install -e .
```

## Quick Start

Minimal household generation flow:

```python
from population_generator import PopulationGenerator
from population_generator.llm import OpenAIModel

generator = PopulationGenerator(
    data_path="./examples/data",
    prompts_path="./examples/prompts"
)

prompt = generator.prompt_manager.load_prompt("basic_household.txt")
schema = generator.data_loader.load_schema("household_basic.json")

llm = OpenAIModel(
    api_key="your-azure-openai-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    model_name="gpt-4o-mini"
)

households = generator.generate_households(
    n_households=100,
    model=llm,
    base_prompt=prompt,
    schema=schema,
    location="London",
    batch_size=10
)
```

Run example scripts from the repository root:

```bash
python examples/comprehensive_example.py
python examples/llama3_simple_example.py
python examples/uk_census_preprocessing_example.py
```

## Registering Demographic Classifiers

```python
from population_generator.contrib.classifiers.uk import (
    UKHouseholdSizeClassifier,
    UKHouseholdCompositionClassifier,
    UKAgeClassifier,
    UKSexClassifier,
)

generator.prompt_manager.register_classifier(
    "HOUSEHOLD_SIZE_STATS",
    UKHouseholdSizeClassifier(),
    target_file="targets/household_size.csv",
    format_type="comparison"
)
```

## Progressive Saving and Resume

```python
households = generator.generate_households(
    n_households=5000,
    model=llm,
    base_prompt=prompt,
    schema=schema,
    location="United Kingdom",
    batch_size=50,
    enable_progressive_saving=True,
    output_dir="./outputs/uk_run",
    checkpoint_name="uk_population"
)

checkpoints = generator.list_checkpoints("./outputs/uk_run/checkpoints")
```

When `output_dir` is provided and progressive saving is enabled, checkpoints default to `output_dir/checkpoints`.

## Saving Results

```python
saved_files = generator.save_population_data(
    households=households,
    model_info={"name": llm.model_name},
    generation_parameters={"n_households": 100, "location": "London"},
    output_dir="./outputs/comprehensive_example",
    output_name="uk_population_comprehensive",
    llm_model=llm
)

print(saved_files)  # {'csv': Path(...), 'metadata': Path(...)}
```

For direct exporter usage:

```python
from population_generator.data.export import PopulationDataSaver

saver = PopulationDataSaver("./outputs")
saved = saver.save_population_data(households, output_name="population")
```

## Project Structure

```text
population_generator/
├── analysis/
├── classifiers/
├── contrib/
│   ├── classifiers/
│   └── data_preprocessors/
├── data/
│   └── export/
├── engine/
├── generation/
└── llm/
```

## Development

```bash
git clone git@github.com:MJones235/LLM-Population-Generator.git
cd LLM-Population-Generator
pip install -r requirements.txt
pip install -e .
```

## License

MIT License — see `LICENSE`.
