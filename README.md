# LLM Population Generator

A Python package for generating synthetic population data using Large Language Models (LLMs). This library provides a flexible, extensible framework for creating realistic household and demographic data that matches target statistical distributions.

## Features

- **Multi-LLM Support**: Works with Azure OpenAI (primary), Ollama, and other LLM providers
- **Progressive Saving**: Automatic checkpointing for large populations to prevent data loss
- **Failure Tracking & Research Analytics**: Comprehensive failure tracking for academic research on LLM reliability
- **Customizable Classifiers**: Extensible household size and composition classification systems
- **Per-Classifier Format Control**: Configure how each statistic appears in prompts (comparison, observed-only, or target-only)
- **Flexible Data Sources**: Support for custom census data and statistical targets
- **Batch Processing**: Efficient generation of large populations with configurable batch sizes
- **Statistical Feedback**: Dynamic prompt adjustment based on generated vs. target distributions
- **Multiple Regions**: Pre-built classifiers for UK Census, UN Global, and custom regions

## Installation

```bash
pip install llm-population-generator
```

## Quick Start

For a complete demonstration of all features, see the [comprehensive example](examples/comprehensive_example.py):

```bash
cd examples
python comprehensive_example.py
```

**Basic Usage:**

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
    model_name="gpt-4o-mini"  # Your deployment name in Azure
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

## Saving Generated Data

The library provides comprehensive data export functionality to save generated population data with detailed metadata for future analysis:

```python
# Generate households
households = generator.generate_households(
    n_households=100,
    model=llm,
    base_prompt=prompt,
    schema=schema,
    location="London"
)

# Save with comprehensive metadata
saved_files = generator.save_population_data(
    households=households,
    model_info={"name": llm.model_name, "version": "gpt-4o"},
    generation_parameters={
        "n_households": 100,
        "location": "London",
        "batch_size": 10
    },
    output_dir="./outputs",
    output_name="london_population_2024",
    include_analysis=True,
    format_type="json_and_csv"  # Options: "json", "csv", "json_and_csv"
)

# Load previously saved data
from population_generator.utils.data_export import PopulationDataSaver
saver = PopulationDataSaver()
loaded_data = saver.load_population_data("outputs/london_population_2024.json")
```

**Saved data includes:**
- Generated household/population data
- Generation metadata (timestamps, model info, parameters)
- Statistical analysis and target comparisons
- **Failure analysis and reliability metrics for academic research**
- Cost tracking information (if enabled)
- Data provenance and source information
- Unique run identifiers for reproducibility
- **Detailed failure records for research into LLM reliability patterns**

## Progressive Saving for Large Populations

For large population generation tasks that may take hours or days, the library provides progressive saving to prevent data loss:

```python
# Generate with automatic checkpointing - checkpoints automatically saved to output_dir/checkpoints
households = generator.generate_households(
    n_households=100000,  # Large population
    model=llm,
    base_prompt=prompt,
    schema=schema,
    location="United Kingdom",
    batch_size=50,
    enable_progressive_saving=True,        # Enable automatic saving
    output_dir="./outputs/uk_project",     # Checkpoints auto-saved to output_dir/checkpoints
    checkpoint_name="uk_population_100k"  # Name for this generation
)

# Resume from checkpoint after interruption
households = generator.generate_households(
    # ... same parameters ...
    enable_progressive_saving=True,
    output_dir="./outputs/uk_project",     # Same output directory
    checkpoint_name="uk_population_100k",
    resume_from_checkpoint=True  # Resume where you left off
)

# Manage checkpoints (checkpoints automatically organized by project)
checkpoints = generator.list_checkpoints("./outputs/uk_project/checkpoints")
for checkpoint in checkpoints:
    print(f"{checkpoint['name']}: {checkpoint['progress']} ({checkpoint['progress_percent']:.1f}%)")
```

**Progressive Saving Features:**
- Automatic checkpointing after each batch
- Resume from any checkpoint after interruption
- Progress tracking and monitoring
- Multiple checkpoint management
- Data safety - never lose more than one batch of work
- **Automatic organization**: Checkpoints saved to `output_dir/checkpoints` when `output_dir` is provided
- **Clean API**: No manual checkpoint directory management required

**Recommended Directory Structure** (automatically created):
```
outputs/
└── your_project/
    ├── checkpoints/           # Progressive saving checkpoints (auto-created)
    │   └── generation_*.json
    ├── final_data.csv        # Final generated population
    └── metadata.json         # Generation metadata
```

See `PROGRESSIVE_SAVING_GUIDE.md` for detailed documentation and examples.

## Project Structure

```
population_generator/
├── core/              # Core generation logic
├── classifiers/       # Household classification systems
├── llm/              # LLM interface implementations
├── utils/            # Utility functions and data loaders
│   ├── failure_tracking.py  # Academic failure analysis system
│   └── data_export.py       # Enhanced metadata export
└── examples/         # Usage examples and sample data
    └── failure_tracking_example.py  # Academic research example
```

## Academic Research Features

This library includes comprehensive failure tracking designed for academic research:

```python
# Automatic failure tracking during generation
households = generator.generate_households(...)

# Get research-ready failure statistics
failure_stats = llm.get_failure_statistics()
print(f"Success rate: {failure_stats['generation_success_metrics']['success_rate']:.1%}")

# Save with failure analysis for research
saved_files = generator.save_population_data(
    households=households,
    model_info=model_info,
    generation_parameters=params,
    output_dir="./research_data",
    output_name="study_population",
    llm_model=llm  # Include for failure analysis
)
```

**Research Applications:**
- Schema complexity vs. failure rate analysis
- Model reliability and consistency studies  
- Cost-effectiveness of retry strategies
- Prompt engineering effectiveness research
- Comparative analysis between different LLMs

See `FAILURE_TRACKING_GUIDE.md` for comprehensive documentation.

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
