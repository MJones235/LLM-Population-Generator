# Getting Started with LLM Population Generator

## Installation

1. **Clone or create the project:**
   ```bash
   git clone <your-repo-url>
   cd LLMPopulationGenerator
   ```

2. **Set up Python environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the package:**
   ```bash
   pip install -e .
   ```

## Quick Start

### 1. Basic Usage

```python
from population_generator import PopulationGenerator
from population_generator.llm import OpenAIModel  # You'll implement this
from population_generator.classifiers import UKHouseholdCompositionClassifier

# Initialize generator
generator = PopulationGenerator(
    data_path="./data",
    prompts_path="./prompts"
)

# Configure LLM (implement your own or use OpenAI example)
llm = OpenAIModel(api_key="your-key", model_name="gpt-4o")

# Generate households
households = generator.generate_households(
    n_households=100,
    model=llm,
    base_prompt="Your prompt template...",
    schema={"type": "array", ...},  # JSON schema
    location="London",
    region="UK"
)
```

### 2. Project Structure

Your data directory should look like:
```
data/
├── census/
│   └── london/
│       ├── household_size.csv
│       ├── household_composition.csv
│       └── age_group.csv
├── schemas/
│   └── household_schema.json
```

### 3. Implementing LLM Interfaces

Create your own LLM implementation:

```python
from population_generator.llm import BaseLLM

class YourLLM(BaseLLM):
    def generate_text(self, prompt, timeout=30):
        # Your LLM API call here
        return "Generated response"
    
    def get_model_metadata(self):
        return {"model": "your-model"}
```

### 4. Custom Classifiers

Create region-specific classifiers:

```python
from population_generator.classifiers.household_type.base import HouseholdCompositionClassifier

class YourRegionClassifier(HouseholdCompositionClassifier):
    def get_name(self):
        return "your_region"
    
    def classify_household_structure(self, group, relationship_col="relationship"):
        # Your classification logic
        return "household_type"
```

## Data Preparation

### Census Data Format

**household_size.csv:**
```csv
Size,Count
1,25000
2,35000
3,20000
4,15000
5+,5000
```

**household_composition.csv:**
```csv
Type,Count
One-person,25000
Couple,30000
Couple with children,35000
Single parent,10000
```

### Prompt Templates

Create prompt files in your prompts directory:

```text
Generate a household for {LOCATION} with {NUM_PEOPLE}.

Requirements:
- Realistic ages and relationships
- Appropriate names for the location
- Return as JSON array

[Additional instructions...]
```

### JSON Schema

Define your household schema:

```json
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "name": {"type": "string"},
      "age": {"type": "integer"},
      "gender": {"type": "string"},
      "relationship": {"type": "string"}
    },
    "required": ["name", "age", "gender", "relationship"]
  }
}
```

## Advanced Features

### Statistical Feedback
The generator can adjust prompts based on generated vs. target distributions:

```python
households = generator.generate_households(
    # ... other params ...
    include_stats=True,           # Enable statistical feedback
    include_guidance=True,        # Include guidance text
    hh_type_classifier=YourClassifier()
)
```

### Batch Processing
Generate large populations efficiently:

```python
households = generator.generate_households(
    n_households=10000,
    batch_size=50,  # Process in batches
    # ... other params ...
)
```

## Examples

Check the `examples/` directory for:

- `basic_usage.py` - Simple example with mock LLM
- `openai_example.py` - Complete OpenAI integration
- `custom_classifier.py` - Creating custom classifiers

## Configuration

Use config files for complex setups:

```json
{
  "data": {
    "census_data_dir": "/path/to/census",
    "prompts_dir": "/path/to/prompts"
  },
  "generation": {
    "default_batch_size": 20,
    "default_timeout": 120
  }
}
```

## Testing

Run the installation test:
```bash
python tests/test_installation.py
```

## Next Steps

1. Implement your LLM interface (OpenAI, Azure, Ollama, etc.)
2. Prepare your census data files
3. Create location-specific prompt templates
4. Define your JSON schemas
5. Optionally create custom classifiers for your region
6. Start generating populations!

For more examples and advanced usage, see the `examples/` directory.
