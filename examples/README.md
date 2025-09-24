# Examples

This directory contains example scripts demonstrating key features of the LLM Population Generator library.

## Available Examples

### `comprehensive_example.py` 
The main example showcasing ALL features of the LLM Population Generator. This comprehensive demonstration includes:

**Core Features:**
- OpenAI GPT-4o-mini integration with Azure OpenAI
- UK demographic classifiers with statistical feedback
- Token tracking and cost analysis
- Failure tracking and analysis
- Custom validation rules
- Comprehensive data export with metadata

**What It Demonstrates:**
- Complete end-to-end population generation workflow
- Statistical feedback between generation batches
- Custom validation rules for realistic household compositions
- Detailed cost tracking and analysis
- Failure pattern analysis for research
- Professional data export for reproducibility
- Real-time statistics and monitoring

This example serves as both a complete demonstration and a template for production use.

## Quick Start

**For first-time users, run the comprehensive example:**

```bash
cd examples
python comprehensive_example.py
```

This will demonstrate all features and create sample output in `outputs/comprehensive_example/`.

## Required Setup

1. Create a `.env` file with your Azure OpenAI credentials:
   ```
   AZURE_OPENAI_API_KEY=your_key_here
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_MODEL=gpt-4o-mini
   ```

2. Install dependencies:
   ```bash
   pip install python-dotenv
   ```

## Data Files

### Target Data (`data/targets/`)
- `uk_household_size.csv` - UK household size distribution (1-7 people)
- `uk_household_composition.csv` - Household types (couples, families, etc.)
- `uk_age_distribution.csv` - Age band distribution (0-9, 10-19, etc.)
- `uk_sex_distribution.csv` - Male/Female distribution

### Prompt Templates (`prompts/`)
- `basic_household.txt` - Main prompt template with statistical placeholders

### Schemas (`data/schemas/`)
- JSON schemas defining expected output format for generated households

The statistical placeholders in the prompt (`{HOUSEHOLD_SIZE_STATS}`, `{AGE_STATS}`, etc.) are automatically replaced with current vs target distributions during generation.
