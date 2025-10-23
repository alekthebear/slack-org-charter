# Org Chart Generator

This project generates organizational charts from Slack workspace data by analyzing user interactions, channel memberships, and other features to infer organizational structure.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Evaluation](#evaluation)
- [Running Tests](#running-tests)

## Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

### 1. Install uv

If you don't have uv installed, install it:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

### 2. Create Virtual Environment and Install Dependencies

```bash
# Navigate to project directory
cd /path/to/gather

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

The `uv sync` command will:
- Create a virtual environment (`.venv/`)
- Install all dependencies from `pyproject.toml`
- Lock dependencies in `uv.lock` for reproducibility

### 3. Install Development Dependencies

Development dependencies (jupyter, pytest, ruff, etc.) are installed automatically with `uv sync`. If you need to install them separately:

```bash
uv sync --group dev
```

## Configuration

Before running the pipeline, you need to configure the data paths in `src/config.py`. The two most important parameters are:

### Required Configuration Parameters

#### `RAW_DATA_ROOT`
**Purpose**: Points to the directory containing raw Slack workspace data files.

**Expected Structure**:
```
RAW_DATA_ROOT/
  ├── users.json          # User profiles and metadata
  ├── channels.json       # Channel information
  └── [channel_id]/       # One directory per channel containing message history
      └── *.json          # Message files
```

**How to Configure**:
```python
# Option 1: Edit src/config.py directly
RAW_DATA_ROOT = "/path/to/your/slack/data/raw"

# Option 2: Set via environment variable (recommended)
export RAW_DATA_ROOT="/path/to/your/slack/data/raw"
```

#### `CACHE_ROOT`
**Purpose**: Specifies where intermediate and processed data should be cached. The pipeline generates several types of cached data to avoid redundant computation:

**How to Configure**:
```python
# Option 1: Edit src/config.py directly
CACHE_ROOT = "/path/to/cache/directory"

# Option 2: Set via environment variable (recommended)
export CACHE_ROOT="/path/to/cache/directory"
```

#### API Keys for LLM Models

**Purpose**: The pipeline uses [LiteLLM](https://github.com/BerriAI/litellm) to interface with language models, which automatically reads the appropriate API key environment variables based on the model provider.

**Required Environment Variables**:

The specific environment variable(s) needed depend on which model provider you're using:

- **OpenAI** models (e.g., `openai/gpt-4`, `openai/gpt-4o`): Requires `OPENAI_API_KEY`
- **Anthropic** models (e.g., `anthropic/claude-3-5-sonnet-20241022`): Requires `ANTHROPIC_API_KEY`
- **Google** models (e.g., `gemini/gemini-pro`): Requires `GEMINI_API_KEY`
- **Other providers**: See [LiteLLM documentation](https://docs.litellm.ai/docs/providers) for required keys

**How to Configure**:

```bash
# For OpenAI (default configuration)
export OPENAI_API_KEY="your-openai-api-key-here"

# For other providers, set the appropriate environment variable
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
```

**Models Used**:
- `DEFAULT_MODEL` (default: `"openai/gpt-5"`): Used for general inference tasks like role identification and manager inference
- `WEB_SEARCH_MODEL` (default: `"openai/gpt-5-search-api"`): Used for web search capabilities when enriching user context

**Note**: If you change `DEFAULT_MODEL` or `WEB_SEARCH_MODEL` in `src/config.py` to use a different provider, ensure you have the corresponding API key environment variable set.

### Other Configuration Parameters

- **`COMPANY_NAME`**: Company name for context in LLM prompts (default: "Gather Town")
- **`RECENCY_CUT_OFF_DATE`**: Messages before this date are excluded from analysis (default: 2025-09-01)
- **`MAX_CONCURRENT_WORKERS`**: Number of concurrent threads for API calls (default: 10)
- **`DEFAULT_MODEL`**: LLM model for inference tasks (default: "openai/gpt-5")
- **`WEB_SEARCH_MODEL`**: Model with web search capability (default: "openai/gpt-5-search-api")

## Running the Pipeline

The main pipeline script is `src/run_pipeline.py`. It orchestrates the following steps:

1. **Data Extraction**: Parse raw Slack JSON files and normalize message data
2. **Feature Extraction**: Extract various features that are fed into the inference stage
3. **Inference**: infer user job roles and manager-employee relationships based on the features
4. **Org Chart Generation**: Construct and export organizational hierarchy based on the inferred data

### Basic Usage

```bash
# Run the full pipeline
python src/run_pipeline.py -o output/org_chart.md
```

### With Cache Refresh

By default, the pipeline uses cached data when available. To force recomputation of all steps:

```bash
python src/run_pipeline.py -o output/org_chart.md --force-refresh
```

**Note**: `--force-refresh` will:
- Re-extract all messages from raw data
- Re-run all feature extraction (including LLM calls)
- Re-run all inference steps

### Command-Line Options

- `-o, --org-chart-output` (required): Path where the org chart markdown file will be saved
- `--force-refresh`: Skip cache and recompute all steps

## Evaluation

To evaluate the quality of your generated org chart against a ground truth file:

### Usage

```bash
# Default: Pretty-printed results to terminal
python src/evaluate.py --pred output/org_chart.md --true path/to/ground_truth.md

# JSON output (useful for programmatic processing)
python src/evaluate.py --pred output/org_chart.md --true path/to/ground_truth.md --json
```

### Command-Line Options

- `--pred` (required): Path to the predicted/generated org chart markdown file
- `--true` (required): Path to the ground truth org chart markdown file
- `--json`: Print results as JSON instead of pretty-printed format (optional)

### What It Evaluates

The evaluation script performs 2 types of analysis:

1. **Coverage Metrics**: 
   - How many ground truth employees were found in predictions
   - Extra employees in predictions not in ground truth
   - Missing employees from ground truth

2. **Manager Relationship Accuracy**:
   - Percentage of correctly identified reporting relationships
   - Breakdown of error types (wrong manager, missing manager, etc.)
   - Detailed error listings for debugging

### Example Output

```
================================================================================
                      ORG CHART EVALUATION RESULTS                      
================================================================================

📊 NAME MATCHING
--------------------------------------------------------------------------------
  Ground Truth Names:  150
  Predicted Names:     145
  Matched:             142 (94.7%)
  Unmatched (GT):      8
  Unmatched (Pred):    3

================================================================================
📈 COVERAGE METRICS
--------------------------------------------------------------------------------
  Employee Coverage: 94.7% (142/150)

================================================================================
👔 MANAGER RELATIONSHIP ACCURACY
--------------------------------------------------------------------------------
  Accuracy: 87.3% (124/142)
  Correct:  124
  Errors:   18

================================================================================
✅ OVERALL SUMMARY
--------------------------------------------------------------------------------
  Name Matching:       94.7%
  Manager Accuracy:    87.3%
```

## Running Tests

This project uses pytest for testing. Tests are located in the `tests/` directory.

### Run All Tests

```bash
# Using pytest directly
pytest

# With verbose output
pytest -v

# With coverage report
pytest --cov=src
```

### Run Specific Test Files

```bash
# Test org chart generation
pytest tests/orgchart/test_generate.py

# Test org chart model
pytest tests/orgchart/test_model.py

# Test utilities
pytest tests/test_utils.py
```

### Run Specific Test Functions

```bash
pytest tests/orgchart/test_model.py::test_function_name -v
```


