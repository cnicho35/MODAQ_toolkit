# MODAQ Toolkit

Convert and analyze MODAQ original data from MCAP files to Parquet format.

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Usage

### Command Line

Convert MCAP files in a directory:

```bash
modaq -i /path/to/mcap/files -o /path/to/output
```

### Python API

```python
from modaq_toolkit import MCAPParser, process_mcap_files

# Process multiple files
process_mcap_files("input_directory", "output_directory")

# Or work with single file
parser = MCAPParser("path/to/file.mcap")
parser.read_mcap()
parser.create_output("output_directory")
```

## Output Structure

The toolkit creates two processing stages:

1. `a1_one_to_one/`: Original data with preserved structure
2. `a2_real_data/`: Expanded array data for time series analysis

Each stage contains:

- Parquet files organized by topic
- Metadata describing schemas and data structure

## Development

Lint code:

```bash
ruff check .
ruff format .
```
