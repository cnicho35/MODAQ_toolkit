# MODAQ Toolkit

A Python-based data conversion library that transforms MODAQ MCAP files into columnar time series
data using [Apache Parquet](https://parquet.apache.org/). Parquet offers an optimal balance of type
safety, storage efficiency, and cross-language support, making it more efficient than
CSV and easier to work with than raw MODAQ output.

### Why Parquet?

- Database-like features including partitioning and filtering
- Schema enforcement with strict typing
- 10-20x smaller file sizes compared to CSV
- Wide language support through official and community libraries:
  - Python:
    - [pandas](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html)
    - [Polars](https://docs.pola.rs/api/python/stable/reference/index.html)
  - MATLAB: [Built-in support](https://www.mathworks.com/help/matlab/ref/matlab.io.datastore.parquetdatastore.html)
  - C++: [Arrow](https://arrow.apache.org/docs/cpp/parquet.html)
  - Rust: [Polars](https://docs.rs/polars/latest/polars/)

## Quick Start

Quick install from GitHub:

```bash
pip install git+https://github.nrel.gov/Water-Power/modaq_toolkit.git
```

For detailed installation options including conda environments, specific versions, and development setup, see the [Installation section](#installation) below.

### Usage

#### Command Line

```bash
modaq -i /path/to/mcap/files -o /path/to/output
```

#### Python

```python
# Convert multiple MCAP files
from modaq_toolkit import process_mcap_files
process_mcap_files("input_directory", "output_directory")

# Or work with a single file
from modaq_toolkit import MCAPParser
parser = MCAPParser("path/to/file.mcap")
parser.read_mcap()
parser.create_output("output_directory")
```

## Output Structure

The toolkit processes data in two stages:

1. `a1_one_to_one/`: Preserves the original data structure

   - Contains Parquet files organized by topic
   - Includes metadata describing schemas and structure

2. `a2_real_data/`: Optimizes data for time series analysis
   - Expands array data for easier analysis
   - Maintains same file organization as stage 1

## Using Parquet Data

### Python with pandas

Read all Parquet files from a directory:

```python
from pathlib import Path

import pandas as pd

# Read all parquet files in a directory
# Path uses forward slash on Windows, MacOS and Linux
directory = Path("path/to/parquet/files").resolve()

df = pd.read_parquet(directory)

# Ensure that the data is sorted by time
df = df.sort_index()

# Print a summary of the df
print(df.info())

# Print the first 5 rows of df
print(df.head())
```

Read a single Parquet file:

```python
import pandas as pd

parquet_file = Path("path/to/file.parquet").resolve()
# Read single parquet file and sort by time
df = pd.read_parquet(parquet_file)
df = df.sort_values('time')

# Print a summary of the df
print(df.info())

# Print the first 5 rows of df
print(df.head())
```

### MATLAB

Read all Parquet files from a directory:

```matlab
% Create a datastore for the directory
ds = parquetDatastore("path/to/parquet/files", "OutputType", "timetable");

% Preview the dataset
preview(ds)

% Read all data and convert to table
df = readall(ds);

% Sort by time
df = sortrows(df, 'time');
```

Read a single Parquet file:

```matlab
% Read single parquet file
df = parquetread("path/to/file.parquet");
```

Note: MODAQ data always includes a `time` column which should be used for sorting and time series analysis.

## Installation

### Prerequisites

#### Python Requirements

- Python 3.10 or higher is required
- We recommend using Anaconda or Miniconda for environment management

#### Installing Python/Anaconda

**For NREL Enterprise Users:**

1. Open "Portal Manager" on your NREL workstation
2. Browse to the Anaconda package
3. Click "Install" to get the full Anaconda distribution with Python
   [A screenshot demonstrating these steps is available in the repository's documentation]

**For Non-NREL Users:**
Download and install either:

- [Anaconda](https://www.anaconda.com/download) (full distribution with many pre-installed packages)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (minimal distribution, recommended for most users)

Create and activate a new environment:

```bash
# Create new environment with Python 3.10
conda create -n modaq python=3.10

# Activate the environment
# On Windows:
conda activate modaq
# On Unix-like systems:
source activate modaq
```

### Clone the Repository

Choose one of the following methods to clone the repository:

#### HTTPS (Recommended for most users)

```bash
git clone https://github.nrel.gov/Water-Power/modaq_toolkit
```

#### SSH (For contributors)

First, ensure you have [configured SSH access for GitHub Enterprise](https://docs.github.com/en/enterprise-cloud@latest/authentication/connecting-to-github-with-ssh). Then:

```bash
git clone git@github.nrel.gov:Water-Power/modaq_toolkit.git
```

#### GitHub Desktop

1. Open GitHub Desktop
2. Go to File > Clone Repository
3. Select the "GitHub Enterprise" tab
4. Search for "modaq_toolkit"
5. Choose your local path
6. Click "Clone"

### Navigate to Repository Directory

Choose the appropriate command for your system:

Windows (Command Prompt):

```cmd
cd C:\path\to\modaq_toolkit
```

Windows (PowerShell):

```powershell
Set-Location -Path C:\path\to\modaq_toolkit
```

Unix-like systems (Linux, macOS):

```bash
cd /path/to/modaq_toolkit
```

### Install the Package

#### Option 1: Install from local directory

Once in the correct directory, install the package:

```bash
pip install -e .
```

#### Option 2: Install directly from GitHub

Install the latest version from the main branch:

```bash
pip install git+https://github.nrel.gov/Water-Power/modaq_toolkit.git
```

Or install a specific branch, tag, or commit:

```bash
# Install from a branch
pip install git+https://github.nrel.gov/Water-Power/modaq_toolkit.git@branch-name

# Install from a tag
pip install git+https://github.nrel.gov/Water-Power/modaq_toolkit.git@v1.0.0

# Install from a specific commit
pip install git+https://github.nrel.gov/Water-Power/modaq_toolkit.git@commit-hash
```

For SSH users:

```bash
pip install git+ssh://git@github.nrel.gov/Water-Power/modaq_toolkit.git
```

> **Note:** The `-e` flag performs an "editable" install, which means:
>
> - The package is installed in "development mode"
> - Changes to the source code take effect immediately without reinstalling
> - The actual package code stays in its current location and is just referenced by Python
> - The `.` tells pip to install the package from the current directory using the configuration in `pyproject.toml`

For development (includes additional dependencies for testing and development):

```bash
pip install -e ".[dev]"
```

Note: Make sure you're in the directory containing the `pyproject.toml` file before running the pip install commands.

```bash
ruff check .
ruff format .
```

## Common Issues and Solutions

### Installation Problems

1. **Missing Dependencies**

   - Make sure you're using Python 3.10+
   - Check that all requirements are installed: `pip list`
   - Try reinstalling in a fresh conda environment

2. **Import Errors**
   - Verify your environment is activated
   - Ensure you're in the correct directory when installing
   - Check for any conflicting packages

### Data Processing

1. **Large Files**

   - Process files individually for better memory management
   - Use the command line tool for batch processing
   - Monitor system resources during conversion
   - Avoid processing files directly from OneDrive, network drives, or remote servers
     - These files must be downloaded locally first, which can significantly slow processing
     - Copy files to a local drive before processing for best performance

2. **Output Structure**
   - Check permissions in output directory
   - Verify input MCAP file integrity
   - Review logs for processing errors

## Support

For issues and feature requests, please use the GitHub issue tracker. Include:

- Your operating system and Python version
- Steps to reproduce the problem
- Any relevant error messages
- Example data (if possible)
