# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-02-13

### Added

- Asynchronous processing support via `--async` flag for improved performance
- Timing summaries for better processing feedback
- Enhanced nested folder handling in output structure
- Time column to a2 data for non-nested datasets

### Changed

- Improved documentation:
  - Clearer installation instructions
  - Better separation of CLI and Python usage examples
  - Added CLI argument documentation

### Fixed

- Missing time column in a2 data without nested structures

## [0.1.0] - 2025-02-10

### Added

- Initial release of MODAQ Toolkit
- Basic MCAP to Parquet conversion
- Two-stage output processing:
  - Stage 1 (a1_one_to_one): Preserves original data structure
  - Stage 2 (a2_real_data): Optimizes for time series analysis
- Command line interface
- Python API
