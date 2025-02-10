"""
MODAQ Toolkit: High-performance MODAQ to Parquet converter and analysis toolkit
"""

__version__ = "0.1.0"

from .message_processing import (
    MessageProcessor,
    expand_array_columns_vertically,
    is_array_column,
    parse_ros_message_definition,
)
from .parser import MCAPParser, process_mcap_files

__all__ = [
    "MessageProcessor",
    "MCAPParser",
    "parse_ros_message_definition",
    "expand_array_columns_vertically",
    "is_array_column",
    "process_mcap_files",
]
