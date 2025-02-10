import json
import logging
import pathlib

import pandas as pd
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from .message_processing import (
    MessageProcessor,
    expand_array_columns_vertically,
    is_array_column,
    parse_ros_message_definition,
)

logger = logging.getLogger(__name__)


class MCAPParser:
    def __init__(self, mcap_path: pathlib.Path):
        self.mcap_path = mcap_path
        self.processors: dict[str, MessageProcessor] = {}
        self.schemas_by_topic: dict[str, dict] = {}
        self.dataframes: dict[str, pd.DataFrame] = {}

    def read_mcap(self):
        logger.info(f"Reading {self.mcap_path}")
        with open(self.mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            summary = reader.get_summary()
            if summary is None:
                raise ValueError("Could not read summary from MCAP file")

            for channel in summary.channels.values():
                schema = summary.schemas[channel.schema_id]
                self.schemas_by_topic[channel.topic] = parse_ros_message_definition(
                    schema.data
                )
                self.processors[channel.topic] = MessageProcessor(
                    self.schemas_by_topic[channel.topic]
                )

            for schema, channel, message, decoded in reader.iter_decoded_messages():
                if decoded is None:
                    logger.warning(
                        f"Could not decode message for topic {channel.topic}"
                    )
                    continue

                try:
                    self.processors[channel.topic].process_message(decoded)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to process message for topic {channel.topic}: {e}"
                    )

            for topic, processor in self.processors.items():
                self.dataframes[topic] = processor.get_dataframe()
                logger.info(
                    f"Topic {topic} DataFrame shape: {self.dataframes[topic].shape}"
                )

    def create_output(self, output_dir: pathlib.Path, stage: str = "a1_one_to_one"):
        """Create partitioned parquet files and metadata JSON for each topic"""
        stage_dir = output_dir / stage
        metadata_dir = output_dir / "metadata"
        stage_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        summary_metadata = {}

        for topic, df in self.dataframes.items():
            safe_topic = topic.replace("/", "_").lstrip("_")

            if df.empty:
                continue

            topic_dir = stage_dir / f"channel={safe_topic}"
            topic_dir.mkdir(exist_ok=True)

            if stage == "a2_real_data":
                object_columns = [col for col in df.columns if is_array_column(df[col])]

                if object_columns:
                    logger.info(
                        f"Found arrays in topic {topic}, expanding for a2 processing"
                    )
                    df = expand_array_columns_vertically(df)
                    df["time"] = pd.to_datetime(
                        df["system_time"], origin="unix", unit="ns", utc=True
                    )
                    df = df.set_index("time")
                    df = df.drop(
                        ["sec", "nanosec", "frame_id", "timestamp"], axis="columns"
                    )

                    time_diffs = df.index.to_series().diff().dt.total_seconds()
                    median_interval = time_diffs.mean()
                    print(f"Sample rate: {1/median_interval:.3f} Hz")
                    print(
                        f"\nMode-based sample rate: {1/time_diffs.value_counts().index[0]:.3f} Hz"
                    )
                else:
                    df["time"] = pd.to_datetime(
                        df["timestamp"], origin="unix", unit="s", utc=True
                    )
                    df = df.set_index("time")
                    logger.info(
                        f"No arrays found in topic {topic}, copying original data to a2"
                    )

            base_filename = f"{self.mcap_path.stem}.{safe_topic}"
            parquet_filename = f"{base_filename}.parquet"
            output_path = topic_dir / parquet_filename
            df.to_parquet(output_path)
            logger.info(f"Saved {output_path}")

            if stage == "a1_one_to_one":
                topic_metadata = {
                    "topic": topic,
                    "source_file": str(self.mcap_path),
                    "n_messages": len(df),
                    "n_columns": len(df.columns),
                    "columns": df.columns.tolist(),
                    "dtypes": {col: str(df[col].dtype) for col in df.columns},
                    "schema": self.schemas_by_topic[topic],
                    "processing_stage": stage,
                    "parquet_file": str(output_path.relative_to(stage_dir)),
                }

                safe_topic_metadata_name = (
                    f"{self.mcap_path.stem}.{safe_topic}.metadata.json"
                )
                topic_metadata_path = metadata_dir / safe_topic_metadata_name
                with open(topic_metadata_path, "w") as f:
                    json.dump(topic_metadata, f, indent=2)
                logger.info(f"Saved metadata to {topic_metadata_path}")

                summary_metadata[topic] = topic_metadata

        if stage == "a1_one_to_one":
            summary = {
                "source_file": str(self.mcap_path),
                "n_topics": len(self.dataframes),
                "n_topics_with_data": sum(
                    1 for df in self.dataframes.values() if not df.empty
                ),
                "creation_time": pd.Timestamp.now().isoformat(),
                "processing_stage": stage,
                "topics": summary_metadata,
            }

            summary_path = metadata_dir / f"{self.mcap_path.stem}.summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved summary metadata to {summary_path}")


def process_mcap_files(input_dir: str, output_dir: str):
    """Process all MCAP files in a directory with single read and dual output"""
    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)

    metadata_dir = output_path / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (output_path / "a1_one_to_one").mkdir(parents=True, exist_ok=True)
    (output_path / "a2_real_data").mkdir(parents=True, exist_ok=True)

    for mcap_file in sorted(input_path.glob("*.mcap")):
        logger.info(f"Processing {mcap_file}")

        parser = MCAPParser(mcap_file)
        parser.read_mcap()
        original_dataframes = parser.dataframes.copy()

        logger.info("Running stage 1: Original data")
        parser.create_output(output_path, stage="a1_one_to_one")

        logger.info("Running stage 2: Expanded arrays")
        for topic, df in original_dataframes.items():
            if not df.empty:
                parser.dataframes[topic] = df
        parser.create_output(output_path, stage="a2_real_data")

        parser.dataframes = original_dataframes
        logger.info(f"Completed processing {mcap_file}")
