"""
Handles exporting the final dataset to various formats.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable

import structlog

try:
    import pyarrow as pa  # type: ignore[import-not-found]
    import pyarrow.parquet as pq  # type: ignore[import-not-found]
    from datasets import Dataset  # type: ignore[import-not-found]

    HAS_ARROW = True
    HAS_DATASETS = True
except ImportError:
    pa = None
    pq = None
    Dataset = None
    HAS_ARROW = False
    HAS_DATASETS = False

from quarrycore.config.config import ExportConfig

logger = structlog.get_logger(__name__)


class BaseExporter(ABC):
    """Abstract base class for all dataset exporters."""

    def __init__(self, config: ExportConfig):
        self.config = config

    @abstractmethod
    def export(self, data: Iterable[Dict[str, Any]], output_path: Path) -> None:
        """Exports the dataset to the specified path."""
        pass


class JsonlExporter(BaseExporter):
    """Exports data to a streaming JSONL file."""

    def export(self, data: Iterable[Dict[str, Any]], output_path: Path) -> None:
        logger.info("Exporting to JSONL", path=str(output_path))
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        logger.info("JSONL export complete", path=str(output_path))


class ParquetExporter(BaseExporter):
    """Exports data to a sharded Parquet dataset."""

    def export(self, data: Iterable[Dict[str, Any]], output_path: Path) -> None:
        logger.info("Exporting to Parquet", path=str(output_path))
        output_path.mkdir(exist_ok=True)
        # We process in batches to manage memory
        writer = None
        for item in data:
            # For simplicity, we'll write one file. Sharding would be more complex.
            # In a real scenario, we'd batch ~1000s of items before writing.
            if writer is None:
                table = pa.Table.from_pylist([item])
                writer = pq.ParquetWriter(output_path / "data.parquet", table.schema)

            table = pa.Table.from_pylist([item])
            writer.write_table(table)

        if writer:
            writer.close()
        logger.info("Parquet export complete", path=str(output_path))


class HuggingFaceExporter(BaseExporter):
    """Exports data to a HuggingFace Dataset and optionally pushes to Hub."""

    def export(self, data: Iterable[Dict[str, Any]], output_path: Path) -> None:
        logger.info("Creating HuggingFace dataset", path=str(output_path))

        # The 'datasets' library is very efficient at handling this
        dataset = Dataset.from_generator(lambda: (yield from data))

        dataset.save_to_disk(str(output_path))
        logger.info("HuggingFace dataset saved to disk", path=str(output_path))

        if self.config.huggingface_repo_id:
            logger.info("Pushing dataset to HuggingFace Hub", repo_id=self.config.huggingface_repo_id)
            # This requires the user to be logged in via `huggingface-cli login`
            dataset.push_to_hub(self.config.huggingface_repo_id)
            logger.info("Push to Hub complete", repo_id=self.config.huggingface_repo_id)


def get_exporter(format_name: str, config: ExportConfig) -> BaseExporter:
    """Factory function to get the appropriate exporter."""
    if format_name == "jsonl":
        return JsonlExporter(config)
    elif format_name == "parquet":
        return ParquetExporter(config)
    elif format_name == "huggingface":
        return HuggingFaceExporter(config)
    else:
        raise ValueError(f"Unknown exporter format: {format_name}")
