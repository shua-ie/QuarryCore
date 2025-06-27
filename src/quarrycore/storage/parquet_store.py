"""
Manages the Parquet store for warm content storage.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, List
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from quarrycore.config.config import ParquetConfig


class ParquetStore:
    """Handles writing to and reading from the Parquet warm storage tier."""

    def __init__(self, config: ParquetConfig):
        self.config = config
        self.base_path = Path(config.base_path)

    async def write(self, data: List[dict[str, Any]]) -> str:
        """
        Writes a batch of data to a new Parquet file in a partitioned directory.

        Args:
            data: A list of records (dicts) to write.

        Returns:
            The path to the newly created Parquet file, relative to the base path.
        """
        if not data:
            return ""

        table = pa.Table.from_pylist(data)

        # Generate a unique filename and a relative path for partitioning
        partition_path = self._get_partition_path(data[0])
        file_path = partition_path / f"{uuid4()}.parquet"

        full_path = self.base_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Run the synchronous pyarrow write operation in a separate thread
        await asyncio.to_thread(
            pq.write_table,
            table,
            where=str(full_path),
            compression=self.config.compression,
        )

        return str(file_path)

    def _get_partition_path(self, record: dict[str, Any]) -> Path:
        """Constructs the partition path based on the config."""
        parts = []
        for key in self.config.partition_by:
            value = record.get(key)
            if key == "date" and isinstance(value, datetime):
                value = value.strftime("%Y-%m-%d")

            if value:
                parts.append(f"{key}={value}")

        return Path(*parts) if parts else Path()

    async def read(self, relative_path: str) -> pa.Table:
        """
        Reads a Parquet file from the store.

        Args:
            relative_path: The path to the file, relative to the base store path.

        Returns:
            A pyarrow Table containing the file's data.
        """
        full_path = self.base_path / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {full_path}")

        # Run the synchronous pyarrow read operation in a separate thread
        return await asyncio.to_thread(pq.read_table, source=str(full_path))
