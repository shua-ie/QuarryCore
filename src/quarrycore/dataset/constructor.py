"""
Orchestrates the entire dataset construction pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from quarrycore.config.config import DatasetConfig
from quarrycore.protocols import (
    ContentMetadata,
    DatasetProtocol,
    ExtractedContent,
    HardwareCapabilities,
    ProcessingResult,
    QualityScore,
)

from .analytics import Analytics
from .chunker import Chunker
from .exporter import get_exporter
from .formatter import Formatter
from .sampler import Sampler


class DatasetConstructor(DatasetProtocol):
    """
    Implements the DatasetProtocol to create high-quality, training-ready
    datasets from processed content.
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.sampler = Sampler(config.sampling)
        self.chunker = Chunker(config.chunking)
        self.formatter = Formatter(config.formatting)
        self.analytics = Analytics(config)
        self.exporters = [get_exporter(f, config.export) for f in config.export.formats]

    async def create_dataset(
        self,
        config: DatasetConfig,
        *,
        output_path: Path,
        hardware_caps: Optional[HardwareCapabilities] = None,
    ) -> Dict[str, Any]:
        """
        Runs the full pipeline to generate a dataset.

        Args:
            config: Dataset configuration
            output_path: Path where dataset should be saved
            hardware_caps: Hardware capabilities for optimization

        Returns:
            A dictionary containing the analytics report of the created dataset.
        """
        print("Starting dataset construction...")

        # TODO: Implement storage query to get available content
        available_content: List[ProcessingResult] = []  # Placeholder

        # 1. Sample documents based on curriculum strategy
        target_size = config.max_documents or len(available_content) or 1000

        # We need to pass metadata and quality scores to the sampler
        docs_for_sampling = [(res[2], res[4]) for res in available_content]  # (metadata, quality_score)
        sampled_docs = self.sampler.sample(docs_for_sampling, target_size)

        # Create a map from URL -> ExtractedContent to find the content for our sampled docs
        content_by_url = {res[2].url: res[1] for res in available_content}  # map url -> content

        final_content_for_chunking = [
            content_by_url[meta.url] for meta, score in sampled_docs if meta.url in content_by_url
        ]

        print(f"Sampled {len(sampled_docs)} documents for the dataset.")

        # 2. Chunk the text of the sampled documents
        texts_to_chunk = [content.text for content in final_content_for_chunking]
        all_chunks = await self.chunker.chunk_batch(texts_to_chunk)
        flat_chunks = [chunk for doc_chunks in all_chunks for chunk in doc_chunks]
        print(f"Created {len(flat_chunks)} chunks from sampled documents.")

        # 3. Format the chunks for training
        formatted_data = self.formatter.format_batch(flat_chunks)
        print(f"Formatted {len(formatted_data)} records for training.")

        # 4. Export the dataset to all configured formats
        for exporter in self.exporters:
            # Re-enable this when content query is ready
            # output_path = self.config.export.output_path / self.config.name
            # exporter.export(formatted_data, output_path)
            print(f"Simulating export with {exporter.__class__.__name__}")

        # 5. Analyze and report on the final dataset
        report = self.analytics.analyze(formatted_data, sampled_docs)
        self.analytics.pretty_print_report(report)

        print("Dataset construction complete.")
        return report

    # Other protocol methods would be implemented here
    async def sample_content(
        self,
        config: DatasetConfig,
        available_content: List[Tuple[ExtractedContent, ContentMetadata, QualityScore]],
    ) -> List[Tuple[ExtractedContent, ContentMetadata, QualityScore]]:
        """Sample content based on configuration."""
        raise NotImplementedError()

    async def format_for_training(
        self,
        content: ExtractedContent,
        metadata: ContentMetadata,
        format_type: str = "text",
    ) -> Dict[str, Any]:
        """Format content for training."""
        raise NotImplementedError()

    async def validate_dataset(self, dataset_path: Path, config: DatasetConfig) -> Dict[str, Any]:
        """Validate dataset quality."""
        raise NotImplementedError()

    async def export_dataset(self, *args: Any, **kwargs: Any) -> None:
        """Export dataset."""
        raise NotImplementedError()
