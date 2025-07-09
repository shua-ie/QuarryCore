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
        # Use default sampling config if not provided
        from quarrycore.config.config import SamplingConfig

        sampling_config = getattr(config, "sampling", SamplingConfig())
        self.sampler = Sampler(sampling_config)

        # Use default chunking config if not provided
        from quarrycore.config.config import ChunkingConfig

        chunking_config = getattr(config, "chunking", ChunkingConfig())
        self.chunker = Chunker(chunking_config)

        # Use default formatting config if not provided
        from quarrycore.config.config import FormattingConfig

        formatting_config = getattr(config, "formatting", FormattingConfig())
        self.formatter = Formatter(formatting_config)

        self.analytics = Analytics(config)

        # Use default export config if not provided
        from quarrycore.config.config import ExportConfig

        export_config = getattr(config, "export", ExportConfig())
        self.exporters = [get_exporter(f, export_config) for f in getattr(export_config, "formats", ["jsonl"])]

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

        # Query storage for available content (simplified implementation)
        available_content: List[ProcessingResult] = await self._query_available_content()

        # 1. Sample documents based on curriculum strategy
        target_size = len(available_content) or 1000

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

    async def _query_available_content(self) -> List[ProcessingResult]:
        """Query storage for available content (simplified implementation)."""
        # For now return empty list - this should be injected with a storage manager
        # In a full implementation, this would use dependency injection
        return []

    async def sample_content(
        self,
        config: DatasetConfig,
        available_content: List[Tuple[ExtractedContent, ContentMetadata, QualityScore]],
    ) -> List[Tuple[ExtractedContent, ContentMetadata, QualityScore]]:
        """Sample content based on configuration."""
        # Apply quality filtering
        filtered_content = [
            (content, metadata, quality)
            for content, metadata, quality in available_content
            if quality.overall_score >= config.quality_threshold
        ]

        # Apply word count filtering
        size_filtered = [
            (content, metadata, quality)
            for content, metadata, quality in filtered_content
            if config.min_word_count <= len(content.text.split()) <= config.max_word_count
        ]

        # Apply language filtering
        language_filtered = [
            (content, metadata, quality)
            for content, metadata, quality in size_filtered
            if content.language in config.allowed_languages
        ]

        # Apply domain exclusion
        domain_filtered = [
            (content, metadata, quality)
            for content, metadata, quality in language_filtered
            if metadata.domain not in config.excluded_domains
        ]

        # Implement sampling strategy
        if config.sampling_strategy == "balanced":
            # Group by domain type and balance
            import random
            from collections import defaultdict

            domain_groups = defaultdict(list)
            for item in domain_filtered:
                domain_groups[item[1].domain_type].append(item)

            # Sample equally from each domain up to max_samples_per_domain
            sampled = []
            for _domain_type, items in domain_groups.items():
                max_samples = min(config.max_samples_per_domain, len(items))
                sampled.extend(random.sample(items, max_samples))

            return sampled
        else:
            # Simple random sampling
            import random

            max_samples = min(len(domain_filtered), config.max_samples_per_domain * 10)
            return random.sample(domain_filtered, max_samples)

    async def format_for_training(
        self,
        content: ExtractedContent,
        metadata: ContentMetadata,
        format_type: str = "text",
    ) -> Dict[str, Any]:
        """Format content for training."""
        if format_type == "text":
            return {
                "text": content.text,
                "title": content.title,
                "url": metadata.url,
                "quality_score": metadata.quality_score,
                "word_count": content.word_count,
                "language": content.language,
                "domain": metadata.domain,
                "published_date": metadata.published_date.isoformat() if metadata.published_date else None,
            }
        elif format_type == "instruction":
            return {
                "instruction": f"Summarize the following content: {content.title}",
                "input": content.text,
                "output": content.title,  # Simple placeholder
                "metadata": {
                    "url": metadata.url,
                    "domain": metadata.domain,
                    "quality_score": metadata.quality_score,
                },
            }
        elif format_type == "chat":
            return {
                "messages": [
                    {"role": "user", "content": f"Tell me about: {content.title}"},
                    {"role": "assistant", "content": content.text},
                ],
                "metadata": {
                    "url": metadata.url,
                    "domain": metadata.domain,
                    "quality_score": metadata.quality_score,
                },
            }
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    async def validate_dataset(self, dataset_path: Path, config: DatasetConfig) -> Dict[str, Any]:
        """Validate dataset quality."""
        import json
        import os
        from datetime import datetime
        from pathlib import Path

        if not dataset_path.exists():
            return {"valid": False, "error": "Dataset path does not exist"}

        validation_results = {
            "valid": True,
            "total_files": 0,
            "total_records": 0,
            "file_sizes": [],
            "format_errors": [],
            "quality_stats": {},
            "validation_timestamp": datetime.utcnow().isoformat(),
        }

        try:
            # Check if it's a directory or file
            if dataset_path.is_dir():
                # Validate directory of files
                for file_path in dataset_path.glob("**/*"):
                    if file_path.is_file():
                        validation_results["total_files"] += 1
                        validation_results["file_sizes"].append(file_path.stat().st_size)

                        # Validate file format
                        if file_path.suffix == ".jsonl":
                            try:
                                with open(file_path, "r", encoding="utf-8") as f:
                                    line_count = 0
                                    for line in f:
                                        json.loads(line)
                                        line_count += 1
                                validation_results["total_records"] += line_count
                            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                                validation_results["format_errors"].append({"file": str(file_path), "error": str(e)})
                                validation_results["valid"] = False
            else:
                # Single file validation
                validation_results["total_files"] = 1
                validation_results["file_sizes"] = [dataset_path.stat().st_size]

                if dataset_path.suffix == ".jsonl":
                    try:
                        with open(dataset_path, "r", encoding="utf-8") as f:
                            for line in f:
                                json.loads(line)
                                validation_results["total_records"] += 1
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        validation_results["format_errors"].append({"file": str(dataset_path), "error": str(e)})
                        validation_results["valid"] = False

        except Exception as e:
            validation_results["valid"] = False
            validation_results["error"] = str(e)

        # Calculate quality stats
        if validation_results["total_files"] > 0:
            file_sizes: List[int] = validation_results["file_sizes"]  # type: ignore
            validation_results["quality_stats"] = {
                "avg_file_size": sum(file_sizes) / len(file_sizes),
                "total_size_mb": sum(file_sizes) / (1024 * 1024),
                "records_per_file": validation_results["total_records"] / validation_results["total_files"],
                "error_rate": len(validation_results["format_errors"]) / validation_results["total_files"],
            }

        return validation_results

    async def export_dataset(self, *args: Any, **kwargs: Any) -> None:
        """Export dataset."""
        dataset = kwargs.get("dataset", args[0] if args else [])
        output_path = kwargs.get("output_path", args[1] if len(args) > 1 else Path("output"))
        format_type = kwargs.get("format_type", args[2] if len(args) > 2 else "jsonl")

        if not dataset:
            raise ValueError("Dataset is empty")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format_type == "jsonl":
            import json

            with open(output_path, "w", encoding="utf-8") as f:
                for record in dataset:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        elif format_type == "parquet":
            import pandas as pd

            df = pd.DataFrame(dataset)
            df.to_parquet(output_path, engine="pyarrow", compression="snappy")

        elif format_type == "json":
            import json

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)

        else:
            raise ValueError(f"Unsupported export format: {format_type}")

        print(f"Dataset exported to {output_path} in {format_type} format")

    # Other protocol methods would be implemented here
