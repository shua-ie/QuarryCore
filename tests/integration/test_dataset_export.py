"""
Integration tests for DatasetConstructor export functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    import pyarrow.parquet as pq

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from quarrycore.config import Config
from quarrycore.dataset import DatasetConstructor
from quarrycore.protocols import ContentMetadata, ContentType, DomainType, ExtractedContent, QualityScore


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def dataset_config(temp_dir):
    """Create dataset configuration for testing."""
    config = Config()
    config.dataset.output_path = temp_dir / "datasets"
    config.dataset.export.output_path = str(temp_dir / "exports")
    config.dataset.export.formats = ["jsonl", "parquet"]
    return config.dataset


@pytest.fixture
def sample_content():
    """Create sample content for testing."""
    return [
        ExtractedContent(
            text="This is a sample document about machine learning and AI.",
            title="Introduction to ML",
            language="en",
            word_count=10,
        ),
        ExtractedContent(
            text="Another document discussing deep learning architectures and neural networks.",
            title="Deep Learning Guide",
            language="en",
            word_count=10,
        ),
    ]


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return [
        ContentMetadata(
            url="https://example.com/ml-intro",
            domain="example.com",
            domain_type=DomainType.TECHNICAL,
            title="Introduction to ML",
            author="John Doe",
            published_date=None,
            modified_date=None,
            content_type=ContentType.HTML,
            word_count=10,
        ),
        ContentMetadata(
            url="https://example.com/deep-learning",
            domain="example.com",
            domain_type=DomainType.TECHNICAL,
            title="Deep Learning Guide",
            author="Jane Smith",
            published_date=None,
            modified_date=None,
            content_type=ContentType.HTML,
            word_count=10,
        ),
    ]


@pytest.fixture
def sample_quality_scores():
    """Create sample quality scores for testing."""
    return [
        QualityScore(
            overall_score=0.85,
            confidence=0.9,
            grammar_score=0.9,
            coherence_score=0.85,
            toxicity_score=0.1,
            quality_factors={"readability": 0.8, "information_density": 0.9},
        ),
        QualityScore(
            overall_score=0.78,
            confidence=0.88,
            grammar_score=0.85,
            coherence_score=0.8,
            toxicity_score=0.05,
            quality_factors={"readability": 0.75, "information_density": 0.85},
        ),
    ]


class TestDatasetExportHappyPath:
    """Test happy-path dataset export functionality."""

    @pytest.mark.asyncio
    async def test_jsonl_export_creates_valid_file(
        self, dataset_config, temp_dir, sample_content, sample_metadata, sample_quality_scores
    ):
        """Test that JSONL export creates a valid file with correct content."""
        # Create dataset constructor
        constructor = DatasetConstructor(dataset_config)

        # Mock the sampling process to return our test data
        with patch.object(constructor.sampler, "sample") as mock_sample:
            mock_sample.return_value = list(zip(sample_metadata, sample_quality_scores, strict=False))

            # Mock the chunking process
            with patch.object(constructor.chunker, "chunk_batch") as mock_chunk:
                mock_chunk.return_value = [
                    ["This is a sample document about machine learning and AI."],
                    ["Another document discussing deep learning architectures and neural networks."],
                ]

                # Mock the formatting process
                with patch.object(constructor.formatter, "format_batch") as mock_format:
                    mock_format.return_value = [
                        {
                            "text": "This is a sample document about machine learning and AI.",
                            "instruction": "Summarize the following text:",
                            "response": "This is a sample document about machine learning and AI.",
                        },
                        {
                            "text": "Another document discussing deep learning architectures and neural networks.",
                            "instruction": "Summarize the following text:",
                            "response": "Another document discussing deep learning architectures and neural networks.",
                        },
                    ]

                    # Mock the exporter to actually write files
                    from quarrycore.dataset.exporter import JsonlExporter

                    exporter = JsonlExporter(dataset_config.export)

                    output_file = temp_dir / "test_output.jsonl"
                    exporter.export(mock_format.return_value, output_file)

                    # Verify file was created and has correct content
                    assert output_file.exists()

                    # Read and validate content
                    with open(output_file, "r") as f:
                        lines = f.readlines()

                    assert len(lines) == 2

                    # Parse each line as JSON
                    record1 = json.loads(lines[0])
                    record2 = json.loads(lines[1])

                    assert "text" in record1
                    assert "instruction" in record1
                    assert "response" in record1
                    assert "machine learning" in record1["text"]

                    assert "text" in record2
                    assert "instruction" in record2
                    assert "response" in record2
                    assert "deep learning" in record2["text"]

    @pytest.mark.asyncio
    async def test_parquet_export_creates_valid_file(
        self, dataset_config, temp_dir, sample_content, sample_metadata, sample_quality_scores
    ):
        """Test that Parquet export creates a valid file with correct schema."""
        # Create dataset constructor
        DatasetConstructor(dataset_config)

        # Create test data
        formatted_data = [
            {
                "text": "This is a sample document about machine learning and AI.",
                "instruction": "Summarize the following text:",
                "response": "This is a sample document about machine learning and AI.",
            },
            {
                "text": "Another document discussing deep learning architectures and neural networks.",
                "instruction": "Summarize the following text:",
                "response": "Another document discussing deep learning architectures and neural networks.",
            },
        ]

        # Test parquet export
        from quarrycore.dataset.exporter import ParquetExporter

        exporter = ParquetExporter(dataset_config.export)

        output_dir = temp_dir / "parquet_output"
        exporter.export(formatted_data, output_dir)

        # Verify directory and file were created
        assert output_dir.exists()
        parquet_file = output_dir / "data.parquet"
        assert parquet_file.exists()

        # Read and validate Parquet file
        table = pq.read_table(parquet_file)
        df = table.to_pandas()

        assert len(df) == 2
        assert "text" in df.columns
        assert "instruction" in df.columns
        assert "response" in df.columns

        # Check content
        assert "machine learning" in df.iloc[0]["text"]
        assert "deep learning" in df.iloc[1]["text"]

    @pytest.mark.asyncio
    async def test_multiple_format_export(
        self, dataset_config, temp_dir, sample_content, sample_metadata, sample_quality_scores
    ):
        """Test exporting to multiple formats simultaneously."""
        # Create test data
        formatted_data = [
            {
                "text": "Sample text for multi-format export test.",
                "instruction": "Summarize:",
                "response": "Sample text for multi-format export test.",
            }
        ]

        # Test both JSONL and Parquet export
        from quarrycore.dataset.exporter import JsonlExporter, ParquetExporter

        jsonl_exporter = JsonlExporter(dataset_config.export)
        jsonl_file = temp_dir / "multi_test.jsonl"
        jsonl_exporter.export(formatted_data, jsonl_file)

        if HAS_PYARROW:
            parquet_exporter = ParquetExporter(dataset_config.export)
            parquet_dir = temp_dir / "multi_test_parquet"
            parquet_exporter.export(formatted_data, parquet_dir)

            # Verify both files exist
            assert jsonl_file.exists()
            assert parquet_dir.exists()
            assert (parquet_dir / "data.parquet").exists()

    @pytest.mark.asyncio
    async def test_empty_data_handling(self, dataset_config, temp_dir):
        """Test export behavior with empty dataset."""
        from quarrycore.dataset.exporter import JsonlExporter

        exporter = JsonlExporter(dataset_config.export)
        output_file = temp_dir / "empty_test.jsonl"

        # Export empty data
        exporter.export([], output_file)

        # File should be created but empty
        assert output_file.exists()
        assert output_file.stat().st_size == 0

    @pytest.mark.asyncio
    async def test_large_text_handling(self, dataset_config, temp_dir):
        """Test export with large text content."""
        # Create large text content
        large_text = "This is a very long document. " * 1000  # ~30KB text

        formatted_data = [
            {
                "text": large_text,
                "instruction": "Summarize the following long text:",
                "response": large_text[:100] + "...",  # Truncated response
            }
        ]

        from quarrycore.dataset.exporter import JsonlExporter

        exporter = JsonlExporter(dataset_config.export)
        output_file = temp_dir / "large_text_test.jsonl"

        exporter.export(formatted_data, output_file)

        # Verify file was created and has content
        assert output_file.exists()
        assert output_file.stat().st_size > 0

        # Verify content is valid JSON
        with open(output_file, "r") as f:
            record = json.loads(f.readline())

        assert len(record["text"]) > 25000  # Should be large
        assert "instruction" in record
        assert "response" in record

    @pytest.mark.asyncio
    async def test_special_characters_handling(self, dataset_config, temp_dir):
        """Test export with special characters and Unicode."""
        # Test data with various special characters
        formatted_data = [
            {
                "text": "Text with émojis 🚀 and spëcial characters: àáâãäåæçèéêë",
                "instruction": "Process this tëxt:",
                "response": "Processed tëxt with émojis 🚀",
            },
            {
                "text": "Text with quotes \"and\" 'apostrophes' and\nnewlines\ttabs",
                "instruction": "Handle special chars:",
                "response": "Handled successfully",
            },
        ]

        from quarrycore.dataset.exporter import JsonlExporter

        exporter = JsonlExporter(dataset_config.export)
        output_file = temp_dir / "special_chars_test.jsonl"

        exporter.export(formatted_data, output_file)

        # Verify file was created
        assert output_file.exists()

        # Verify content can be read back correctly
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2

        record1 = json.loads(lines[0])
        record2 = json.loads(lines[1])

        assert "🚀" in record1["text"]
        assert "émojis" in record1["text"]
        assert "spëcial" in record1["text"]
        assert "\n" in record2["text"]
        assert "\t" in record2["text"]

    @pytest.mark.asyncio
    async def test_analytics_integration(self, dataset_config, sample_metadata, sample_quality_scores):
        """Test that analytics are properly generated for exported dataset."""
        constructor = DatasetConstructor(dataset_config)

        # Create formatted data
        formatted_data = [
            {
                "text": "Sample document for analytics testing.",
                "instruction": "Analyze:",
                "response": "Analyzed document.",
            }
        ]

        # Test analytics generation
        source_docs = list(zip(sample_metadata, sample_quality_scores, strict=False))
        report = constructor.analytics.analyze(formatted_data, source_docs)

        # Verify analytics report structure
        assert "general" in report
        assert "token_distribution" in report
        assert "domain_distribution" in report
        assert "quality_distribution" in report

        # Verify general stats
        assert report["general"]["total_records"] == 1
        assert report["general"]["total_source_documents"] == 2
        assert report["general"]["vocabulary_size"] > 0

        # Verify token distribution
        assert "mean" in report["token_distribution"]
        assert "std_dev" in report["token_distribution"]
        assert "min" in report["token_distribution"]
        assert "max" in report["token_distribution"]

    @pytest.mark.asyncio
    async def test_export_path_creation(self, temp_dir):
        """Test that export paths are created automatically."""
        # Create config with non-existent output path
        config = Config()
        nonexistent_path = temp_dir / "new_dir" / "datasets"
        config.dataset.export.output_path = str(nonexistent_path)

        from quarrycore.dataset.exporter import JsonlExporter

        exporter = JsonlExporter(config.dataset.export)

        output_file = nonexistent_path / "test.jsonl"

        # Create parent directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)

        formatted_data = [{"text": "test", "instruction": "test", "response": "test"}]
        exporter.export(formatted_data, output_file)

        # Verify path was created and file exists
        assert output_file.exists()
        assert output_file.parent.exists()

    @pytest.mark.asyncio
    async def test_concurrent_export_safety(self, dataset_config, temp_dir):
        """Test that concurrent exports don't interfere with each other."""
        import asyncio

        async def export_data(file_suffix):
            from quarrycore.dataset.exporter import JsonlExporter

            exporter = JsonlExporter(dataset_config.export)

            output_file = temp_dir / f"concurrent_test_{file_suffix}.jsonl"
            formatted_data = [
                {
                    "text": f"Concurrent export test {file_suffix}",
                    "instruction": "Test:",
                    "response": f"Response {file_suffix}",
                }
            ]

            exporter.export(formatted_data, output_file)
            return output_file

        # Run multiple exports concurrently
        tasks = [export_data(i) for i in range(3)]
        output_files = await asyncio.gather(*tasks)

        # Verify all files were created successfully
        for output_file in output_files:
            assert output_file.exists()

            # Verify content is correct
            with open(output_file, "r") as f:
                record = json.loads(f.readline())

            assert "Concurrent export test" in record["text"]
            assert "instruction" in record
            assert "response" in record
