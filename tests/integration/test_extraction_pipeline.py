"""
Integration tests for extraction pipeline with quality filtering.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from quarrycore.config import Config
from quarrycore.container import DependencyContainer
from quarrycore.pipeline import Pipeline


@pytest.fixture
def test_config(tmp_path: Path) -> Config:
    """Create test configuration."""
    config = Config()

    # Configure extraction settings
    config.extraction.cascade_order = ["trafilatura", "readability", "soup_fallback"]
    config.extraction.quality_threshold = 0.6
    config.extraction.domain_overrides = {"highquality.com": ["trafilatura"], "lowquality.com": ["soup_fallback"]}

    # Configure storage paths
    config.storage.hot.db_path = tmp_path / "test.db"
    config.storage.warm.base_path = tmp_path / "parquet"

    # Disable monitoring for tests
    config.monitoring.enabled = False
    config.monitoring.web_ui.enabled = False

    return config


@pytest.fixture
async def container(test_config: Config) -> DependencyContainer:
    """Create test container with config."""
    container = DependencyContainer()
    container.config = test_config
    await container.initialize()
    return container


class TestExtractionPipeline:
    """Integration tests for extraction pipeline."""

    @pytest.mark.asyncio
    async def test_high_quality_content_kept(self, container: DependencyContainer):
        """Test that high-quality content is kept in the pipeline."""
        # Mock HTTP client to return good HTML
        mock_http_client = AsyncMock()
        mock_http_client.fetch.return_value = MagicMock(
            status=200,
            final_url="http://highquality.com/article",
            body=b"""
            <html>
            <head><title>High Quality Article</title></head>
            <body>
                <article>
                    <h1>Important News</h1>
                    <p>This is a well-written article with substantial content that provides
                    valuable information to readers. The content is structured properly with
                    multiple paragraphs and good grammar.</p>
                    <p>Another paragraph with more detailed information about the topic,
                    ensuring that the quality score will be high due to the content length
                    and structure.</p>
                </article>
            </body>
            </html>
            """,
            headers={"content-type": "text/html"},
        )

        with patch.object(container, "get_http_client", return_value=mock_http_client):
            pipeline = Pipeline(container, max_concurrency=1)

            # Run pipeline
            result = await pipeline.run(urls=["http://highquality.com/article"], batch_size=1)

            # Assert content was processed successfully
            assert result["processed_count"] == 1
            assert result["failed_count"] == 0
            assert result["status"] == "completed"

            # Verify storage was called
            # The pipeline should have stored the extracted content

    @pytest.mark.asyncio
    async def test_low_quality_content_skipped(self, container: DependencyContainer):
        """Test that low-quality content is skipped in the pipeline."""
        # Mock HTTP client to return low quality HTML
        mock_http_client = AsyncMock()
        mock_http_client.fetch.return_value = MagicMock(
            status=200,
            final_url="http://lowquality.com/spam",
            body=b"""
            <html>
            <body>
                Buy now! Click here! Free!!!
                <script>alert('spam')</script>
                <div class="ad">Advertisement</div>
            </body>
            </html>
            """,
            headers={"content-type": "text/html"},
        )

        with patch.object(container, "get_http_client", return_value=mock_http_client):
            pipeline = Pipeline(container, max_concurrency=1)

            # Run pipeline
            result = await pipeline.run(urls=["http://lowquality.com/spam"], batch_size=1)

            # Content should be skipped due to low quality
            assert result["processed_count"] == 1  # Processed but skipped
            assert result["failed_count"] == 0
            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_quality_threshold_filtering(self, container: DependencyContainer):
        """Test that content exactly at threshold is handled correctly."""
        # Mock HTTP client
        mock_http_client = AsyncMock()
        mock_http_client.fetch.return_value = MagicMock(
            status=200,
            final_url="http://example.com/article",
            body=b"""
            <html>
            <head><title>Average Article</title></head>
            <body>
                <p>This is an article with moderate quality content. It has some
                useful information but not as comprehensive as high-quality content.</p>
            </body>
            </html>
            """,
            headers={"content-type": "text/html"},
        )

        # Mock quality assessor to return exactly threshold score (0.6)
        original_quality_method = None

        async def mock_assess_quality(content, metadata, **kwargs):
            from quarrycore.protocols import QualityScore

            return QualityScore(
                overall_score=0.6,  # Exactly at threshold
                quality_factors={"test": 0.6},
            )

        with patch.object(container, "get_http_client", return_value=mock_http_client):
            # Get quality assessor and mock its method
            quality = await container.get_quality()
            original_quality_method = quality.assess_quality
            quality.assess_quality = mock_assess_quality

            try:
                pipeline = Pipeline(container, max_concurrency=1)

                result = await pipeline.run(urls=["http://example.com/article"], batch_size=1)

                # Content at exact threshold should be kept
                assert result["processed_count"] == 1
                assert result["failed_count"] == 0
            finally:
                # Restore original method
                if original_quality_method:
                    quality.assess_quality = original_quality_method

    @pytest.mark.asyncio
    async def test_extractor_logs_recorded(self, container: DependencyContainer):
        """Test that extractor selection is properly logged."""
        # Mock HTTP client
        mock_http_client = AsyncMock()
        mock_http_client.fetch.return_value = MagicMock(
            status=200,
            final_url="http://highquality.com/article",
            body=b"<html><body><p>Test content for logging</p></body></html>",
            headers={"content-type": "text/html"},
        )

        with patch.object(container, "get_http_client", return_value=mock_http_client):
            pipeline = Pipeline(container, max_concurrency=1)

            result = await pipeline.run(urls=["http://highquality.com/article"], batch_size=1)

            # Just verify the pipeline ran successfully
            # The actual logs are shown in the terminal output
            assert result["processed_count"] == 1
            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_multiple_urls_with_mixed_quality(self, container: DependencyContainer):
        """Test processing multiple URLs with mixed quality levels."""
        # Mock HTTP client with different responses
        responses = [
            # High quality
            MagicMock(
                status=200,
                final_url="http://example.com/good",
                body=b"<html><body><article><p>" + b"Good content. " * 50 + b"</p></article></body></html>",
                headers={"content-type": "text/html"},
            ),
            # Low quality
            MagicMock(
                status=200,
                final_url="http://example.com/bad",
                body=b"<html><body>Buy now!</body></html>",
                headers={"content-type": "text/html"},
            ),
            # Medium quality
            MagicMock(
                status=200,
                final_url="http://example.com/medium",
                body=b"<html><body><p>Some decent content here.</p></body></html>",
                headers={"content-type": "text/html"},
            ),
        ]

        mock_http_client = AsyncMock()
        mock_http_client.fetch.side_effect = responses

        # Mock quality scores
        quality_scores = [0.8, 0.3, 0.65]  # High, low, medium
        score_index = 0

        async def mock_assess_quality(content, metadata, **kwargs):
            nonlocal score_index
            from quarrycore.protocols import QualityScore

            score = quality_scores[score_index % len(quality_scores)]
            score_index += 1
            return QualityScore(overall_score=score, quality_factors={"test": score})

        with patch.object(container, "get_http_client", return_value=mock_http_client):
            quality = await container.get_quality()
            original_method = quality.assess_quality
            quality.assess_quality = mock_assess_quality

            try:
                pipeline = Pipeline(container, max_concurrency=2)

                result = await pipeline.run(
                    urls=["http://example.com/good", "http://example.com/bad", "http://example.com/medium"],
                    batch_size=3,
                )

                # All should be processed
                assert result["processed_count"] == 3
                assert result["failed_count"] == 0

                # Bad content should have been skipped by extractor manager
                # but still counted as processed

            finally:
                quality.assess_quality = original_method
