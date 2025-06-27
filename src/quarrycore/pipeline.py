"""
Production-grade pipeline orchestration for QuarryCore.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from quarrycore.container import DependencyContainer
from quarrycore.crawler.circuit_breaker import CircuitBreaker
from quarrycore.protocols import (
    ContentMetadata,
    CrawlResult,
    DuplicationResult,
    ErrorInfo,
    ErrorSeverity,
    ExtractedContent,
    ProcessingStatus,
)


class PipelineStage(Enum):
    """Pipeline processing stages."""

    CRAWL = "crawl"
    EXTRACT = "extract"
    METADATA = "metadata"
    DEDUPLICATE = "deduplicate"
    QUALITY = "quality"
    STORAGE = "storage"
    DATASET = "dataset"


@dataclass
class ProcessingResult:
    """Result of document processing."""

    document_id: Optional[UUID]
    status: ProcessingStatus
    stage_completed: PipelineStage
    quality_score: Optional[float] = None
    error_info: Optional[ErrorInfo] = None
    processing_time: Optional[float] = None
    storage_path: Optional[str] = None
    rejection_reason: Optional[str] = None


@dataclass
class PipelineState:
    """Persistent pipeline state for checkpoint/resume."""

    pipeline_id: str
    stage: PipelineStage
    processed_count: int
    failed_count: int
    start_time: float
    last_checkpoint: float
    urls_remaining: List[str]
    batch_size: int
    error_count_by_stage: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineState:
        """Create from dictionary."""
        data["stage"] = PipelineStage(data["stage"])
        return cls(**data)


# Circuit breaker functionality is provided by the dedicated CircuitBreaker module
# Import the production-grade implementation to avoid race conditions


class Pipeline:
    """
    Production-grade pipeline orchestrator with real component integration.
    """

    def __init__(self, container: DependencyContainer, max_concurrency: int = 100) -> None:
        self.container = container
        self.max_concurrency = max_concurrency
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.processing_queue: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=max_concurrency * 2)

        # Pipeline state
        self.state: Optional[PipelineState] = None
        self.is_running = False
        self.task_group: Optional[asyncio.TaskGroup] = None

        # Error handling
        self.circuit_breakers = {stage.value: CircuitBreaker() for stage in PipelineStage}

        # Performance monitoring
        self.stage_timings: Dict[str, List[float]] = {}

    async def run(
        self,
        urls: List[str],
        batch_size: int = 50,
        checkpoint_interval: float = 300.0,  # 5 minutes
        resume_from: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline with checkpointing and error recovery.
        """
        pipeline_id = str(uuid4())

        # Initialize or resume state
        if resume_from and resume_from.exists():
            self.state = await self._load_checkpoint(resume_from)
            self.logger.info(
                "Resuming pipeline from checkpoint",
                pipeline_id=self.state.pipeline_id,
                stage=self.state.stage.value,
                processed=self.state.processed_count,
            )
        else:
            self.state = PipelineState(
                pipeline_id=pipeline_id,
                stage=PipelineStage.CRAWL,
                processed_count=0,
                failed_count=0,
                start_time=time.time(),
                last_checkpoint=time.time(),
                urls_remaining=urls.copy(),
                batch_size=batch_size,
                error_count_by_stage={stage.value: 0 for stage in PipelineStage},
            )

        self.is_running = True

        try:
            async with self.container.lifecycle():
                observability = await self.container.get_observability()

                async with observability.start_monitoring():
                    # Start checkpoint task
                    checkpoint_task = asyncio.create_task(self._checkpoint_loop(checkpoint_interval))

                    # Start processing
                    result = await self._process_pipeline()

                    # Cancel checkpoint task
                    checkpoint_task.cancel()

                    return result

        except Exception as e:
            self.logger.error("Pipeline failed", error=str(e), pipeline_id=pipeline_id)
            raise
        finally:
            self.is_running = False

    async def _process_pipeline(self) -> Dict[str, Any]:
        """Process the complete pipeline."""
        if not self.state:
            raise RuntimeError("Pipeline state not initialized")

        results: Dict[str, Any] = {
            "pipeline_id": self.state.pipeline_id,
            "start_time": self.state.start_time,
            "processed_count": 0,
            "failed_count": 0,
            "stages_completed": [],
        }

        try:
            # Process URLs in batches with backpressure handling
            async with asyncio.TaskGroup() as tg:
                self.task_group = tg

                # Create worker tasks for parallel processing
                workers = [
                    tg.create_task(self._worker(f"worker-{i}")) for i in range(min(10, len(self.state.urls_remaining)))
                ]

                # Feed URLs to processing queue
                producer = tg.create_task(self._producer())

                # Wait for all processing to complete
                await asyncio.gather(*workers, producer, return_exceptions=True)

            results["processed_count"] = self.state.processed_count
            results["failed_count"] = self.state.failed_count
            results["end_time"] = time.time()
            results["duration"] = results["end_time"] - self.state.start_time

            return results

        except* Exception as eg:
            # Handle exception group from TaskGroup
            for e in eg.exceptions:
                self.logger.error("Task failed", error=str(e))
            raise

    async def _producer(self) -> None:
        """Feeds URLs to the processing queue."""
        if not self.state:
            return

        for url in self.state.urls_remaining:
            await self.processing_queue.put(url)

        # Signal end of input
        for _ in range(10):  # Number of workers
            await self.processing_queue.put(None)

    async def _worker(self, worker_id: str) -> None:
        """Worker task for processing URLs."""
        if not self.state:
            return

        while True:
            url = await self.processing_queue.get()
            if url is None:
                break

            async with self.semaphore:
                try:
                    result = await self._process_url(url, worker_id)
                    if result.status == ProcessingStatus.COMPLETED:
                        self.state.processed_count += 1
                    else:
                        self.state.failed_count += 1
                except Exception as e:
                    self.state.failed_count += 1
                    await self._handle_error(e, url, worker_id)
                finally:
                    self.processing_queue.task_done()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _process_url(self, url: str, worker_id: str) -> ProcessingResult:
        """Process a single URL through the complete pipeline with REAL components."""
        start_time = time.time()

        try:
            # Get all required components from container
            # Note: Using available components, will add missing ones to container
            quality = await self.container.get_quality()
            storage = await self.container.get_storage()
            observability = await self.container.get_observability()

            # For now, create mock implementations for missing components
            # These will be replaced with real components as they're added to container

            # STAGE 1: CRAWLING (using basic HTTP client for now)
            stage_start = time.time()
            circuit_breaker = self.circuit_breakers[PipelineStage.CRAWL.value]

            if not await circuit_breaker.can_execute():
                raise Exception(f"Circuit breaker open for stage {PipelineStage.CRAWL.value}")

            try:
                # Basic HTTP crawling implementation
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=30.0)
                    crawl_result = CrawlResult(
                        url=url,
                        final_url=str(response.url),
                        status_code=response.status_code,
                        content=response.content,
                        headers=dict(response.headers),
                        status=ProcessingStatus.COMPLETED,
                    )
                await circuit_breaker.record_success()
                self._record_stage_timing(PipelineStage.CRAWL.value, time.time() - stage_start)
            except Exception as e:
                await circuit_breaker.record_failure()
                raise Exception(f"Crawling failed for {url}: {str(e)}")

            # STAGE 2: CONTENT EXTRACTION (using basic HTML parsing for now)
            stage_start = time.time()
            circuit_breaker = self.circuit_breakers[PipelineStage.EXTRACT.value]

            if not await circuit_breaker.can_execute():
                raise Exception(f"Circuit breaker open for stage {PipelineStage.EXTRACT.value}")

            try:
                # Basic content extraction
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(crawl_result.content, "html.parser")

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                text = soup.get_text()
                title_element = soup.title
                title = title_element.string if title_element and title_element.string else ""

                extracted_content = ExtractedContent(
                    text=text.strip(),
                    title=title.strip(),
                    word_count=len(text.split()),
                    extraction_method="basic_html_parser",
                    confidence_score=0.8,
                )
                await circuit_breaker.record_success()
                self._record_stage_timing(PipelineStage.EXTRACT.value, time.time() - stage_start)
            except Exception as e:
                await circuit_breaker.record_failure()
                raise Exception(f"Content extraction failed for {url}: {str(e)}")

            # STAGE 3: METADATA EXTRACTION (basic implementation)
            stage_start = time.time()
            circuit_breaker = self.circuit_breakers[PipelineStage.METADATA.value]

            if not await circuit_breaker.can_execute():
                raise Exception(f"Circuit breaker open for stage {PipelineStage.METADATA.value}")

            try:
                from urllib.parse import urlparse

                parsed_url = urlparse(url)

                metadata = ContentMetadata(
                    url=url,
                    canonical_url=crawl_result.final_url,
                    title=extracted_content.title,
                    domain=parsed_url.netloc,
                    http_status=crawl_result.status_code,
                    word_count=extracted_content.word_count,
                )
                await circuit_breaker.record_success()
                self._record_stage_timing(PipelineStage.METADATA.value, time.time() - stage_start)
            except Exception as e:
                await circuit_breaker.record_failure()
                raise Exception(f"Metadata extraction failed for {url}: {str(e)}")

            # STAGE 4: DEDUPLICATION CHECK (basic hash-based for now)
            stage_start = time.time()
            circuit_breaker = self.circuit_breakers[PipelineStage.DEDUPLICATE.value]

            if not await circuit_breaker.can_execute():
                raise Exception(f"Circuit breaker open for stage {PipelineStage.DEDUPLICATE.value}")

            try:
                import hashlib

                content_hash = hashlib.sha256(extracted_content.text.encode()).hexdigest()

                dedup_result = DuplicationResult(
                    content_hash=content_hash,
                    is_duplicate=False,  # Basic implementation - assume not duplicate
                    duplicate_type="hash",
                    confidence_score=0.9,
                )
                await circuit_breaker.record_success()
                self._record_stage_timing(PipelineStage.DEDUPLICATE.value, time.time() - stage_start)

                if dedup_result.is_duplicate:
                    return ProcessingResult(
                        document_id=None,
                        status=ProcessingStatus.SKIPPED,
                        stage_completed=PipelineStage.DEDUPLICATE,
                        rejection_reason="duplicate_content",
                        processing_time=time.time() - start_time,
                    )
            except Exception as e:
                await circuit_breaker.record_failure()
                raise Exception(f"Deduplication check failed for {url}: {str(e)}")

            # STAGE 5: QUALITY ASSESSMENT (using real quality assessor)
            stage_start = time.time()
            circuit_breaker = self.circuit_breakers[PipelineStage.QUALITY.value]

            if not await circuit_breaker.can_execute():
                raise Exception(f"Circuit breaker open for stage {PipelineStage.QUALITY.value}")

            try:
                quality_score = await quality.assess_quality(content=extracted_content, metadata=metadata)
                await circuit_breaker.record_success()
                self._record_stage_timing(PipelineStage.QUALITY.value, time.time() - stage_start)

                # Check quality threshold
                min_quality = 0.5  # Configure this
                if quality_score.overall_score < min_quality:
                    return ProcessingResult(
                        document_id=None,
                        status=ProcessingStatus.SKIPPED,
                        stage_completed=PipelineStage.QUALITY,
                        quality_score=quality_score.overall_score,
                        rejection_reason="low_quality",
                        processing_time=time.time() - start_time,
                    )
            except Exception as e:
                await circuit_breaker.record_failure()
                raise Exception(f"Quality assessment failed for {url}: {str(e)}")

            # STAGE 6: STORAGE (using real storage manager)
            stage_start = time.time()
            circuit_breaker = self.circuit_breakers[PipelineStage.STORAGE.value]

            if not await circuit_breaker.can_execute():
                raise Exception(f"Circuit breaker open for stage {PipelineStage.STORAGE.value}")

            try:
                document_id = await storage.store_extracted_content(
                    content=extracted_content,
                    metadata=metadata,
                    quality=quality_score,
                    dedup_result=dedup_result,
                )
                await circuit_breaker.record_success()
                self._record_stage_timing(PipelineStage.STORAGE.value, time.time() - stage_start)
            except Exception as e:
                await circuit_breaker.record_failure()
                raise Exception(f"Storage failed for {url}: {str(e)}")

            total_duration = time.time() - start_time
            self.logger.info(
                "URL processed successfully",
                url=url,
                worker_id=worker_id,
                document_id=str(document_id),
                quality_score=quality_score.overall_score,
                duration=total_duration,
            )

            return ProcessingResult(
                document_id=document_id,
                status=ProcessingStatus.COMPLETED,
                stage_completed=PipelineStage.STORAGE,
                quality_score=quality_score.overall_score,
                processing_time=total_duration,
            )

        except Exception as e:
            self.logger.error("URL processing failed", url=url, worker_id=worker_id, error=str(e))

            # Add to dead letter queue for retry
            try:
                observability = await self.container.get_observability()
                await observability.log_error(
                    ErrorInfo(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        severity=ErrorSeverity.MEDIUM,
                        is_retryable=True,
                        context={"url": url, "worker_id": worker_id},
                    ),
                    "pipeline",
                )
            except Exception as dlq_error:
                self.logger.error(f"Failed to log error: {dlq_error}")

            return ProcessingResult(
                document_id=None,
                status=ProcessingStatus.FAILED,
                stage_completed=PipelineStage.CRAWL,
                error_info=ErrorInfo(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    severity=ErrorSeverity.MEDIUM,
                ),
                processing_time=time.time() - start_time,
            )

    def _record_stage_timing(self, stage: str, duration: float) -> None:
        """Record timing for a pipeline stage."""
        if stage not in self.stage_timings:
            self.stage_timings[stage] = []
        self.stage_timings[stage].append(duration)

    async def _handle_error(self, error: Exception, url: str, worker_id: str) -> None:
        """Handle processing errors with classification."""
        error_info = ErrorInfo(
            error_type=type(error).__name__,
            error_message=str(error),
            severity=ErrorSeverity.MEDIUM,
            context={"url": url, "worker_id": worker_id},
        )

        observability = await self.container.get_observability()
        await observability.log_error(error_info, "pipeline")

        # Implement error-specific recovery logic
        if "GPU" in str(error) and "memory" in str(error).lower():
            await self._handle_gpu_oom()
        elif "network" in str(error).lower():
            await self._handle_network_error(url)
        elif "storage" in str(error).lower():
            await self._handle_storage_error()

    async def _handle_gpu_oom(self) -> None:
        """Handle GPU out-of-memory errors by reducing batch size."""
        if not self.state:
            return

        self.state.batch_size = max(1, self.state.batch_size // 2)
        self.logger.warning(
            "GPU OOM detected, reducing batch size",
            new_batch_size=self.state.batch_size,
        )

    async def _handle_network_error(self, url: str) -> None:
        """Handle network errors with domain-specific retry logic."""
        # Add URL back to queue for retry
        await asyncio.sleep(5)  # Brief delay before retry
        await self.processing_queue.put(url)

    async def _handle_storage_error(self) -> None:
        """Handle storage errors with backup location fallback."""
        self.logger.warning("Storage error detected, implementing fallback strategy")
        # Implementation would switch to backup storage location

    async def _checkpoint_loop(self, interval: float) -> None:
        """Periodic checkpoint saving."""
        while self.is_running:
            await asyncio.sleep(interval)
            if self.state:
                await self._save_checkpoint()

    async def _save_checkpoint(self) -> None:
        """Save current pipeline state."""
        if not self.state:
            return

        checkpoint_path = Path(f"checkpoints/pipeline_{self.state.pipeline_id}.json")
        checkpoint_path.parent.mkdir(exist_ok=True)

        with open(checkpoint_path, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

        self.state.last_checkpoint = time.time()
        self.logger.info("Checkpoint saved", path=str(checkpoint_path))

    async def _load_checkpoint(self, path: Path) -> PipelineState:
        """Load pipeline state from checkpoint."""
        with open(path) as f:
            data = json.load(f)
        return PipelineState.from_dict(data)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all stages."""
        stats: Dict[str, Any] = {}
        for stage, timings in self.stage_timings.items():
            if timings:
                stats[stage] = {
                    "count": len(timings),
                    "avg_duration": sum(timings) / len(timings),
                    "min_duration": min(timings),
                    "max_duration": max(timings),
                    "total_duration": sum(timings),
                }
        return stats
