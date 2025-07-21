"""
Pipeline orchestration for QuarryCore.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field
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
from quarrycore.recovery.dead_letter import DeadLetterQueue
from quarrycore.utils import atomic_write_json, slugify


class PipelineSettings(BaseModel):
    """Configurable pipeline settings with environment variable support."""

    checkpoint_interval: float = Field(default=60.0, description="Checkpoint save interval in seconds")
    checkpoint_dir: Path = Field(default=Path("checkpoints"), description="Directory for checkpoint files")
    domain_failure_threshold: int = Field(
        default=5, description="Max failures per domain in failure_window before backoff"
    )
    domain_failure_window: float = Field(default=60.0, description="Time window in seconds for domain failure tracking")
    domain_backoff_duration: float = Field(default=120.0, description="Backoff duration in seconds for failing domains")
    dead_letter_db_path: Path = Field(default=Path("dead_letter.db"), description="Path to dead letter queue database")

    @classmethod
    def from_env(cls) -> "PipelineSettings":
        """Create settings from environment variables."""
        return cls(
            checkpoint_interval=float(os.getenv("CHECKPOINT_INTERVAL", "60.0")),
            checkpoint_dir=Path(os.getenv("CHECKPOINT_DIR", "checkpoints")),
            domain_failure_threshold=int(os.getenv("DOMAIN_FAILURE_THRESHOLD", "5")),
            domain_failure_window=float(os.getenv("DOMAIN_FAILURE_WINDOW", "60.0")),
            domain_backoff_duration=float(os.getenv("DOMAIN_BACKOFF_DURATION", "120.0")),
            dead_letter_db_path=Path(os.getenv("DEAD_LETTER_DB_PATH", "dead_letter.db")),
        )


@dataclass
class DomainFailureTracker:
    """Tracks failures per domain for backpressure control."""

    def __init__(self, threshold: int = 5, window: float = 60.0, backoff_duration: float = 120.0):
        self.threshold = threshold
        self.window = window
        self.backoff_duration = backoff_duration
        self.domain_failures: Dict[str, deque] = defaultdict(lambda: deque())
        self.domain_backoff: Dict[str, float] = {}

    def record_failure(self, domain: str) -> None:
        """Record a failure for a domain."""
        current_time = time.time()
        failures = self.domain_failures[domain]

        # Add new failure
        failures.append(current_time)

        # Remove failures outside the window
        while failures and failures[0] < current_time - self.window:
            failures.popleft()

        # Check if threshold exceeded
        if len(failures) >= self.threshold:
            self.domain_backoff[domain] = current_time + self.backoff_duration

    def is_domain_backed_off(self, domain: str) -> bool:
        """Check if a domain is currently backed off."""
        current_time = time.time()
        backoff_until = self.domain_backoff.get(domain, 0)

        if backoff_until > current_time:
            return True

        # Clean up expired backoffs
        if domain in self.domain_backoff and backoff_until <= current_time:
            del self.domain_backoff[domain]

        return False

    def get_backoff_remaining(self, domain: str) -> float:
        """Get remaining backoff time for a domain."""
        current_time = time.time()
        backoff_until = self.domain_backoff.get(domain, 0)
        return max(0, backoff_until - current_time)


class PipelineStage(Enum):
    """Pipeline processing stages."""

    CRAWL = "crawl"
    EXTRACT = "extract"
    METADATA = "metadata"
    DEDUPLICATE = "deduplicate"
    QUALITY = "quality"
    STORAGE = "storage"
    DATASET = "dataset"


class PipelineCheckpoint(BaseModel):
    """Pydantic model for pipeline checkpoint with full validation."""

    job_id: str = Field(..., description="Unique job identifier")
    pipeline_id: str = Field(..., description="Pipeline instance identifier")
    stage: str = Field(..., description="Current pipeline stage")
    processed_count: int = Field(ge=0, description="Number of URLs processed")
    failed_count: int = Field(ge=0, description="Number of URLs that failed")
    start_time: float = Field(..., description="Pipeline start timestamp")
    last_checkpoint: float = Field(..., description="Last checkpoint timestamp")
    urls_remaining: List[str] = Field(default_factory=list, description="URLs still to be processed")
    batch_size: int = Field(gt=0, description="Current batch size")
    error_count_by_stage: Dict[str, int] = Field(default_factory=dict, description="Error counts per stage")

    @classmethod
    def from_pipeline_state(cls, state: "PipelineState", job_id: str) -> "PipelineCheckpoint":
        """Create checkpoint from pipeline state."""
        return cls(
            job_id=job_id,
            pipeline_id=state.pipeline_id,
            stage=state.stage.value,
            processed_count=state.processed_count,
            failed_count=state.failed_count,
            start_time=state.start_time,
            last_checkpoint=state.last_checkpoint,
            urls_remaining=state.urls_remaining,
            batch_size=state.batch_size,
            error_count_by_stage=state.error_count_by_stage,
        )

    def to_pipeline_state(self) -> "PipelineState":
        """Convert checkpoint to pipeline state."""
        return PipelineState(
            pipeline_id=self.pipeline_id,
            stage=PipelineStage(self.stage),
            processed_count=self.processed_count,
            failed_count=self.failed_count,
            start_time=self.start_time,
            last_checkpoint=self.last_checkpoint,
            urls_remaining=self.urls_remaining,
            batch_size=self.batch_size,
            error_count_by_stage=self.error_count_by_stage,
        )


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

    def __init__(
        self, container: DependencyContainer, max_concurrency: int = 100, settings: Optional[PipelineSettings] = None
    ) -> None:
        self.container = container
        self.max_concurrency = max_concurrency
        self.settings = settings or PipelineSettings.from_env()
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.processing_queue: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=max_concurrency * 2)

        # Pipeline state
        self.state: Optional[PipelineState] = None
        self.is_running = False
        self.task_group: Optional[asyncio.TaskGroup] = None
        self.job_id: str = str(uuid4())

        # Error handling
        self.circuit_breakers = {stage.value: CircuitBreaker() for stage in PipelineStage}

        # Performance monitoring
        self.stage_timings: Dict[str, List[float]] = {}

        # Dead letter queue integration
        self.dead_letter_queue: Optional[DeadLetterQueue] = None

        # Signal handling for graceful shutdown
        self._shutdown_requested = False
        self._shutdown_event: Optional[asyncio.Event] = None
        self._original_sigint_handler = None
        self._original_sigterm_handler = None

        # Domain failure tracking for backpressure (AC-05)
        self.domain_failure_tracker = DomainFailureTracker(
            threshold=self.settings.domain_failure_threshold,
            window=self.settings.domain_failure_window,
            backoff_duration=self.settings.domain_backoff_duration,
        )

    async def run(
        self,
        urls: List[str],
        batch_size: int = 50,
        checkpoint_interval: Optional[float] = None,
        resume_from: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline with checkpointing and error recovery.
        """
        # Use settings-based interval if not specified
        interval = checkpoint_interval or self.settings.checkpoint_interval

        # Initialize dead letter queue
        self.dead_letter_queue = DeadLetterQueue(db_path=self.settings.dead_letter_db_path)
        await self.dead_letter_queue.initialize()

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        pipeline_id = str(uuid4())

        # Initialize or resume state
        if resume_from and resume_from.exists():
            checkpoint = await self._load_checkpoint(resume_from)
            self.state = checkpoint.to_pipeline_state()
            self.job_id = checkpoint.job_id

            # AC-03: Exact-state resume - if no URLs remaining, exit with completed status
            if not self.state.urls_remaining:
                self.logger.info(
                    "Resuming from completed checkpoint - no URLs remaining",
                    job_id=self.job_id,
                    processed=self.state.processed_count,
                    failed=self.state.failed_count,
                )
                return {
                    "job_id": self.job_id,
                    "pipeline_id": self.state.pipeline_id,
                    "status": "completed",
                    "processed_count": self.state.processed_count,
                    "failed_count": self.state.failed_count,
                    "duration": 0,
                    "message": "Resumed from completed checkpoint",
                }

            self.logger.info(
                "Resuming pipeline from checkpoint",
                job_id=self.job_id,
                pipeline_id=self.state.pipeline_id,
                stage=self.state.stage.value,
                processed=self.state.processed_count,
                remaining=len(self.state.urls_remaining),
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
                    checkpoint_task = asyncio.create_task(self._checkpoint_loop(interval))

                    # Start processing
                    result = await self._process_pipeline()

                    # Cancel checkpoint task
                    checkpoint_task.cancel()

                    # Save final checkpoint
                    await self._save_checkpoint()

                    return result

        except Exception as e:
            # Save checkpoint on error
            if self.state:
                await self._save_checkpoint()
            self.logger.error("Pipeline failed", error=str(e), job_id=self.job_id)
            raise
        finally:
            self.is_running = False
            # Clean up dead letter queue
            if self.dead_letter_queue:
                await self.dead_letter_queue.close()
            # Restore original signal handlers
            self._cleanup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown and checkpoint saving."""

        # Create shutdown event in current event loop
        try:
            self._shutdown_event = asyncio.Event()
        except RuntimeError:
            # No event loop available yet
            self._shutdown_event = None

        def signal_handler(signum: int, frame: Any) -> None:
            """Handle shutdown signals."""
            self.logger.info(f"Received signal {signum}, initiating shutdown")
            self._shutdown_requested = True

            # Set the shutdown event if it exists
            if self._shutdown_event:
                self._shutdown_event.set()

            # Try to schedule graceful shutdown in the event loop
            try:
                loop = asyncio.get_running_loop()
                # Only create task if loop is still running
                if not loop.is_closed():
                    loop.create_task(self._graceful_shutdown(f"signal_{signum}"))
            except RuntimeError:
                # No running event loop - only attempt emergency checkpoint if we have state
                if self.is_running and self.state:
                    self.logger.warning(
                        "No event loop available for graceful shutdown, attempting emergency checkpoint"
                    )
                    # Emergency synchronous checkpoint attempt
                    self._emergency_checkpoint()
                else:
                    # No need for emergency checkpoint if not running or no state
                    self.logger.debug("Shutdown signal received but pipeline not active")

        # Store original handlers
        self._original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, signal_handler)

    async def _graceful_shutdown(self, reason: str) -> None:
        """Perform graceful shutdown with checkpoint saving."""
        if not self.is_running:
            return

        self.logger.info(f"Starting graceful shutdown: {reason}")

        try:
            # Cancel any in-flight tasks
            if self.task_group:
                # Signal all workers to stop by putting None in the queue
                for _ in range(self.max_concurrency):
                    try:
                        self.processing_queue.put_nowait(None)
                    except asyncio.QueueFull:
                        pass

            # Save checkpoint with timeout
            if self.state:
                try:
                    await asyncio.wait_for(self._save_checkpoint(), timeout=2.0)
                    self.logger.info("Checkpoint saved during graceful shutdown")
                except asyncio.TimeoutError:
                    self.logger.error("Checkpoint save timed out during shutdown")

            # Close resources
            if hasattr(self, "dead_letter_queue") and self.dead_letter_queue:
                try:
                    await asyncio.wait_for(self.dead_letter_queue.close(), timeout=1.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Dead letter queue close timed out")

        except Exception as e:
            self.logger.error(f"Error during graceful shutdown: {e}")
        finally:
            self.is_running = False

    def _cleanup_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigint_handler:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
        if self._original_sigterm_handler:
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)

    async def _process_pipeline(self) -> Dict[str, Any]:
        """Process the complete pipeline."""
        if not self.state:
            raise RuntimeError("Pipeline state not initialized")

        results: Dict[str, Any] = {
            "job_id": self.job_id,
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
                num_workers = min(10, len(self.state.urls_remaining))
                workers = [tg.create_task(self._worker(f"worker-{i}")) for i in range(num_workers)]

                # Feed URLs to processing queue
                producer = tg.create_task(self._producer())

                # Wait for all processing to complete
                await asyncio.gather(*workers, producer, return_exceptions=True)

            results["processed_count"] = self.state.processed_count
            results["failed_count"] = self.state.failed_count
            results["end_time"] = time.time()
            results["duration"] = results["end_time"] - self.state.start_time
            results["status"] = "completed"

            return results

        except* Exception as eg:
            # Handle exception group from TaskGroup
            for e in eg.exceptions:
                self.logger.error("Task failed", error=str(e))
            raise

    async def _producer(self) -> None:
        """Feeds URLs to the processing queue with domain backoff checks."""
        if not self.state:
            return

        # Create a copy to avoid modifying the list while iterating
        urls_to_process = self.state.urls_remaining.copy()
        skipped_urls = []  # URLs skipped due to domain backoff

        for url in urls_to_process:
            # Check for shutdown signal
            if self._shutdown_requested:
                self.logger.info("Shutdown requested, stopping URL production")
                break

            # AC-05: Check domain backoff before processing
            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            if self.domain_failure_tracker.is_domain_backed_off(domain):
                backoff_remaining = self.domain_failure_tracker.get_backoff_remaining(domain)
                self.logger.info(
                    "Skipping URL due to domain backoff", url=url, domain=domain, backoff_remaining=backoff_remaining
                )
                skipped_urls.append(url)
                continue

            await self.processing_queue.put(url)

        # Re-queue skipped URLs at the end if there's still backoff time
        for url in skipped_urls:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            if not self.domain_failure_tracker.is_domain_backed_off(domain):
                await self.processing_queue.put(url)

        # Signal end of input to workers
        num_workers = min(10, len(self.state.urls_remaining))
        for _ in range(num_workers):
            await self.processing_queue.put(None)

    async def _worker(self, worker_id: str) -> None:
        """Worker task for processing URLs."""
        if not self.state:
            return

        # Check for test mode worker delay
        worker_delay = float(os.environ.get("QUARRY_PIPELINE__WORKER_DELAY", "0"))

        while True:
            # Check for shutdown signal
            if self._shutdown_requested:
                self.logger.info(f"Shutdown requested, stopping worker {worker_id}")
                break

            url = await self.processing_queue.get()
            if url is None:
                break

            # Add delay in test mode to allow signal interruption
            if worker_delay > 0:
                await asyncio.sleep(worker_delay)

            async with self.semaphore:
                try:
                    result = await self._process_url(url, worker_id)
                    if result:  # Check if result is not None (meaning it was accepted)
                        self.state.processed_count += 1
                    else:  # result is None (meaning it was rejected or failed)
                        self.state.failed_count += 1

                        # AC-05: Record domain failure for backpressure
                        parsed_url = urlparse(url)
                        domain = parsed_url.netloc
                        self.domain_failure_tracker.record_failure(domain)

                    # Remove URL from urls_remaining after processing (successful or failed)
                    if url in self.state.urls_remaining:
                        self.state.urls_remaining.remove(url)

                except Exception as e:
                    self.state.failed_count += 1

                    # AC-05: Record domain failure for backpressure
                    parsed_url = urlparse(url)
                    domain = parsed_url.netloc
                    self.domain_failure_tracker.record_failure(domain)

                    # Remove URL from urls_remaining even on exception
                    if url in self.state.urls_remaining:
                        self.state.urls_remaining.remove(url)
                    await self._handle_error(e, url, worker_id)
                finally:
                    self.processing_queue.task_done()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _process_url(self, url: str, worker_id: str) -> Optional[Dict[str, Any]]:
        """Process a single URL through the complete pipeline with REAL components."""
        start_time = time.time()

        try:
            # Get all required components from container
            # Note: Using available components, will add missing ones to container
            storage = await self.container.get_storage()
            observability = await self.container.get_observability()

            # Initialize stats if not present
            if not hasattr(self, "stats"):
                self.stats = {"accepted": 0, "rejected": 0}
            if not hasattr(self, "metrics"):
                self.metrics = observability

            # STAGE 1: CRAWLING (using production HTTP client)
            stage_start = time.time()
            circuit_breaker = self.circuit_breakers[PipelineStage.CRAWL.value]

            if not await circuit_breaker.can_execute():
                raise Exception(f"Circuit breaker open for stage {PipelineStage.CRAWL.value}")

            try:
                # Use production HTTP client with robots.txt compliance and rate limiting
                http_client = await self.container.get_http_client()
                crawler_response = await http_client.fetch(url, timeout=30.0)

                # Convert to CrawlResult
                crawl_result = CrawlResult(
                    url=url,
                    final_url=crawler_response.final_url,
                    status_code=crawler_response.status,
                    content=crawler_response.body,
                    headers=crawler_response.headers,
                    status=ProcessingStatus.COMPLETED if crawler_response.status < 400 else ProcessingStatus.FAILED,
                )
                await circuit_breaker.record_success()
                self._record_stage_timing(PipelineStage.CRAWL.value, time.time() - stage_start)
            except Exception as e:
                await circuit_breaker.record_failure()
                raise Exception(f"Crawling failed for {url}: {str(e)}")

            # STAGE 2: CONTENT EXTRACTION (using ExtractorManager)
            stage_start = time.time()
            circuit_breaker = self.circuit_breakers[PipelineStage.EXTRACT.value]

            if not await circuit_breaker.can_execute():
                raise Exception(f"Circuit breaker open for stage {PipelineStage.EXTRACT.value}")

            try:
                # Use ExtractorManager for cascading extraction with quality gating
                extractor_manager = await self.container.get_extractor_manager()
                # Decode HTML content from bytes to string
                html_content = crawl_result.content.decode("utf-8", errors="replace")
                extract_result = await extractor_manager.extract(url, html_content)

                if extract_result is None:
                    # All extractors failed or content was below quality threshold
                    self.stats["rejected"] += 1
                    # Use metrics directly since observability doesn't have observe method
                    from quarrycore.observability import increment

                    increment("documents_rejected_total", 1)
                    self.logger.info("Low quality or failed extraction", url=url, worker_id=worker_id)
                    # Return a minimal ProcessedDoc to indicate it was processed (but rejected)
                    return {
                        "url": url,
                        "extractor": None,
                        "quality": 0.0,
                        "content": "",
                        "metadata": {"rejected": True, "reason": "low_quality_or_extraction_failed"},
                    }

                # Convert to protocol's ExtractedContent
                extracted_content = ExtractedContent(
                    text=extract_result.text,
                    title=extract_result.title or "",  # Handle None title
                    word_count=len(extract_result.text.split()),
                    extraction_method="extractor_manager",
                    confidence_score=extract_result.score,
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
                    self.stats["rejected"] += 1
                    from quarrycore.observability import increment

                    increment("documents_rejected_total", 1)
                    return None
            except Exception as e:
                await circuit_breaker.record_failure()
                raise Exception(f"Deduplication check failed for {url}: {str(e)}")

            # STAGE 5: QUALITY ASSESSMENT
            stage_start = time.time()

            # Get quality assessor from container
            quality_assessor = await self.container.get_quality()

            # Score the extracted text
            quality_score_value = await quality_assessor.score(extract_result.text)

            # Check against minimum quality threshold
            min_quality_score = self.container.config.quality.min_score
            if quality_score_value < min_quality_score:
                if "quality_reject_total" in self.metrics:
                    self.metrics["quality_reject_total"].inc()

                self.logger.info(
                    "quality.reject",
                    score=quality_score_value,
                    threshold=min_quality_score,
                    url=url,
                    worker_id=worker_id,
                )

                # Track in state
                async with self.state_lock:
                    if self.state:
                        self.state.failed_count += 1

                return None

            # Create QualityScore object for storage
            from quarrycore.protocols import QualityScore

            quality_score = QualityScore(
                overall_score=quality_score_value,
                quality_factors={
                    "length": quality_score_value,  # Will be populated by individual scorers
                    "language": quality_score_value,
                    "coherence": quality_score_value,
                },
            )

            self._record_stage_timing(PipelineStage.QUALITY.value, time.time() - stage_start)

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

            # Create ProcessedDoc result
            processed_doc: Dict[str, Any] = {
                "url": url,
                "extractor": "extractor_manager",  # This should come from ExtractorManager
                "quality": extract_result.score,
                "content": extract_result.text,
                "metadata": {
                    "title": metadata.title,
                    "domain": metadata.domain,
                    "word_count": metadata.word_count,
                    "document_id": str(document_id),
                },
            }

            # Update stats and metrics
            self.stats["accepted"] += 1
            from quarrycore.observability import increment

            increment("documents_accepted_total", 1)

            # Structured log for integration tests
            self.logger.info(
                "extractor_selected",
                extra={
                    "event": "extractor_selected",
                    "extractor": processed_doc["extractor"],
                    "quality": processed_doc["quality"],
                    "url": url,
                },
                url=url,
                worker_id=worker_id,
                document_id=str(document_id),
                quality_score=quality_score.overall_score,
                duration=total_duration,
            )

            return processed_doc

        except Exception as e:
            self.logger.error("URL processing failed", url=url, worker_id=worker_id, error=str(e))

            # Add to dead letter queue for failed URLs
            if self.dead_letter_queue:
                try:
                    await self.dead_letter_queue.add_failed_document(
                        url=url,
                        failure_stage=PipelineStage.CRAWL.value,  # Default to crawl stage
                        error_info=ErrorInfo(
                            error_type=type(e).__name__,
                            error_message=str(e),
                            severity=ErrorSeverity.MEDIUM,
                            is_retryable=True,
                            context={"url": url, "worker_id": worker_id},
                        ),
                        metadata={
                            "job_id": self.job_id,
                            "pipeline_id": self.state.pipeline_id if self.state else "unknown",
                        },
                    )
                except Exception as dlq_error:
                    self.logger.error(f"Failed to add URL to dead letter queue: {dlq_error}")

            # Return None for failed processing
            return None

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

        # Add to dead letter queue
        if self.dead_letter_queue:
            try:
                await self.dead_letter_queue.add_failed_document(
                    url=url,
                    failure_stage="unknown",
                    error_info=error_info,
                    metadata={
                        "job_id": self.job_id,
                        "pipeline_id": self.state.pipeline_id if self.state else "unknown",
                    },
                )
            except Exception as dlq_error:
                self.logger.error(f"Failed to add error to dead letter queue: {dlq_error}")

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
        """Handle storage errors by checking disk space and permissions."""
        # Log warning about storage error
        self.logger.warning("Storage error detected - checking disk space and permissions", job_id=self.job_id)
        # TODO: Implement storage-specific recovery logic
        # - Check available disk space
        # - Verify write permissions
        # - Consider alternative storage locations

    def _emergency_checkpoint(self) -> Optional[Path]:
        """
        Create an emergency checkpoint when no event loop is available.

        Returns:
            Path to checkpoint file if successful, None otherwise
        """
        if not self.state:
            self.logger.error("Cannot create emergency checkpoint: no state available")
            return None

        try:
            # Ensure checkpoint directory exists
            self.settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = self.settings.checkpoint_dir / f"emergency_{self.job_id}.json"
            checkpoint = PipelineCheckpoint.from_pipeline_state(self.state, self.job_id)

            # Try to write using atomic_write_json (sync version)
            try:
                atomic_write_json(checkpoint_path, checkpoint.model_dump())
                self.logger.info(f"Emergency checkpoint saved to {checkpoint_path}")
                return checkpoint_path
            except Exception as atomic_error:
                # Fallback to direct write
                self.logger.warning(f"Atomic write failed, using fallback: {atomic_error}")
                checkpoint_path.write_text(json.dumps(checkpoint.model_dump(), indent=2), encoding="utf-8")
                self.logger.info(f"Emergency checkpoint saved (fallback) to {checkpoint_path}")
                return checkpoint_path

        except Exception as e:
            self.logger.error(f"Failed to create emergency checkpoint: {e}")
            return None

    async def _checkpoint_loop(self, interval: float) -> None:
        """Periodic checkpoint saving every interval seconds."""
        while self.is_running and not self._shutdown_requested:
            await asyncio.sleep(interval)
            if self.state:
                await self._save_checkpoint()

    async def _save_checkpoint(self) -> None:
        """Save current pipeline state atomically using cross-platform atomic writes."""
        if not self.state:
            return

        # AC-02: Slugify job_id for safe filenames
        safe_job_id = slugify(self.job_id, replacement="-", max_length=100, lowercase=True)

        # AC-06: Use configurable checkpoint directory
        checkpoint_dir = self.settings.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{safe_job_id}.json"

        # Create checkpoint model
        checkpoint = PipelineCheckpoint.from_pipeline_state(self.state, self.job_id)

        # AC-01: Use atomic write utility for cross-platform compatibility
        try:
            from quarrycore.utils.atomic import atomic_json_dump

            checkpoint_data = checkpoint.model_dump()
            success = await atomic_json_dump(checkpoint_data, checkpoint_path, timeout=2.0)

            if success:
                self.state.last_checkpoint = time.time()
                self.logger.info(
                    "Checkpoint saved atomically",
                    path=str(checkpoint_path),
                    job_id=self.job_id,
                    safe_job_id=safe_job_id,
                    processed=self.state.processed_count,
                    remaining=len(self.state.urls_remaining),
                )
            else:
                self.logger.error("Failed to save checkpoint: write timeout or error")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    async def _load_checkpoint(self, path: Path) -> PipelineCheckpoint:
        """Load pipeline checkpoint from file."""
        with open(path) as f:
            data = json.load(f)
        return PipelineCheckpoint.model_validate(data)

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
