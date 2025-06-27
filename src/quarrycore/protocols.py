"""
Production-grade protocols and dataclasses for QuarryCore AI Training Data Miner.

This module defines the core contracts and data structures for the entire system,
ensuring type safety, performance monitoring, and hardware adaptability from
Raspberry Pi to GPU workstations.

Architecture Overview:
- 8 loosely-coupled modules with async-first design
- Multi-modal content processing (text, tables, code, images, links)
- 4-stage deduplication pipeline (exact, bloom, minhash, semantic)
- Hardware-adaptive processing with GPU acceleration
- Production-grade error handling and observability
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncContextManager, AsyncIterator, Dict, List, Optional, Protocol, Set, Tuple
from uuid import UUID, uuid4

import numpy as np
import polars as pl

# ============================================================================
# Enums and Constants
# ============================================================================


class ProcessingStatus(Enum):
    """Status codes for processing pipeline stages."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class ErrorSeverity(Enum):
    """Error severity levels for structured error handling."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContentType(Enum):
    """Supported content types for multi-modal processing."""

    TEXT = "text"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    XML = "xml"
    MARKDOWN = "markdown"
    CODE = "code"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class DomainType(Enum):
    """Supported domain types for specialized processing."""

    MEDICAL = "medical"
    LEGAL = "legal"
    ECOMMERCE = "ecommerce"
    TECHNICAL = "technical"
    NEWS = "news"
    ACADEMIC = "academic"
    SOCIAL = "social"
    GENERAL = "general"


class HardwareType(Enum):
    """Hardware configuration types for adaptive processing."""

    RASPBERRY_PI = "raspberry_pi"
    LAPTOP = "laptop"
    WORKSTATION = "workstation"
    SERVER = "server"
    CLOUD = "cloud"


# ============================================================================
# Core Dataclasses
# ============================================================================


@dataclass(frozen=True)
class PerformanceMetrics:
    """Performance metrics for system monitoring and optimization."""

    # Timing metrics (milliseconds)
    total_duration_ms: float
    download_duration_ms: float
    parsing_duration_ms: float
    extraction_duration_ms: float
    processing_duration_ms: float

    # Throughput metrics
    bytes_downloaded: int
    bytes_processed: int
    documents_per_second: float

    # Resource utilization
    peak_memory_mb: float
    avg_cpu_percent: float
    gpu_utilization_percent: Optional[float] = None
    cache_hit_ratio: float = 0.0

    # Quality metrics
    success_rate: float = 1.0
    retry_count: int = 0
    error_count: int = 0

    def __post_init__(self) -> None:
        """Validate performance metrics."""
        if self.total_duration_ms < 0:
            raise ValueError("Duration cannot be negative")
        if not 0 <= self.success_rate <= 1:
            raise ValueError("Success rate must be between 0 and 1")


@dataclass
class ErrorInfo:
    """Detailed error information for debugging and monitoring."""

    error_id: UUID = field(default_factory=uuid4)
    error_type: str = ""
    error_message: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[UUID] = None

    # Recovery information
    retry_count: int = 0
    max_retries: int = 3
    recovery_action: Optional[str] = None
    is_retryable: bool = True


@dataclass
class CrawlResult:
    """Result of web crawling operation with comprehensive metrics."""

    # Core identifiers
    request_id: UUID = field(default_factory=uuid4)
    url: str = ""
    final_url: str = ""  # After redirects

    # Response data
    status_code: int = 0
    content: bytes = b""
    headers: Dict[str, str] = field(default_factory=dict)

    # Performance and monitoring
    performance: PerformanceMetrics = field(
        default_factory=lambda: PerformanceMetrics(
            total_duration_ms=0.0,
            download_duration_ms=0.0,
            parsing_duration_ms=0.0,
            extraction_duration_ms=0.0,
            processing_duration_ms=0.0,
            bytes_downloaded=0,
            bytes_processed=0,
            documents_per_second=0.0,
            peak_memory_mb=0.0,
            avg_cpu_percent=0.0,
        )
    )

    # Status tracking
    status: ProcessingStatus = ProcessingStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Error handling
    errors: List[ErrorInfo] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Network details
    ip_address: Optional[str] = None
    user_agent: str = ""
    cookies: Dict[str, str] = field(default_factory=dict)

    # Content validation
    content_type: str = ""
    content_encoding: str = ""
    content_length: int = 0
    is_valid: bool = False
    robots_allowed: bool = True


@dataclass
class ExtractedContent:
    """Multi-modal content extraction result with comprehensive metadata."""

    # Core content
    text: str = ""
    title: str = ""
    language: str = "en"

    # Multi-modal content
    tables: List[Dict[str, Any]] = field(default_factory=list)
    code_blocks: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)

    # Structured data
    headings: List[Dict[str, Any]] = field(default_factory=list)
    lists: List[Dict[str, Any]] = field(default_factory=list)
    quotes: List[str] = field(default_factory=list)

    # Document structure
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0

    # Content quality indicators
    reading_time_minutes: float = 0.0
    lexical_diversity: float = 0.0
    readability_score: float = 0.0

    # Extraction metadata
    extraction_method: str = ""
    confidence_score: float = 0.0
    processing_time_ms: float = 0.0

    # Domain-specific data
    domain_data: Dict[str, Any] = field(default_factory=dict)

    # Error tracking
    extraction_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ContentMetadata:
    """Comprehensive metadata with OpenGraph, Schema.org, and social metrics."""

    # Basic metadata
    url: str = ""
    canonical_url: str = ""
    title: str = ""
    description: str = ""
    keywords: List[str] = field(default_factory=list)

    # OpenGraph metadata
    og_title: str = ""
    og_description: str = ""
    og_image: str = ""
    og_url: str = ""
    og_type: str = ""
    og_site_name: str = ""

    # Twitter Card metadata
    twitter_card: str = ""
    twitter_title: str = ""
    twitter_description: str = ""
    twitter_image: str = ""
    twitter_creator: str = ""

    # Schema.org structured data
    schema_type: str = ""
    schema_data: Dict[str, Any] = field(default_factory=dict)

    # Author and publication info
    author: str = ""
    author_url: str = ""
    published_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None

    # Domain and categorization
    domain: str = ""
    domain_type: DomainType = DomainType.GENERAL
    content_type: ContentType = ContentType.HTML

    # Social metrics
    social_shares: Dict[str, int] = field(default_factory=dict)
    social_engagement_score: float = 0.0

    # Technical metadata
    http_status: int = 200
    response_headers: Dict[str, str] = field(default_factory=dict)
    encoding: str = "utf-8"
    content_length: int = 0

    # Extraction metadata
    extraction_timestamp: datetime = field(default_factory=datetime.utcnow)
    extractor_version: str = "0.1.0"

    # --- Fields to be populated by MetadataExtractor ---
    authors: List[Any] = field(default_factory=list)
    publication_date: Optional[datetime] = None
    date_confidence: float = 0.0
    date_extraction_method: str = ""
    social_metrics: Optional[Dict[str, Any]] = None
    word_count: int = 0
    reading_time_minutes: float = 0.0
    lexical_diversity: float = 0.0
    content_categories: List[str] = field(default_factory=list)
    quality_indicators: Optional[Dict[str, Any]] = None
    dom_metrics: Optional[Any] = None
    quality_score: float = 0.0
    featured_image: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None
    processing_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DuplicationResult:
    """Result of 4-stage deduplication pipeline."""

    # Input identifiers
    content_id: UUID = field(default_factory=uuid4)
    content_hash: str = ""

    # Stage 1: Exact match (hash-based)
    exact_match: bool = False
    exact_match_id: Optional[UUID] = None

    # Stage 2: Bloom filter (approximate membership)
    bloom_match_probability: float = 0.0
    bloom_false_positive_rate: float = 0.01

    # Stage 3: MinHash LSH (Jaccard similarity)
    minhash_signature: Optional[np.ndarray] = None
    jaccard_similarity: float = 0.0
    jaccard_threshold: float = 0.8
    near_duplicates: List[UUID] = field(default_factory=list)

    # Stage 4: Semantic similarity (neural embeddings)
    embedding: Optional[np.ndarray] = None
    semantic_similarity: float = 0.0
    semantic_threshold: float = 0.9
    semantic_clusters: List[UUID] = field(default_factory=list)

    # Final decision
    is_duplicate: bool = False
    duplicate_type: str = ""  # exact, bloom, minhash, semantic
    confidence_score: float = 0.0

    # Performance metrics
    processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_time_ms: Optional[float] = None

    # Metadata
    deduplication_timestamp: datetime = field(default_factory=datetime.utcnow)
    deduplication_version: str = "0.1.0"


@dataclass
class QualityScore:
    """Comprehensive quality assessment with neural model outputs."""

    # Overall quality
    overall_score: float = 0.0  # 0-1 scale
    confidence: float = 0.0

    # Linguistic quality
    grammar_score: float = 0.0
    spelling_score: float = 0.0
    readability_score: float = 0.0
    coherence_score: float = 0.0

    # Content quality
    information_density: float = 0.0
    factual_accuracy: float = 0.0
    bias_score: float = 0.0  # 0 = no bias, 1 = highly biased
    toxicity_score: float = 0.0  # 0 = safe, 1 = toxic

    # Domain-specific quality
    domain_relevance: float = 0.0
    expertise_level: float = 0.0
    authority_score: float = 0.0

    # Neural model outputs
    perplexity: Optional[float] = None
    embedding_quality: Optional[float] = None
    classification_confidence: Optional[float] = None

    # Content characteristics
    length_score: float = 0.0  # Optimal length for domain
    structure_score: float = 0.0  # Document structure quality
    citation_score: float = 0.0  # Quality of references

    # Metadata
    model_version: str = "0.1.0"
    processing_time_ms: float = 0.0
    quality_timestamp: datetime = field(default_factory=datetime.utcnow)

    # Detailed breakdown
    quality_factors: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DatasetConfig:
    """Configuration for intelligent dataset construction with curriculum learning."""

    # Basic configuration
    name: str = ""
    version: str = "0.1.0"
    description: str = ""

    # Curriculum learning parameters
    enable_curriculum_learning: bool = True
    difficulty_progression: str = "linear"  # linear, exponential, custom
    difficulty_metrics: List[str] = field(default_factory=lambda: ["readability", "complexity", "length"])

    # Sampling strategy
    sampling_strategy: str = "balanced"  # balanced, weighted, random
    domain_weights: Dict[DomainType, float] = field(default_factory=dict)
    quality_threshold: float = 0.7
    max_samples_per_domain: int = 10000

    # Data distribution
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    temporal_split: bool = True  # Use time-based splitting

    # Content filtering
    min_word_count: int = 50
    max_word_count: int = 10000
    allowed_languages: Set[str] = field(default_factory=lambda: {"en"})
    excluded_domains: Set[str] = field(default_factory=set)

    # Format configuration
    output_format: str = "jsonl"  # jsonl, parquet, hf_dataset, tfrecord
    include_metadata: bool = True
    include_embeddings: bool = False
    compression: str = "zstd"

    # Augmentation settings
    enable_augmentation: bool = False
    augmentation_ratio: float = 0.1
    augmentation_methods: List[str] = field(default_factory=list)

    # Export settings
    chunk_size: int = 1000
    max_file_size_mb: int = 500
    parallel_writers: int = 4

    # Validation
    enable_validation: bool = True
    validation_checks: List[str] = field(default_factory=lambda: ["duplicates", "quality", "format"])


@dataclass
class HardwareCapabilities:
    """Hardware detection and capability assessment for adaptive processing."""

    # System identification
    hardware_type: HardwareType = HardwareType.LAPTOP
    system_id: str = ""

    # CPU capabilities
    cpu_cores: int = 1
    cpu_threads: int = 1
    cpu_frequency_ghz: float = 0.0
    cpu_architecture: str = ""

    # Memory configuration
    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    memory_type: str = ""  # DDR4, DDR5, LPDDR4, etc.

    # GPU capabilities
    has_gpu: bool = False
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    gpu_compute_capability: str = ""
    cuda_version: str = ""

    # Storage capabilities
    storage_type: str = ""  # SSD, HDD, NVMe
    storage_available_gb: float = 0.0
    storage_speed_mbps: float = 0.0

    # Network capabilities
    network_bandwidth_mbps: float = 0.0
    network_latency_ms: float = 0.0

    # Performance estimates
    estimated_throughput_docs_per_min: int = 0
    recommended_batch_size: int = 1
    recommended_workers: int = 1
    max_concurrent_requests: int = 10

    # Optimization flags
    enable_gpu_acceleration: bool = False
    enable_multiprocessing: bool = True
    enable_memory_mapping: bool = True
    enable_async_io: bool = True

    # Resource limits
    max_memory_usage_gb: float = 0.0
    max_gpu_memory_usage_gb: float = 0.0
    max_cpu_usage_percent: float = 80.0

    # Detection metadata
    detection_timestamp: datetime = field(default_factory=datetime.utcnow)
    detection_method: str = "automatic"
    confidence_score: float = 1.0


@dataclass
class SystemMetrics:
    """Real-time system performance metrics."""

    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    documents_in_flight: int = 0
    total_documents_processed: int = 0
    docs_per_minute: float = 0.0


# ============================================================================
# Protocol Interfaces for 8 Core Modules
# ============================================================================


class CrawlerProtocol(Protocol):
    """Protocol for web crawling with adaptive rate limiting and error handling."""

    async def crawl_url(
        self,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        respect_robots: bool = True,
    ) -> CrawlResult:
        """Crawl a single URL with comprehensive error handling."""
        ...

    async def crawl_batch(
        self,
        urls: List[str],
        *,
        concurrency: Optional[int] = None,
        rate_limit: Optional[float] = None,
        hardware_caps: Optional[HardwareCapabilities] = None,
    ) -> AsyncIterator[CrawlResult]:
        """Crawl multiple URLs with adaptive concurrency."""
        ...

    async def get_robots_txt(self, domain: str) -> Dict[str, Any]:
        """Fetch and parse robots.txt for domain."""
        ...

    async def adapt_to_hardware(self, capabilities: HardwareCapabilities) -> None:
        """Adapt crawler settings based on hardware capabilities."""
        ...

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics and statistics."""
        ...


class ExtractorProtocol(Protocol):
    """Protocol for content extraction with multi-modal support."""

    async def extract_content(
        self,
        crawl_result: CrawlResult,
        *,
        extract_tables: bool = True,
        extract_images: bool = True,
        extract_code: bool = True,
        extract_links: bool = True,
    ) -> ExtractedContent:
        """Extract multi-modal content from crawled data."""
        ...

    async def extract_batch(
        self,
        crawl_results: List[CrawlResult],
        *,
        hardware_caps: Optional[HardwareCapabilities] = None,
        parallel_workers: Optional[int] = None,
    ) -> AsyncIterator[ExtractedContent]:
        """Extract content from multiple crawl results."""
        ...

    async def detect_content_type(self, content: bytes) -> ContentType:
        """Detect content type from raw bytes."""
        ...

    async def validate_extraction(self, content: ExtractedContent) -> bool:
        """Validate extraction quality and completeness."""
        ...


class MetadataProtocol(Protocol):
    """Protocol for metadata extraction with OpenGraph and Schema.org support."""

    async def extract_metadata(
        self,
        crawl_result: CrawlResult,
        extracted_content: ExtractedContent,
    ) -> ContentMetadata:
        """Extract comprehensive metadata from content."""
        ...

    async def extract_social_metrics(
        self,
        url: str,
        metadata: ContentMetadata,
    ) -> Dict[str, int]:
        """Extract social engagement metrics."""
        ...

    async def classify_domain(self, url: str, content: str) -> DomainType:
        """Classify content domain type."""
        ...

    async def validate_metadata(self, metadata: ContentMetadata) -> bool:
        """Validate metadata completeness and accuracy."""
        ...


class DeduplicatorProtocol(Protocol):
    """Protocol for 4-stage deduplication pipeline."""

    async def check_duplicates(
        self,
        content: ExtractedContent,
        metadata: ContentMetadata,
        *,
        enable_semantic: bool = True,
        hardware_caps: Optional[HardwareCapabilities] = None,
    ) -> DuplicationResult:
        """Run complete 4-stage deduplication check."""
        ...

    async def build_bloom_filter(
        self,
        content_hashes: List[str],
        *,
        false_positive_rate: float = 0.01,
    ) -> None:
        """Build bloom filter for approximate membership testing."""
        ...

    async def compute_minhash(self, text: str) -> np.ndarray:
        """Compute MinHash signature for Jaccard similarity."""
        ...

    async def compute_embedding(
        self,
        text: str,
        *,
        use_gpu: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Compute semantic embedding for similarity search."""
        ...

    async def find_similar_content(
        self,
        embedding: np.ndarray,
        threshold: float = 0.9,
    ) -> List[Tuple[UUID, float]]:
        """Find semantically similar content using vector search."""
        ...


class QualityProtocol(Protocol):
    """Protocol for content quality assessment with neural models."""

    async def assess_quality(
        self,
        content: ExtractedContent,
        metadata: ContentMetadata,
        *,
        domain_specific: bool = True,
        use_neural_models: bool = True,
        hardware_caps: Optional[HardwareCapabilities] = None,
    ) -> QualityScore:
        """Comprehensive quality assessment."""
        ...

    async def assess_batch(
        self,
        content_batch: List[Tuple[ExtractedContent, ContentMetadata]],
        *,
        batch_size: int = 32,
        use_gpu: bool = True,
    ) -> AsyncIterator[QualityScore]:
        """Batch quality assessment for efficiency."""
        ...

    async def detect_bias(self, text: str) -> float:
        """Detect potential bias in content."""
        ...

    async def detect_toxicity(self, text: str) -> float:
        """Detect toxic content."""
        ...

    async def assess_domain_relevance(
        self,
        content: str,
        domain: DomainType,
    ) -> float:
        """Assess relevance to specific domain."""
        ...


class StorageProtocol(Protocol):
    """Protocol for hybrid storage with SQLite and Parquet."""

    async def store_crawl_result(self, result: CrawlResult) -> UUID:
        """Store crawl result with metadata indexing."""
        ...

    async def store_extracted_content(
        self,
        content: ExtractedContent,
        metadata: ContentMetadata,
        quality: QualityScore,
        dedup_result: DuplicationResult,
    ) -> UUID:
        """Store processed content with all associated data."""
        ...

    async def query_content(
        self,
        *,
        domain: Optional[DomainType] = None,
        quality_threshold: Optional[float] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[Tuple[ExtractedContent, ContentMetadata, QualityScore]]:
        """Query stored content with filtering."""
        ...

    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics and health metrics."""
        ...

    async def optimize_storage(self) -> None:
        """Optimize storage layout and indexes."""
        ...

    async def backup_data(self, backup_path: Path) -> None:
        """Create backup of all stored data."""
        ...


class DatasetProtocol(Protocol):
    """Protocol for intelligent dataset construction."""

    async def create_dataset(
        self,
        config: DatasetConfig,
        *,
        output_path: Path,
        hardware_caps: Optional[HardwareCapabilities] = None,
    ) -> Dict[str, Any]:
        """Create dataset with curriculum learning and export."""
        ...

    async def sample_content(
        self,
        config: DatasetConfig,
        available_content: List[Tuple[ExtractedContent, ContentMetadata, QualityScore]],
    ) -> List[Tuple[ExtractedContent, ContentMetadata, QualityScore]]:
        """Sample content according to configuration strategy."""
        ...

    async def format_for_training(
        self,
        content: ExtractedContent,
        metadata: ContentMetadata,
        format_type: str = "instruction",
    ) -> Dict[str, Any]:
        """Format content for specific training paradigm."""
        ...

    async def validate_dataset(
        self,
        dataset_path: Path,
        config: DatasetConfig,
    ) -> Dict[str, Any]:
        """Validate dataset quality and characteristics."""
        ...

    async def export_dataset(
        self,
        dataset: List[Dict[str, Any]],
        output_path: Path,
        format_type: str = "jsonl",
    ) -> None:
        """Export dataset in specified format."""
        ...


class ObservabilityProtocol(Protocol):
    """Protocol for monitoring, metrics, and observability."""

    async def start_monitoring(self) -> AsyncContextManager[None]:
        """Start monitoring context with resource tracking."""
        ...

    async def log_performance_metrics(
        self,
        component: str,
        metrics: PerformanceMetrics,
        correlation_id: Optional[UUID] = None,
    ) -> None:
        """Log performance metrics with correlation."""
        ...

    async def log_error(
        self,
        error: ErrorInfo,
        component: str,
        correlation_id: Optional[UUID] = None,
    ) -> None:
        """Log structured error information."""
        ...

    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        ...

    async def get_performance_report(
        self,
        time_range: Optional[Tuple[datetime, datetime]] = None,
    ) -> Dict[str, Any]:
        """Generate performance analysis report."""
        ...

    async def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        ...

    async def create_alert(
        self,
        condition: str,
        threshold: float,
        action: str,
    ) -> UUID:
        """Create monitoring alert with automatic actions."""
        ...


# ============================================================================
# Hardware Detection Protocol
# ============================================================================


class HardwareDetectorProtocol(Protocol):
    """Protocol for hardware capability detection and adaptation."""

    async def detect_capabilities(self) -> HardwareCapabilities:
        """Detect current system hardware capabilities."""
        ...

    async def benchmark_performance(
        self,
        test_duration_seconds: float = 10.0,
    ) -> Dict[str, float]:
        """Benchmark system performance for optimization."""
        ...

    async def optimize_for_hardware(
        self,
        capabilities: HardwareCapabilities,
    ) -> Dict[str, Any]:
        """Generate optimized configuration for hardware."""
        ...

    async def monitor_resources(self) -> AsyncIterator[Dict[str, float]]:
        """Monitor real-time resource utilization."""
        ...


# ============================================================================
# Utility Functions and Type Aliases
# ============================================================================

# Type aliases for common patterns
ContentTuple = Tuple[ExtractedContent, ContentMetadata, QualityScore]
ProcessingResult = Tuple[CrawlResult, ExtractedContent, ContentMetadata, DuplicationResult, QualityScore]
BatchProcessor = AsyncIterator[ProcessingResult]

# Vector operation types for numpy/polars integration
EmbeddingVector = np.ndarray
SimilarityMatrix = np.ndarray
FeatureDataFrame = pl.DataFrame


def create_correlation_id() -> UUID:
    """Create a new correlation ID for request tracking."""
    return uuid4()


def calculate_processing_efficiency(
    performance: PerformanceMetrics,
    hardware: HardwareCapabilities,
) -> float:
    """Calculate processing efficiency score based on hardware utilization."""
    if hardware.estimated_throughput_docs_per_min == 0:
        return 0.0

    actual_throughput = performance.documents_per_second * 60
    theoretical_max = hardware.estimated_throughput_docs_per_min

    return min(actual_throughput / theoretical_max, 1.0)


def estimate_gpu_batch_size(
    model_size_mb: float,
    available_gpu_memory_gb: float,
    safety_factor: float = 0.8,
) -> int:
    """Estimate optimal batch size for GPU processing."""
    available_mb = available_gpu_memory_gb * 1024 * safety_factor
    return max(1, int(available_mb // (model_size_mb * 4)))  # 4x for activation memory


def create_default_hardware_capabilities() -> HardwareCapabilities:
    """Create default hardware capabilities for fallback scenarios."""
    return HardwareCapabilities(
        hardware_type=HardwareType.LAPTOP,
        cpu_cores=4,
        cpu_threads=8,
        total_memory_gb=8.0,
        available_memory_gb=6.0,
        estimated_throughput_docs_per_min=100,
        recommended_batch_size=10,
        recommended_workers=4,
        max_concurrent_requests=20,
        max_memory_usage_gb=4.0,
        max_cpu_usage_percent=80.0,
    )


__all__ = [
    # Enums
    "ProcessingStatus",
    "ErrorSeverity",
    "ContentType",
    "DomainType",
    "HardwareType",
    # Core dataclasses
    "PerformanceMetrics",
    "ErrorInfo",
    "CrawlResult",
    "ExtractedContent",
    "ContentMetadata",
    "DuplicationResult",
    "QualityScore",
    "DatasetConfig",
    "HardwareCapabilities",
    "SystemMetrics",
    # Protocols
    "CrawlerProtocol",
    "ExtractorProtocol",
    "MetadataProtocol",
    "DeduplicatorProtocol",
    "QualityProtocol",
    "StorageProtocol",
    "DatasetProtocol",
    "ObservabilityProtocol",
    "HardwareDetectorProtocol",
    # Type aliases
    "ContentTuple",
    "ProcessingResult",
    "BatchProcessor",
    "EmbeddingVector",
    "SimilarityMatrix",
    "FeatureDataFrame",
    # Utilities
    "create_correlation_id",
    "calculate_processing_efficiency",
    "estimate_gpu_batch_size",
    "create_default_hardware_capabilities",
]
