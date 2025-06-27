"""
Configuration management for QuarryCore using Pydantic.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import yaml
from pydantic import BaseModel, Field, ValidationError, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from quarrycore.protocols import DomainType

# --- Setup Logging ---
log = logging.getLogger(__name__)

# --- Nested Configuration Models ---


class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60


class RateLimiterConfig(BaseModel):
    max_requests: int = 100
    per_seconds: int = 60


class CrawlerConfig(BaseModel):
    user_agent: str = "QuarryCore/0.1 (AI Data Collector; +http://your-project-website.com)"
    max_concurrent_requests: int = 10
    request_delay_ms: int = 100
    timeout_seconds: int = 30
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    rate_limiter: RateLimiterConfig = Field(default_factory=RateLimiterConfig)
    max_depth: int = 3
    max_pages_per_domain: int = 1000


class SQLiteConfig(BaseModel):
    """Configuration for SQLite hot storage."""

    db_path: str = Field(default="./data/quarrycore.db", description="Path to the SQLite database file.")
    pool_size: int = Field(default=10, description="Size of the connection pool.")
    wal_mode: bool = Field(default=True, description="Enable Write-Ahead Logging for higher concurrency.")
    fts_version: Literal["fts5"] = Field(default="fts5", description="Full-Text Search module to use.")

    @validator("db_path", pre=True, always=True)
    def create_parent_dir(cls, v: str | Path) -> str:
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)


class ParquetConfig(BaseModel):
    """Configuration for Parquet warm storage."""

    base_path: str = Field(default="./data/parquet_store", description="Base path for Parquet files.")
    compression: Literal["snappy", "gzip", "brotli", "zstd"] = Field(
        default="snappy", description="Compression codec for Parquet files."
    )
    partition_by: list[str] = Field(
        default_factory=lambda: ["domain", "date"],
        description="Fields to partition data by.",
    )

    @validator("base_path", pre=True, always=True)
    def create_path(cls, v: str | Path) -> str:
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)


class RetentionConfig(BaseModel):
    """Configuration for data retention and archival."""

    archive_after_days: int | None = Field(
        default=90,
        description="Days after which to move warm data to cold storage. None to disable.",
    )
    cold_storage_path: str = Field(
        default="./data/cold_store",
        description="Path for Zstandard-compressed archives.",
    )

    @validator("cold_storage_path", pre=True, always=True)
    def create_path(cls, v: str | Path) -> str:
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)


class BackupConfig(BaseModel):
    """Configuration for database backups."""

    path: str = Field(default="./data/backups", description="Directory to store database backups.")
    frequency_hours: int | None = Field(
        default=24,
        description="How often to perform backups in hours. None to disable.",
    )

    @validator("path", pre=True, always=True)
    def create_path(cls, v: str | Path) -> str:
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)


class StorageConfig(BaseModel):
    """Configuration for the tiered storage system."""

    hot: SQLiteConfig = Field(default_factory=SQLiteConfig)
    warm: ParquetConfig = Field(default_factory=ParquetConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    backup: BackupConfig = Field(default_factory=BackupConfig)


class BloomFilterConfig(BaseModel):
    path: str = Field(default="./data/dedup/bloom_filter.bin")
    capacity: int = 1_000_000
    error_rate: float = 0.001

    @validator("path", pre=True, always=True)
    def create_parent_dir(cls, v: str | Path) -> str:
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)


class MinHashLSHConfig(BaseModel):
    path: str = Field(default="./data/dedup/minhash_lsh.pkl")
    threshold: float = 0.85
    num_perm: int = 128

    @validator("path", pre=True, always=True)
    def create_parent_dir(cls, v: str | Path) -> str:
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)


class SemanticDedupConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"
    index_path: str = Field(default="./data/dedup/faiss.index")
    threshold: float = 0.95
    batch_size: int = 32

    @validator("index_path", pre=True, always=True)
    def create_parent_dir(cls, v: str | Path) -> str:
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)


class FuzzyMatcherConfig(BaseModel):
    threshold: int = Field(default=90, ge=0, le=100)


class DeduplicationConfig(BaseModel):
    enabled_levels: list[int] = Field(default=[1, 2, 3])
    bloom_filter: BloomFilterConfig = Field(default_factory=BloomFilterConfig)
    minhash_lsh: MinHashLSHConfig = Field(default_factory=MinHashLSHConfig)
    semantic_dedup: SemanticDedupConfig = Field(default_factory=SemanticDedupConfig)
    fuzzy_matcher: FuzzyMatcherConfig = Field(default_factory=FuzzyMatcherConfig)
    domains: dict[DomainType, DomainQualityConfig] = Field(
        default_factory=dict, description="Domain-specific quality thresholds."
    )


class DomainQualityConfig(BaseModel):
    """Configuration for quality scoring for a specific domain."""

    min_overall_score: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description="Minimum overall quality score to accept content.",
    )
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "lexical": 0.15,
            "grammar": 0.15,
            "neural_coherence": 0.3,
            "toxicity": -0.5,  # Negative weight for toxicity
            "heuristic_info_density": 0.1,
            "heuristic_domain_relevance": 0.2,
            "heuristic_spam": -0.3,
        },
        description="Weights for aggregating the final score.",
    )


class QualityConfig(BaseModel):
    """Configuration for the quality assessment pipeline."""

    min_content_length: int = Field(default=50, description="Minimum word count to perform quality assessment.")
    max_content_length: int = Field(default=50000, description="Maximum word count to perform quality assessment.")
    default: DomainQualityConfig = Field(default_factory=DomainQualityConfig)
    domains: dict[DomainType, DomainQualityConfig] = Field(
        default_factory=dict, description="Domain-specific quality thresholds."
    )


class ChunkingConfig(BaseModel):
    """Configuration for token-aware chunking."""

    tokenizer_name: str = Field(
        default="mistralai/Mistral-7B-v0.1",
        description="HuggingFace tokenizer to use for chunking.",
    )
    chunk_size: int = Field(default=2048, description="Target size of each chunk in tokens.")
    chunk_overlap: int = Field(default=128, description="Number of tokens to overlap between chunks.")


class SamplingConfig(BaseModel):
    """Configuration for sampling strategy."""

    strategy: Literal["curriculum", "balanced", "random"] = Field(
        default="curriculum", description="Strategy for sampling and ordering data."
    )
    quality_weight: float = Field(
        default=0.7,
        description="Weight given to quality score during selection (0=no weight, 1=only quality).",
    )
    rejection_sampling_factor: float = Field(
        default=2.0,
        description="How many candidates to consider for each final slot (higher means stricter quality selection).",
    )
    domain_balance: dict[DomainType, float] | None = Field(
        default=None,
        description="Target proportions for each domain. If None, balance is equal.",
    )


class FormattingConfig(BaseModel):
    """Configuration for output formatting."""

    format_type: Literal["instruction", "conversation", "document"] = Field(
        default="instruction", description="The format for the final training data."
    )
    instruction_template: str = Field(
        default="Summarize the following text:\n\n{text}",
        description="Template for generating instructions.",
    )


class ExportConfig(BaseModel):
    """Configuration for dataset exporting."""

    formats: list[Literal["jsonl", "parquet", "huggingface"]] = Field(default=["jsonl", "parquet"])
    output_path: str = Field(default="./data/datasets", description="Base path for exported datasets.")
    huggingface_repo_id: str | None = Field(default=None, description="Optional HuggingFace Hub repo ID to push to.")

    @validator("output_path", pre=True, always=True)
    def create_data_path(cls, v: str | Path) -> str:
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)


class DatasetConfig(BaseModel):
    """Configuration for the intelligent dataset construction pipeline."""

    name: str = Field(default="quarrycore_dataset", description="Name of the dataset.")
    max_documents: int | None = Field(
        default=100000,
        description="Maximum number of documents to process for the dataset.",
    )

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    formatting: FormattingConfig = Field(default_factory=FormattingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)


class DomainConfig(BaseModel):
    medical: bool = True
    legal: bool = True
    ecommerce: bool = False
    technical: bool = True


class WebUIConfig(BaseModel):
    """Configuration for the real-time web UI."""

    enabled: bool = Field(default=True, description="Enable the FastAPI web server.")
    host: str = Field(default="127.0.0.1", description="Host for the web server.")
    port: int = Field(default=8000, description="Port for the web server.")


class MonitoringConfig(BaseModel):
    """Configuration for the observability and monitoring system."""

    enabled: bool = True
    log_level: str = Field(default="INFO", description="Logging level (e.g., DEBUG, INFO, WARNING).")
    log_file: str | None = Field(
        default="./logs/quarrycore.log",
        description="Path to log file. If None, logs to console.",
    )
    prometheus_port: int | None = Field(
        default=9090,
        description="Port for Prometheus metrics exporter. None to disable.",
    )
    web_ui: WebUIConfig = Field(default_factory=WebUIConfig)

    @validator("log_file", pre=True, always=True)
    def create_parent_dir(cls, v: str | Path | None) -> str | None:
        if v is None:
            return None
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)


class DebugConfig(BaseModel):
    test_mode: bool = False
    max_urls_to_process: int | None = 10000


# --- Main Configuration Class ---


class Config(BaseSettings):
    project_name: str = "QuarryCore"
    version: str = "0.1.0"
    crawler: CrawlerConfig = Field(default_factory=CrawlerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    domains: DomainConfig = Field(default_factory=DomainConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)

    model_config = SettingsConfigDict(env_prefix="QUARRY_", env_nested_delimiter="__", case_sensitive=False)

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        log.debug("Loading configuration from YAML file: %s", path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found or is not a file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        if not yaml_data:
            log.warning("Configuration file is empty: %s. Using default settings.", path)
            return cls.model_validate({})
        return cls.model_validate(yaml_data or {})


def find_config_file() -> Path | None:
    current_dir = Path.cwd()
    paths_to_check = [
        current_dir / "config.yaml",
        current_dir / "config.yml",
    ]
    for path in paths_to_check:
        if path.exists():
            return path

    example_path = current_dir / "config.example.yaml"
    if example_path.exists():
        return example_path

    return None


# --- Lazy Configuration Loader ---


class LazyConfig:
    """
    A proxy for the Config object that delays its loading and validation
    until an attribute is first accessed. This prevents configuration errors
    from crashing the application on import.
    """

    _config: ClassVar[Config | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        if self.__class__._config is None:
            with self.__class__._lock:
                if self.__class__._config is None:
                    self.__class__._config = self._load_config_with_fallback()
        return getattr(self.__class__._config, name)

    def _load_config_with_fallback(self) -> Config:
        """Load configuration from file or fall back to defaults."""
        config_path = find_config_file()
        if config_path:
            try:
                log.info("Lazy loading configuration from: %s", config_path)
                return Config.from_yaml(config_path)
            except (ValidationError, FileNotFoundError, Exception) as e:
                log.error(
                    "Failed to load or validate configuration from '%s': %s. "
                    "Falling back to default settings. Please check your config file.",
                    config_path,
                    e,
                    exc_info=log.getEffectiveLevel() <= logging.DEBUG,
                )
        else:
            log.info("No config file found. Using default settings for lazy load.")

        # Fallback to default settings
        try:
            return Config()
        except ValidationError as e:
            log.critical("FATAL: Default configuration is invalid: %s", e, exc_info=True)
            # This is a critical failure. We cannot proceed.
            raise RuntimeError(f"Default configuration is invalid, cannot start: {e}") from e


# --- Global Settings Instance ---
# Use a type hint with a forward reference to Config to help type checkers,
# while the actual instance is the LazyConfig proxy.
settings: "Config" = cast("Config", LazyConfig())
