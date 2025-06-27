"""
Main orchestrator for the tiered storage system.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Tuple
from uuid import UUID

from quarrycore.config.config import StorageConfig
from quarrycore.protocols import (
    ContentMetadata,
    CrawlResult,
    DuplicationResult,
    ExtractedContent,
    QualityScore,
    StorageProtocol,
)

from .parquet_store import ParquetStore
from .sqlite_manager import SQLiteManager

if TYPE_CHECKING:
    from pathlib import Path


class StorageManager(StorageProtocol):
    """
    Implements the StorageProtocol and orchestrates hot (SQLite) and
    warm (Parquet) storage tiers.
    """

    def __init__(self, config: StorageConfig) -> None:
        self.config = config
        self.sqlite = SQLiteManager(config.hot)
        self.parquet = ParquetStore(config.warm)

    async def initialize(self) -> None:
        """Initializes all underlying storage managers."""
        await self.sqlite.initialize()

    async def store_crawl_result(self, result: CrawlResult) -> UUID:
        # For now, we only store the final processed content.
        # This could be implemented to store raw crawl data if needed.
        raise NotImplementedError("Storing raw crawl results is not yet implemented.")

    async def store_extracted_content(
        self,
        content: ExtractedContent,
        metadata: ContentMetadata,
        quality: QualityScore,
        dedup_result: DuplicationResult,
    ) -> UUID:
        """
        Stores processed content across the tiered storage system with atomic operations.
        1. Content -> Warm Storage (Parquet)
        2. Metadata -> Hot Storage (SQLite)

        Implements rollback capability for multi-tier consistency.
        """
        content_id = dedup_result.content_id
        parquet_path = None

        try:
            # 1. Store main content in Parquet store
            parquet_data = [
                {
                    "content_id": str(content_id),
                    "text": content.text,
                    "title": content.title,
                    "html": "",  # Not storing raw HTML by default to save space
                }
            ]
            parquet_path = await self.parquet.write(parquet_data)

            # 2. Prepare and store metadata in SQLite
            metadata_record = {
                "content_id": str(content_id),
                "url": metadata.url,
                "content_hash": dedup_result.content_hash,
                "parquet_path": str(parquet_path),  # Ensure string conversion
                "title": content.title,
                "description": metadata.description,
                "domain": metadata.domain,
                "author": metadata.author,
                "published_date": metadata.published_date,
                "quality_score": quality.overall_score,
                "is_duplicate": dedup_result.is_duplicate,
                "toxicity_score": quality.toxicity_score,
                "coherence_score": quality.coherence_score,
                "grammar_score": quality.grammar_score,
                "full_metadata": json.dumps(self._serialize_metadata(metadata)),
            }

            await self.sqlite.store_batch([metadata_record])
            return content_id

        except Exception as e:
            # Rollback: Clean up parquet file if SQLite storage failed
            if parquet_path:
                try:
                    from pathlib import Path

                    full_parquet_path = self.config.warm.base_path / Path(parquet_path)
                    if full_parquet_path.exists():
                        full_parquet_path.unlink()
                except Exception:
                    pass  # Best effort cleanup
            raise e

    def _serialize_metadata(self, metadata: ContentMetadata, *args: Any) -> Dict[str, Any]:
        """Serialize metadata to dictionary format."""
        return {
            "url": metadata.url,
            "title": metadata.title or "",
            "author": metadata.author or "",
            "publication_date": metadata.publication_date.isoformat() if metadata.publication_date else None,
            "language": getattr(metadata, "language", "") or "",
            "word_count": metadata.word_count,
            "domain_type": metadata.domain_type.value if metadata.domain_type else None,
        }

    async def query_content(self, **kwargs) -> AsyncIterator[Tuple[ExtractedContent, ContentMetadata, QualityScore]]:
        # This would involve querying SQLite first, then retrieving from Parquet
        raise NotImplementedError("Querying content is not yet implemented.")

    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            total_content = await self.sqlite.get_total_count()  # type: ignore
        except AttributeError:
            total_content = 0
        recent_count = len(await self.sqlite.get_recent_content(limit=24))

        stats: Dict[str, Any] = {
            "total_content": total_content,
            "recent_content_24h": recent_count,
            "storage_backends": ["sqlite", "parquet"],
        }

        return stats

    async def optimize_storage(self) -> None:
        """Performs optimization tasks like VACUUM on the database."""
        await self.sqlite.vacuum()

    async def backup_data(self, backup_path: Path) -> None:
        """Backs up the hot storage tier."""
        await self.sqlite.backup(backup_path)

    async def close(self) -> None:
        """Closes all storage connections."""
        await self.sqlite.close()

    async def store_document(self, document: Dict[str, Any]) -> UUID:
        """
        Protocol compliance method - stores a document.

        This method exists to match validation expectations.
        Stores document data in the appropriate storage tier.
        """
        import uuid

        from quarrycore.protocols import ContentMetadata, DuplicationResult, ExtractedContent, QualityScore

        # Generate a document ID
        doc_id = uuid.uuid4()

        # Extract components from the document dict
        content = ExtractedContent(text=document.get("text", ""), title=document.get("metadata", {}).get("title", ""))

        metadata = ContentMetadata(
            url=document.get("metadata", {}).get("source", f"doc_{doc_id}"),
            title=document.get("metadata", {}).get("title", ""),
            domain=document.get("metadata", {}).get("source", "unknown"),
        )

        quality = QualityScore(overall_score=document.get("metadata", {}).get("quality_score", 0.0))

        dedup_result = DuplicationResult(
            content_id=doc_id, content_hash=str(hash(document.get("text", ""))), is_duplicate=False
        )

        # Use the existing store_extracted_content method
        return await self.store_extracted_content(content, metadata, quality, dedup_result)
