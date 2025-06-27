"""
Database schema definition for QuarryCore's SQLite metadata store.
"""

from __future__ import annotations

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, MetaData, Table, Text
from sqlalchemy.sql import func

# Using a standard naming convention for database objects
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=convention)

# Table for tracking schema migrations
schema_version_table = Table(
    "schema_version",
    metadata,
    Column("version", Integer, primary_key=True),
    Column("updated_at", DateTime, server_default=func.now()),
)


processed_content_table = Table(
    "processed_content",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("content_id", Text, nullable=False, unique=True),
    Column("url", Text, nullable=False),
    Column("content_hash", Text, nullable=False, index=True),
    Column("processed_at", DateTime, server_default=func.now(), index=True),
    # Foreign key to warm storage
    Column("parquet_path", Text, nullable=False, comment="Path to the content in warm storage"),
    # Metadata
    Column("title", Text),
    Column("description", Text),
    Column("domain", Text, index=True),
    Column("author", Text),
    Column("published_date", DateTime),
    # Quality Scores
    Column("quality_score", Float, index=True),
    Column("is_duplicate", Boolean, default=False),
    Column("toxicity_score", Float),
    Column("coherence_score", Float),
    Column("grammar_score", Float),
    # Full metadata object for extensibility
    Column("full_metadata", JSON),
)

# FTS5 virtual table for full-text search on title and description
# Note: This is SQLite-specific syntax that will be executed directly.
# The table definition here is for ORM/Core mapping purposes.
content_fts_table = Table(
    "content_fts",
    metadata,
    Column("rowid", Integer, primary_key=True),
    Column("title", Text),
    Column("description", Text),
    Column("content", Text, default="processed_content"),  # Which table the content is from
    Column("content_rowid", Integer, default="id"),  # Column in content table to map to
)


# Indexes for common query patterns
Index("ix_processed_content_url", processed_content_table.c.url)
Index(
    "ix_processed_content_domain_quality",
    processed_content_table.c.domain,
    processed_content_table.c.quality_score,
)
