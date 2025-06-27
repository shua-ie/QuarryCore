# Pipeline Checkpointing and Recovery Guide

This guide explains QuarryCore's robust checkpointing and recovery system, including failure handling and dead letter queue semantics.

## Overview

QuarryCore implements production-grade checkpointing that enables:
- **Fault tolerance** - Resume processing after interruptions
- **Progress persistence** - Never lose processing progress
- **Atomic operations** - Cross-platform safe checkpoint storage
- **Dead letter handling** - Systematic failure tracking and analysis

## Checkpoint System

### Automatic Checkpointing

The pipeline automatically saves checkpoints at regular intervals:

```python
from quarrycore import Pipeline, PipelineSettings

# Configure checkpoint behavior
settings = PipelineSettings(
    checkpoint_interval=60.0,                    # Save every 60 seconds
    checkpoint_dir=Path("/app/checkpoints"),     # Custom location
)

pipeline = Pipeline(container, settings=settings)
```

### Environment Configuration

Set checkpoint behavior via environment variables for production:

```bash
# Checkpoint configuration
export CHECKPOINT_INTERVAL=30.0              # More frequent saves
export CHECKPOINT_DIR=/persistent/checkpoints # Persistent storage

# Run pipeline
python -m quarrycore.pipeline --config production.yaml
```

### Checkpoint Structure

Checkpoints are stored as JSON files with complete state information:

```json
{
  "job_id": "job-2024-01-15-12-30-45",
  "pipeline_id": "pipeline-abc123",
  "stage": "quality",
  "processed_count": 1247,
  "failed_count": 23,
  "start_time": 1705312245.123,
  "last_checkpoint": 1705312845.456,
  "urls_remaining": [
    "https://example.com/page1",
    "https://example.com/page2"
  ],
  "batch_size": 50,
  "error_count_by_stage": {
    "crawl": 10,
    "extract": 8,
    "quality": 5
  }
}
```

## Resume Behavior

### Exact-State Resume (AC-03)

When resuming from a checkpoint:

1. **Complete checkpoint**: If `urls_remaining` is empty, the pipeline exits immediately with `status: "completed"`
2. **Partial checkpoint**: Processing resumes from exactly where it left off
3. **State preservation**: All counters and stage information are maintained

```python
# Resume from specific checkpoint
result = await pipeline.run(
    urls=[],  # Empty - resuming from checkpoint
    resume_from=Path("checkpoints/job-123.json")
)

if result["status"] == "completed":
    print(f"All {result['processed_count']} URLs already processed")
else:
    print(f"Resumed and processed {result['processed_count']} total URLs")
```

## Safe Filename Generation (AC-02)

Job IDs are automatically slugified for cross-platform compatibility:

| Original Job ID | Safe Filename |
|----------------|---------------|
| `job:2024/01/01-12:30:45` | `job-2024-01-01-12-30-45.json` |
| `pipeline\\batch#123` | `pipeline-batch-123.json` |
| `My Job (v2.1)` | `my-job-v2-1.json` |

Unsafe characters (`/`, `\`, `:`, `<`, `>`, `|`) are replaced with hyphens.

## Atomic Checkpoint Storage (AC-01)

Checkpoints use atomic write operations for reliability:

```python
# Cross-platform atomic write process:
# 1. Write to temporary file in same directory
# 2. Attempt os.replace() (atomic on POSIX/Windows)
# 3. Fallback to shutil.move() if needed
# 4. Clean up on any failure

from quarrycore.utils import atomic_write_json

data = {"job_id": "test", "processed": 100}
atomic_write_json(Path("checkpoint.json"), data)
```

This ensures:
- **No partial writes** - Checkpoint is complete or doesn't exist
- **Cross-platform** - Works on Linux, Windows, macOS
- **Concurrent safe** - Multiple processes won't corrupt files

## Dead Letter Queue System

### Purpose

The dead letter queue systematically tracks failed URLs for analysis and potential retry:

```python
# Failed URLs are automatically added to dead_letter.db
# with complete failure information
```

### Duplicate Handling (AC-04)

The dead letter queue enforces unique constraints and upsert behavior:

**Database Schema:**
```sql
CREATE TABLE failed_documents (
    -- ... other fields ...
    url TEXT NOT NULL,
    failure_stage TEXT NOT NULL,
    failure_count INTEGER DEFAULT 1,
    -- ... other fields ...
    UNIQUE(url, failure_stage)
)
```

**Upsert Behavior:**
- First failure for URL+stage: Creates new record with `failure_count = 1`
- Subsequent identical failures: Increments `failure_count` and updates `last_failure_time`
- Different stages: Separate records (e.g., URL fails at both crawl and extract stages)

### Example Dead Letter Records

```python
# Same URL, same stage = upsert (increment count)
await dlq.add_failed_document("https://example.com", "crawl", error_info)
await dlq.add_failed_document("https://example.com", "crawl", error_info)
# Result: 1 record with failure_count = 2

# Same URL, different stage = separate records  
await dlq.add_failed_document("https://example.com", "crawl", error_info)
await dlq.add_failed_document("https://example.com", "extract", error_info) 
# Result: 2 separate records
```

## Domain-Based Backpressure (AC-05)

### Failure Tracking

The pipeline tracks failures per domain to prevent overwhelming failing sites:

```python
# Configuration
settings = PipelineSettings(
    domain_failure_threshold=5,      # Max failures per domain
    domain_failure_window=60.0,      # In 60 seconds
    domain_backoff_duration=120.0    # Backoff for 2 minutes
)
```

### Backpressure Behavior

When a domain exceeds the failure threshold:

1. **Detection**: More than 5 failures for `example.com` in 60 seconds
2. **Backoff**: All URLs for `example.com` are paused for 120 seconds  
3. **Isolation**: Other domains continue processing normally
4. **Recovery**: After backoff expires, processing resumes

```python
# Example timeline:
# 10:00:00 - First failure for example.com
# 10:00:30 - 5th failure for example.com → BACKOFF TRIGGERED
# 10:00:31 - URLs for example.com skipped
# 10:02:30 - Backoff expires, processing resumes
```

## Signal Handling (AC-07)

### Graceful Shutdown

The pipeline handles SIGINT and SIGTERM gracefully:

```python
# Signal handling process:
# 1. Receive SIGINT/SIGTERM
# 2. Set shutdown flag
# 3. Save immediate checkpoint
# 4. Stop accepting new work
# 5. Complete current tasks
# 6. Exit with code 0
```

### Docker Integration

Works correctly with Docker's `--init` flag:

```bash
# Graceful shutdown in Docker
docker run --init quarrycore:latest python -m quarrycore.pipeline

# Send SIGINT for graceful stop
docker kill --signal=SIGINT <container_id>
```

## Production Best Practices

### 1. Checkpoint Storage

```bash
# Use persistent volumes for checkpoints
mkdir -p /app/persistent/checkpoints
export CHECKPOINT_DIR=/app/persistent/checkpoints

# Ensure directory permissions
chmod 755 /app/persistent/checkpoints
```

### 2. Monitoring

```python
# Monitor checkpoint health
checkpoint_age = time.time() - checkpoint.last_checkpoint
if checkpoint_age > settings.checkpoint_interval * 2:
    logger.warning("Checkpoint may be stale", age=checkpoint_age)
```

### 3. Cleanup

```bash
# Clean old checkpoints (completed jobs)
find /app/checkpoints -name "*.json" -mtime +7 -delete
```

### 4. Dead Letter Analysis

```python
# Analyze failed URLs for patterns
dlq = DeadLetterQueue()
stats = await dlq.get_failure_statistics()

print(f"Total failures: {stats['total_failures']}")
print(f"By stage: {stats['failures_by_stage']}")
print(f"By error: {stats['failures_by_error']}")
```

## Recovery Scenarios

### Scenario 1: System Crash During Processing

```python
# System crashes at 50% completion
# On restart:
result = await pipeline.run(
    urls=[],  # Empty URLs
    resume_from=Path("checkpoints/job-abc.json")
)
# → Continues from 50%, processes remaining 50%
```

### Scenario 2: Planned Maintenance

```python
# Send SIGINT for graceful shutdown
# Pipeline saves checkpoint and exits cleanly
# After maintenance:
result = await pipeline.run(urls=[], resume_from=checkpoint_path)
# → Resumes exactly where it left off
```

### Scenario 3: All URLs Processed

```python
# Resume from completed checkpoint
result = await pipeline.run(urls=[], resume_from=completed_checkpoint)
# → result["status"] == "completed", duration = 0
```

## Troubleshooting

### Common Issues

**Checkpoint not saving:**
- Check directory permissions
- Verify disk space
- Check logs for atomic write errors

**Resume not working:**
- Verify checkpoint file exists and is valid JSON
- Check job_id format for unsafe characters
- Ensure urls_remaining is not empty

**Dead letter queue issues:**
- Check database file permissions
- Verify SQLite version compatibility
- Monitor database size growth

### Debug Commands

```bash
# Validate checkpoint file
python -c "
import json
from pathlib import Path
from quarrycore.pipeline import PipelineCheckpoint

data = json.loads(Path('checkpoint.json').read_text())
checkpoint = PipelineCheckpoint.model_validate(data)
print(f'Valid checkpoint: {checkpoint.job_id}')
"

# Check dead letter queue
python -c "
import asyncio
from quarrycore.recovery.dead_letter import DeadLetterQueue

async def check_dlq():
    dlq = DeadLetterQueue()
    await dlq.initialize()
    stats = await dlq.get_failure_statistics()
    print(f'Failed documents: {stats}')
    await dlq.close()

asyncio.run(check_dlq())
"
```

## Configuration Reference

### PipelineSettings Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `checkpoint_interval` | float | 60.0 | Seconds between automatic checkpoints |
| `checkpoint_dir` | Path | "checkpoints" | Directory for checkpoint storage |
| `domain_failure_threshold` | int | 5 | Max failures per domain before backoff |
| `domain_failure_window` | float | 60.0 | Time window for failure tracking |
| `domain_backoff_duration` | float | 120.0 | Backoff duration for failing domains |
| `dead_letter_db_path` | Path | "dead_letter.db" | Dead letter queue database file |

### Environment Variables

| Variable | Example | Description |
|----------|---------|-------------|
| `CHECKPOINT_INTERVAL` | "30.0" | Override checkpoint interval |
| `CHECKPOINT_DIR` | "/app/checkpoints" | Override checkpoint directory |
| `DOMAIN_FAILURE_THRESHOLD` | "10" | Override failure threshold |
| `DOMAIN_FAILURE_WINDOW` | "120.0" | Override failure window |
| `DOMAIN_BACKOFF_DURATION` | "300.0" | Override backoff duration |
| `DEAD_LETTER_DB_PATH` | "/app/dead_letter.db" | Override dead letter DB path | 