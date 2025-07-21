# Changelog

All notable changes to QuarryCore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- feat(extraction): Introduce ExtractorManager for cascading content extraction with quality gating
  - Configurable extractor cascade order
  - Quality threshold filtering (default 0.6)
  - Domain-specific extractor overrides
  - Performance metrics tracking
  - Resilient error handling with automatic fallback

### Changed
- Pipeline now uses ExtractorManager instead of basic HTML parsing
- Quality assessment integrated into extraction phase for early filtering

### Configuration
- New `extraction` config section with `cascade_order`, `quality_threshold`, and `domain_overrides` 