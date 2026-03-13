# Changelog

All notable changes to this project are documented in this file.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0] - 2026-03-13

### Added
- Predefined prefixes used during the preprocessing step.
- MCP authentication support.
- Langfuse-based evaluation setup.
- Database-backed configuration persistence and question answering using DB-stored configuration.
- SPARQL chat history update and deletion capabilities.
- Quota-aware feature usage controls.

### Changed
- Scenario architecture refactored to class-based structure with dedicated configuration handling.
- Repository refresh and metadata update for the v2 release.
- API documentation improvements.

### Fixed
- `db_key` loading behavior.
- Fallback to KG endpoint when ontology endpoint is not specified in configuration.
- API documentation blank page issue.
- Missing user fields and KG configuration fields.
- Warning message quality and consistency.

## [1.0] - 2025-05-12

### Added
- Initial tagged release of Gen²KGBot.

---

> Note: entries are summarized from git history and may be refined over time.
