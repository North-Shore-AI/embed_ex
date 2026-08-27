# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-01-25

### Added

- **Ollama Provider** - New provider for local embeddings via Ollama
  - Full implementation of `EmbedEx.Provider` behaviour
  - Support for `nomic-embed-text`, `mxbai-embed-large`, `all-minilm`, and `snowflake-arctic-embed` models
  - Native batch processing using Ollama's `/api/embed` endpoint
  - Health check function (`health_check/1`) to verify Ollama availability
  - Model availability checking (`model_available?/2`)
  - List available models (`list_models/1`, `list_embedding_models/1`)
  - Model dimensions lookup (`model_dimensions/1`)
  - Configurable host via `:host` option, `OLLAMA_HOST` env var, or application config

- **PHI-Safe Clinical Embeddings** - Local inference support for medical/clinical applications
  - All data stays on-premises when using Ollama provider
  - No API keys or external services required
  - Suitable for Protected Health Information (PHI) processing

- **Documentation**
  - Comprehensive Ollama Provider Guide (`guides/ollama_provider.md`)
  - Updated README with Ollama provider section and examples
  - Example script for Ollama usage (`examples/ollama_embeddings.exs`)
  - Examples README and `run_all.sh` script

- **Provider Updates**
  - Added `:ollama` to main `EmbedEx.providers/0` list
  - Added `:ollama` routing in `EmbedEx` and `EmbedEx.Batch`
  - Support for `:ollama` as default provider

### Changed

- Bumped version to 0.2.0
- Updated roadmap in README to reflect current state
- Enhanced documentation configuration in mix.exs with module groups

### Documentation

- Added `guides/ollama_provider.md` - Comprehensive guide for Ollama provider
- Updated `README.md` with Ollama provider documentation
- Added `examples/ollama_embeddings.exs` - Example usage for Ollama
- Added `examples/README.md` - Documentation for examples
- Added `examples/run_all.sh` - Script to run all examples

## [0.1.0] - 2024-12-06

### Added

- Initial release
- **Core Features**
  - Unified embedding API across multiple providers
  - `EmbedEx.Embedding` struct for embedding representation
  - `EmbedEx.Provider` behaviour for provider implementations

- **Providers**
  - OpenAI provider (`EmbedEx.Providers.OpenAI`)
  - Cohere provider (`EmbedEx.Providers.Cohere`)
  - Voyage AI provider (`EmbedEx.Providers.Voyage`)

- **Batch Processing** (`EmbedEx.Batch`)
  - Automatic chunking based on provider limits
  - Parallel processing with configurable concurrency
  - Progress callbacks for long-running operations
  - Streaming support for large datasets

- **Caching** (`EmbedEx.Cache`)
  - Cachex-based caching layer
  - SHA256-based cache keys (text + model + provider + dimensions)
  - TTL support (default: 24 hours)
  - Size limits (default: 10,000 entries)
  - Statistics tracking

- **Similarity Operations** (`EmbedEx.Similarity`)
  - Cosine similarity
  - Euclidean distance
  - Dot product
  - Top-k similar search
  - Pairwise similarity matrix
  - Nx-based GPU acceleration support

- **Documentation**
  - Comprehensive README
  - HexDocs documentation
  - Provider comparison guide
  - Performance tuning guide

[Unreleased]: https://github.com/North-Shore-AI/embed_ex/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/North-Shore-AI/embed_ex/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/North-Shore-AI/embed_ex/releases/tag/v0.1.0
