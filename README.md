<p align="center">
  <img src="assets/embed_ex.svg" alt="EmbedEx" width="200">
</p>

<h1 align="center">EmbedEx</h1>

<p align="center">
  <a href="https://github.com/North-Shore-AI/embed_ex/actions"><img src="https://github.com/North-Shore-AI/embed_ex/workflows/CI/badge.svg" alt="CI Status"></a>
  <a href="https://hex.pm/packages/embed_ex"><img src="https://img.shields.io/hexpm/v/embed_ex.svg" alt="Hex.pm"></a>
  <a href="https://hexdocs.pm/embed_ex"><img src="https://img.shields.io/badge/docs-hexdocs-blue.svg" alt="Documentation"></a>
  <img src="https://img.shields.io/badge/elixir-%3E%3D%201.14-purple.svg" alt="Elixir">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
</p>

<p align="center">
  Vector embeddings service with multiple providers and similarity search
</p>

---

Vector embeddings service for the NSAI (North Shore AI) ecosystem. A unified interface for generating and working with text embeddings across multiple providers with built-in caching, batch processing, and similarity computations.

## Features

- **Multiple Provider Support** - OpenAI, Cohere, Voyage, Ollama (local) with unified API
- **PHI-Safe Local Inference** - Ollama provider for privacy-sensitive data that never leaves your environment
- **Automatic Caching** - ETS-based cache with TTL and optional Redis backend
- **Batch Processing** - Efficient parallel processing with automatic chunking
- **Similarity Computations** - Cosine similarity, Euclidean distance, dot product using Nx
- **GPU Acceleration** - Optional GPU support via Nx when available
- **Progress Tracking** - Built-in progress callbacks for long-running operations

## Installation

Add `embed_ex` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:embed_ex, "~> 0.2.0"}
  ]
end
```

## Quick Start

```elixir
# Single embedding
{:ok, embedding} = EmbedEx.embed("Hello world", provider: :openai)

# Batch embeddings
{:ok, embeddings} = EmbedEx.embed_batch([
  "First text",
  "Second text",
  "Third text"
], provider: :openai)

# Compute similarity
similarity = EmbedEx.cosine_similarity(embedding1, embedding2)
# => 0.87

# Find similar embeddings
{:ok, results} = EmbedEx.find_similar(
  query_embedding,
  corpus_embeddings,
  top_k: 5
)
# => [{0.95, 0}, {0.87, 2}, {0.82, 5}, {0.79, 1}, {0.75, 8}]
```

## Configuration

```elixir
# config/config.exs

config :embed_ex,
  default_provider: :openai

config :embed_ex, :cache,
  enabled: true,
  ttl: :timer.hours(24),
  limit: 10_000

# Provider configuration
config :embed_ex, :openai,
  api_key: System.get_env("OPENAI_API_KEY"),
  default_model: "text-embedding-3-small"
```

Environment variables:
- `OPENAI_API_KEY` - Your OpenAI API key (required for OpenAI provider)
- `OLLAMA_HOST` - Ollama server URL (default: `http://localhost:11434`)

## Usage

### Single Embeddings

```elixir
# Using default provider (OpenAI)
{:ok, embedding} = EmbedEx.embed("Hello world")

# Specifying provider and model
{:ok, embedding} = EmbedEx.embed(
  "Hello world",
  provider: :openai,
  model: "text-embedding-3-large"
)

# Disable caching for this request
{:ok, embedding} = EmbedEx.embed("Hello world", use_cache: false)
```

### Batch Embeddings

```elixir
texts = ["Text 1", "Text 2", "Text 3", ...]

# Basic batch embedding
{:ok, embeddings} = EmbedEx.embed_batch(texts, provider: :openai)

# With progress tracking
{:ok, embeddings} = EmbedEx.embed_batch(
  texts,
  provider: :openai,
  on_progress: fn completed, total ->
    IO.puts("Progress: #{completed}/#{total}")
  end
)

# Control concurrency and batch size
{:ok, embeddings} = EmbedEx.embed_batch(
  texts,
  provider: :openai,
  batch_size: 100,
  concurrency: 10
)
```

### Similarity Computations

```elixir
# Cosine similarity (returns -1 to 1, where 1 is identical)
similarity = EmbedEx.cosine_similarity(embedding1, embedding2)

# Euclidean distance (lower is more similar)
distance = EmbedEx.euclidean_distance(embedding1, embedding2)

# Dot product
dot = EmbedEx.dot_product(embedding1, embedding2)

# Find top-k most similar
{:ok, results} = EmbedEx.find_similar(
  query_embedding,
  corpus_embeddings,
  top_k: 5,
  metric: :cosine
)

# With similarity threshold
{:ok, results} = EmbedEx.find_similar(
  query_embedding,
  corpus_embeddings,
  top_k: 10,
  threshold: 0.8
)

# Pairwise similarity matrix
matrix = EmbedEx.pairwise_similarity([emb1, emb2, emb3], metric: :cosine)
# Returns Nx.Tensor of shape {3, 3}
```

### Cache Management

```elixir
# Clear all cached embeddings
{:ok, count} = EmbedEx.clear_cache()

# Get cache statistics
{:ok, stats} = EmbedEx.cache_stats()
# => %{hits: 150, misses: 50, ...}
```

## Providers

### OpenAI

Supports OpenAI's embedding models via their API.

**Supported Models:**
- `text-embedding-3-small` (default) - 1536 dimensions, cost-effective
- `text-embedding-3-large` - 3072 dimensions, highest quality
- `text-embedding-ada-002` - 1536 dimensions, legacy model

**Configuration:**
```elixir
{:ok, embedding} = EmbedEx.embed(
  "Hello world",
  provider: :openai,
  model: "text-embedding-3-large",
  api_key: "sk-..." # Optional, defaults to OPENAI_API_KEY env var
)
```

**Batch Limits:**
- Max batch size: 2048 texts per request
- Automatic chunking for larger batches

### Ollama (Local)

Local embeddings via [Ollama](https://ollama.ai/) for PHI-safe processing. No data leaves your environment.

**Supported Models:**
- `nomic-embed-text` (default) - 768 dimensions, general purpose
- `mxbai-embed-large` - 1024 dimensions, higher quality
- `all-minilm` - 384 dimensions, fast and lightweight
- `snowflake-arctic-embed` - 1024 dimensions, high performance

**Prerequisites:**
```bash
# Install Ollama
brew install ollama  # macOS
# or curl -fsSL https://ollama.ai/install.sh | sh  # Linux

# Start server and pull model
ollama serve
ollama pull nomic-embed-text
```

**Configuration:**
```elixir
# config/config.exs
config :embed_ex, :ollama,
  host: "http://localhost:11434",
  default_model: "nomic-embed-text"
```

**Usage:**
```elixir
# Single embedding
{:ok, embedding} = EmbedEx.embed("Hello world", provider: :ollama)

# With specific model
{:ok, embedding} = EmbedEx.embed("Hello world",
  provider: :ollama,
  model: "mxbai-embed-large"
)

# Batch embedding
{:ok, embeddings} = EmbedEx.embed_batch(texts, provider: :ollama)

# PHI-safe clinical text processing
{:ok, embedding} = EmbedEx.embed(
  "Patient presents with acute chest pain",
  provider: :ollama
)
```

**Batch Limits:**
- Max batch size: 512 texts per request
- Native batch processing (Ollama's `/api/embed` endpoint)

See the [Ollama Provider Guide](guides/ollama_provider.md) for detailed documentation.

### Cohere

Supports Cohere's embedding models with multilingual support.

**Supported Models:**
- `embed-english-v3.0` (default) - English embeddings
- `embed-multilingual-v3.0` - 100+ languages
- `embed-english-light-v3.0` - Faster, lighter English model

### Voyage AI

Supports Voyage AI's high-performance and domain-specific models.

**Supported Models:**
- `voyage-3` (default) - General purpose, 1024 dimensions
- `voyage-code-3` - Optimized for code
- `voyage-finance-2` - Domain-specific for finance
- `voyage-law-2` - Domain-specific for legal text

## Architecture

```
embed_ex/
├── lib/
│   └── embed_ex/
│       ├── embedding.ex        # Embedding struct and utilities
│       ├── provider.ex         # Provider behaviour
│       ├── providers/
│       │   ├── openai.ex       # OpenAI implementation
│       │   ├── cohere.ex       # Cohere implementation
│       │   ├── voyage.ex       # Voyage AI implementation
│       │   └── ollama.ex       # Ollama (local) implementation
│       ├── cache.ex            # Caching layer (Cachex)
│       ├── similarity.ex       # Vector similarity (Nx)
│       ├── batch.ex            # Batch processing
│       └── application.ex      # OTP application
└── test/
    └── embed_ex/
        ├── embedding_test.exs
        ├── similarity_test.exs
        ├── cache_test.exs
        └── providers/
            ├── ollama_test.exs
            └── ollama_integration_test.exs
```

### Key Components

**EmbedEx.Embedding** - Struct representing an embedding with metadata:
```elixir
%EmbedEx.Embedding{
  text: "original text",
  vector: [0.1, 0.2, ...],
  model: "text-embedding-3-small",
  provider: :openai,
  dimensions: 1536,
  metadata: %{}
}
```

**EmbedEx.Provider** - Behaviour for implementing providers:
- `embed/2` - Embed single text
- `embed_batch/2` - Embed batch of texts
- `default_model/0` - Get default model
- `max_batch_size/0` - Get max batch size
- `available_models/0` - List available models
- `validate_config/1` - Validate configuration

**EmbedEx.Cache** - Automatic caching with:
- SHA256-based cache keys (text + model + provider + dimensions)
- TTL support (default: 24 hours)
- Size limits (default: 10,000 entries)
- Statistics tracking

**EmbedEx.Similarity** - Vector operations using Nx:
- GPU acceleration when available
- Efficient batch operations
- Multiple similarity metrics

**EmbedEx.Batch** - Parallel processing:
- Automatic chunking
- Concurrent requests
- Progress callbacks
- Cache integration

## Integration with NSAI Ecosystem

EmbedEx is designed to integrate seamlessly with other NSAI projects:

### With CNS (Critic-Network Synthesis)

```elixir
# Embed claims for similarity-based retrieval
{:ok, claim_embeddings} = EmbedEx.embed_batch(
  claims,
  provider: :openai
)

# Find similar claims for antagonist
{:ok, similar} = EmbedEx.find_similar(
  query_embedding,
  claim_embeddings,
  top_k: 5,
  threshold: 0.8
)
```

### With Crucible Framework

```elixir
# Embed experimental results
{:ok, embeddings} = EmbedEx.embed_batch(
  experiment_descriptions,
  provider: :openai
)

# Cluster similar experiments
matrix = EmbedEx.pairwise_similarity(embeddings)
```

### With LlmGuard

```elixir
# Embed prompts for semantic similarity detection
{:ok, prompt_embedding} = EmbedEx.embed(prompt, provider: :openai)

# Compare against known attack patterns
{:ok, similar_attacks} = EmbedEx.find_similar(
  prompt_embedding,
  attack_pattern_embeddings,
  top_k: 1,
  threshold: 0.9
)
```

## Performance

### Benchmarks (OpenAI provider)

- Single embedding: ~100ms (with cache: <1ms)
- Batch of 100: ~2s (with cache: ~50ms)
- Similarity computation (1000x1000): ~10ms (GPU) / ~100ms (CPU)

### Caching Impact

Cache hit rates typically exceed 80% in production workloads, resulting in:
- 100x faster response times
- Significant API cost reduction
- Reduced rate limiting issues

## Testing

```bash
# Run all tests
mix test

# Run with coverage
mix test --cover

# Run specific test file
mix test test/embed_ex/similarity_test.exs
```

All tests pass:
```
Finished in 0.1 seconds (0.1s async, 0.02s sync)
38 tests, 0 failures
```

## Development

```bash
# Get dependencies
mix deps.get

# Compile
mix compile

# Format code
mix format

# Generate documentation
mix docs

# Run dialyzer (static analysis)
mix dialyzer
```

## Roadmap

### v0.2.0 (Current)
- [x] Ollama provider for local PHI-safe inference
- [x] Clinical embedding models support (nomic-embed-text, mxbai-embed-large)
- [x] Health check and model availability functions
- [x] Cohere provider implementation
- [x] Voyage AI provider implementation

### v0.3.0
- [ ] Redis cache backend
- [ ] Streaming embeddings for very large datasets
- [ ] Local models via Bumblebee (BERT, Sentence Transformers)
- [ ] Advanced similarity metrics (Manhattan, Chebyshev)

### v0.4.0
- [ ] Vector quantization for reduced memory
- [ ] Approximate nearest neighbors (ANN) search
- [ ] Integration with vector databases (Pinecone, Weaviate, Qdrant)

### v0.5.0
- [ ] Fine-tuning support for custom embeddings
- [ ] Multi-modal embeddings (text + images)
- [ ] Embedding aggregation strategies (mean pooling, max pooling)
- [ ] Phoenix LiveView component for embedding exploration

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`mix test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Support

For issues, questions, or contributions, please visit:
- GitHub: https://github.com/North-Shore-AI/embed_ex
- North Shore AI: https://github.com/North-Shore-AI

## Acknowledgments

Part of the North Shore AI monorepo - an Elixir-based ML reliability research ecosystem.

Related projects:
- **cns** - Critic-Network Synthesis for dialectical reasoning
- **crucible_framework** - ML experimentation orchestration
- **LlmGuard** - AI safety and security firewall
