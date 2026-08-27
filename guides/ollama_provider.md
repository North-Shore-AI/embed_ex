# Ollama Provider Guide

The Ollama provider enables local embedding generation using [Ollama](https://ollama.ai/),
a tool for running large language models locally. This is essential for privacy-sensitive
applications where data cannot be sent to cloud providers.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Supported Models](#supported-models)
- [Usage](#usage)
- [PHI-Safe Clinical Embeddings](#phi-safe-clinical-embeddings)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### 1. Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [ollama.ai](https://ollama.ai/)

### 2. Start Ollama Server

```bash
ollama serve
```

The server runs on `http://localhost:11434` by default.

### 3. Pull Embedding Models

```bash
# Recommended for general use (768 dimensions)
ollama pull nomic-embed-text

# Higher quality, larger model (1024 dimensions)
ollama pull mxbai-embed-large

# Fast, lightweight model (384 dimensions)
ollama pull all-minilm

# High performance model (1024 dimensions)
ollama pull snowflake-arctic-embed
```

## Installation

Add EmbedEx to your `mix.exs` dependencies:

```elixir
def deps do
  [
    {:embed_ex, "~> 0.2.0"}
  ]
end
```

## Configuration

### Application Configuration

```elixir
# config/config.exs
config :embed_ex, :ollama,
  host: "http://localhost:11434",
  default_model: "nomic-embed-text"

# For PHI-safe applications, you may want to set Ollama as the default provider
config :embed_ex,
  default_provider: :ollama
```

### Environment Variables

You can also configure the Ollama host via environment variable:

```bash
export OLLAMA_HOST="http://localhost:11434"
```

### Runtime Configuration

Options can be passed directly to embedding functions:

```elixir
{:ok, embedding} = EmbedEx.embed("Hello world",
  provider: :ollama,
  host: "http://custom-host:11434",
  model: "mxbai-embed-large",
  timeout: 60_000  # 60 seconds
)
```

## Supported Models

| Model | Dimensions | Use Case | Notes |
|-------|------------|----------|-------|
| `nomic-embed-text` | 768 | General purpose | Default, good balance of quality/speed |
| `mxbai-embed-large` | 1024 | High quality | Better semantic understanding |
| `all-minilm` | 384 | Fast inference | Good for resource-constrained environments |
| `snowflake-arctic-embed` | 1024 | High performance | Excellent for retrieval tasks |

### Checking Available Models

```elixir
# List all models in Ollama
{:ok, models} = EmbedEx.Providers.Ollama.list_models()
# => ["nomic-embed-text:latest", "mxbai-embed-large:latest", ...]

# List only embedding models
{:ok, embed_models} = EmbedEx.Providers.Ollama.list_embedding_models()
# => ["nomic-embed-text:latest", "mxbai-embed-large:latest"]

# Check model dimensions
EmbedEx.Providers.Ollama.model_dimensions("nomic-embed-text")
# => 768
```

## Usage

### Single Embedding

```elixir
# Using default model (nomic-embed-text)
{:ok, embedding} = EmbedEx.embed("Hello world", provider: :ollama)

# Using specific model
{:ok, embedding} = EmbedEx.embed("Hello world",
  provider: :ollama,
  model: "mxbai-embed-large"
)

# Access embedding data
embedding.vector     # [0.123, -0.456, ...]
embedding.dimensions # 768 or 1024
embedding.model      # "nomic-embed-text"
embedding.provider   # :ollama
```

### Batch Embeddings

```elixir
texts = [
  "First document",
  "Second document",
  "Third document"
]

# Basic batch embedding
{:ok, embeddings} = EmbedEx.embed_batch(texts, provider: :ollama)

# With progress tracking
{:ok, embeddings} = EmbedEx.embed_batch(texts,
  provider: :ollama,
  on_progress: fn completed, total ->
    IO.puts("Progress: #{completed}/#{total}")
  end
)

# Control batch size and concurrency
{:ok, embeddings} = EmbedEx.embed_batch(texts,
  provider: :ollama,
  batch_size: 100,
  concurrency: 4
)
```

### Similarity Search

```elixir
# Embed query and corpus
{:ok, query_emb} = EmbedEx.embed("search query", provider: :ollama)
{:ok, corpus_embs} = EmbedEx.embed_batch(documents, provider: :ollama)

# Find top 5 most similar
{:ok, results} = EmbedEx.find_similar(query_emb, corpus_embs, top_k: 5)
# => [{0.95, 0}, {0.87, 2}, {0.82, 5}, ...]
```

### Direct Provider Usage

```elixir
alias EmbedEx.Providers.Ollama

# Health check
:ok = Ollama.health_check()

# Validate configuration
:ok = Ollama.validate_config(host: "http://localhost:11434")

# Direct embedding
{:ok, embedding} = Ollama.embed("Hello world")
{:ok, embeddings} = Ollama.embed_batch(["text1", "text2"])
```

## PHI-Safe Clinical Embeddings

The Ollama provider is designed for processing Protected Health Information (PHI)
because all inference happens locally. No data is sent to external servers.

### Clinical Use Example

```elixir
# Safe for clinical data - all processing is local
{:ok, embedding} = EmbedEx.embed(
  "Patient presents with acute chest pain radiating to left arm",
  provider: :ollama,
  model: "nomic-embed-text"
)

# Batch process clinical notes
clinical_notes = [
  "HPI: 65yo male with chest pain x 2 hours",
  "PMH: HTN, DM2, Hyperlipidemia",
  "Medications: Metformin, Lisinopril, Atorvastatin"
]

{:ok, embeddings} = EmbedEx.embed_batch(clinical_notes, provider: :ollama)
```

### Integration with MedlumLlm

For clinical applications using MedlumLlm:

```elixir
# MedlumLlm.Embedder wraps EmbedEx with PHI protection
{:ok, embedding} = MedlumLlm.Embedder.embed("clinical text")

# is_phi: true (default) forces local provider
{:ok, embedding} = MedlumLlm.Embedder.embed("clinical text", is_phi: true)
```

## Performance Considerations

### Hardware Requirements

- **CPU**: Works on any modern CPU, but slower than GPU
- **GPU**: NVIDIA with CUDA support significantly improves performance
- **RAM**: 4GB minimum, 8GB+ recommended for larger models
- **VRAM**: 2GB+ for GPU acceleration (model-dependent)

### Benchmarks

Approximate embedding times on different hardware (single text):

| Hardware | nomic-embed-text | mxbai-embed-large |
|----------|------------------|-------------------|
| CPU (i7) | ~200ms | ~400ms |
| RTX 3060 | ~50ms | ~100ms |
| RTX 4090 | ~20ms | ~40ms |

### Optimization Tips

1. **Use Batch Processing**: Much more efficient than individual calls
   ```elixir
   # Good - batch processing
   {:ok, embeddings} = EmbedEx.embed_batch(texts, provider: :ollama)

   # Inefficient - individual calls
   embeddings = Enum.map(texts, fn text ->
     {:ok, emb} = EmbedEx.embed(text, provider: :ollama)
     emb
   end)
   ```

2. **Enable Caching**: Avoid re-computing embeddings for the same text
   ```elixir
   # Caching is enabled by default
   {:ok, emb1} = EmbedEx.embed("Hello", provider: :ollama)  # Computed
   {:ok, emb2} = EmbedEx.embed("Hello", provider: :ollama)  # From cache
   ```

3. **Choose the Right Model**: Use smaller models when speed is critical
   ```elixir
   # Fast inference
   {:ok, emb} = EmbedEx.embed(text, provider: :ollama, model: "all-minilm")

   # Higher quality
   {:ok, emb} = EmbedEx.embed(text, provider: :ollama, model: "mxbai-embed-large")
   ```

## Troubleshooting

### Connection Errors

**Error:** `{:error, {:connection_error, ...}}`

**Solutions:**
1. Ensure Ollama is running: `ollama serve`
2. Check the host URL: `curl http://localhost:11434/api/tags`
3. Verify firewall settings

### Model Not Found

**Error:** `{:error, {:http_error, 404, ...}}`

**Solutions:**
1. Pull the model: `ollama pull nomic-embed-text`
2. Check available models: `ollama list`
3. Verify model name spelling

### Timeout Errors

**Error:** `{:error, :timeout}`

**Solutions:**
1. Increase timeout: `EmbedEx.embed(text, provider: :ollama, timeout: 120_000)`
2. Use a smaller model
3. Check system resources

### Health Check

```elixir
# Check if Ollama is available
case EmbedEx.Providers.Ollama.health_check() do
  :ok -> IO.puts("Ollama is running")
  {:error, reason} -> IO.puts("Ollama error: #{inspect(reason)}")
end

# Check if model is available
if EmbedEx.Providers.Ollama.model_available?("nomic-embed-text") do
  IO.puts("Model is ready")
else
  IO.puts("Pull model: ollama pull nomic-embed-text")
end
```

## API Reference

### Functions

- `EmbedEx.Providers.Ollama.embed/2` - Embed single text
- `EmbedEx.Providers.Ollama.embed_batch/2` - Embed multiple texts
- `EmbedEx.Providers.Ollama.default_model/0` - Get default model name
- `EmbedEx.Providers.Ollama.max_batch_size/0` - Get max batch size
- `EmbedEx.Providers.Ollama.available_models/0` - List supported models
- `EmbedEx.Providers.Ollama.model_dimensions/1` - Get dimensions for model
- `EmbedEx.Providers.Ollama.health_check/1` - Check server availability
- `EmbedEx.Providers.Ollama.model_available?/2` - Check if model is pulled
- `EmbedEx.Providers.Ollama.list_models/1` - List all Ollama models
- `EmbedEx.Providers.Ollama.list_embedding_models/1` - List embedding models
- `EmbedEx.Providers.Ollama.validate_config/1` - Validate configuration

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:host` | String | `"http://localhost:11434"` | Ollama server URL |
| `:model` | String | `"nomic-embed-text"` | Embedding model to use |
| `:timeout` | Integer | `30_000` | Request timeout in ms |
