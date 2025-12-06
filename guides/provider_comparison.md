# Provider Comparison Guide

This guide helps you choose the right embedding provider for your use case.

## Quick Comparison

| Provider | Best For | Max Batch | Dimensions | Cost | Latency |
|----------|----------|-----------|------------|------|---------|
| **OpenAI** | General purpose, high quality | 2048 | 1536-3072 | Medium | Low |
| **Cohere** | Multilingual, search optimization | 96 | 1024 | Low | Medium |
| **Voyage AI** | Domain-specific (code, finance, law) | 128 | 512-1024 | Medium | Low |

## Detailed Provider Comparison

### OpenAI

**Models:**
- `text-embedding-3-small` - 1536 dimensions (default)
- `text-embedding-3-large` - 3072 dimensions
- `text-embedding-ada-002` - 1536 dimensions (legacy)

**Strengths:**
- Excellent general-purpose embeddings
- High quality semantic understanding
- Large batch sizes (up to 2048)
- Configurable dimensions for cost/quality tradeoff

**Best For:**
- General semantic search
- Question answering systems
- Content recommendation
- RAG (Retrieval Augmented Generation)

**Example:**
```elixir
{:ok, embedding} = EmbedEx.embed(
  "What is the meaning of life?",
  provider: :openai,
  model: "text-embedding-3-small"
)
```

**Pricing:** ~$0.02 per 1M tokens (3-small), ~$0.13 per 1M tokens (3-large)

---

### Cohere

**Models:**
- `embed-english-v3.0` - English-optimized (default)
- `embed-multilingual-v3.0` - 100+ languages
- `embed-english-light-v3.0` - Faster, lighter
- `embed-multilingual-light-v3.0` - Lighter multilingual

**Strengths:**
- Excellent multilingual support
- Input type optimization (search_query vs search_document)
- Competitive pricing
- Good for asymmetric search (query vs document)

**Best For:**
- Multilingual applications
- Search systems (separate query/document embeddings)
- Classification tasks
- Clustering

**Example:**
```elixir
# For search queries
{:ok, query_emb} = EmbedEx.embed(
  "best restaurants in tokyo",
  provider: :cohere,
  model: "embed-english-v3.0",
  input_type: :search_query
)

# For documents to be searched
{:ok, doc_embs} = EmbedEx.embed_batch(
  ["Restaurant A review...", "Restaurant B review..."],
  provider: :cohere,
  input_type: :search_document
)
```

**Pricing:** ~$0.10 per 1M tokens

**Unique Features:**
- Input type optimization improves search accuracy by 10-15%
- Compression via `embedding_types` parameter

---

### Voyage AI

**Models:**
- `voyage-3` - Latest general model (default)
- `voyage-3-lite` - Faster, smaller
- `voyage-code-3` - Code-optimized
- `voyage-finance-2` - Finance domain
- `voyage-law-2` - Legal domain
- `voyage-multilingual-2` - Multilingual

**Strengths:**
- Domain-specific models for code, finance, law
- High-performance general model
- Competitive latency
- Good context length support

**Best For:**
- Code search and similarity
- Financial document analysis
- Legal document search
- Domain-specific applications

**Example:**
```elixir
# Code embeddings
{:ok, code_emb} = EmbedEx.embed(
  "def fibonacci(n), do: ...",
  provider: :voyage,
  model: "voyage-code-3"
)

# Finance documents
{:ok, fin_emb} = EmbedEx.embed(
  "Q3 earnings report shows...",
  provider: :voyage,
  model: "voyage-finance-2"
)
```

**Pricing:** ~$0.12 per 1M tokens (voyage-3), ~$0.06 per 1M tokens (voyage-3-lite)

---

## Use Case Recommendations

### Semantic Search
**Recommended:** OpenAI (text-embedding-3-small) or Cohere (embed-english-v3.0)

Use Cohere's input type optimization:
```elixir
# Index documents
{:ok, doc_embeddings} = EmbedEx.embed_batch(
  documents,
  provider: :cohere,
  input_type: :search_document
)

# Query
{:ok, query_embedding} = EmbedEx.embed(
  user_query,
  provider: :cohere,
  input_type: :search_query
)

{:ok, results} = EmbedEx.find_similar(query_embedding, doc_embeddings, top_k: 10)
```

### Multilingual Applications
**Recommended:** Cohere (embed-multilingual-v3.0)

```elixir
# Supports 100+ languages
texts = [
  "Hello world",           # English
  "Bonjour le monde",      # French
  "こんにちは世界",          # Japanese
  "مرحبا بالعالم"          # Arabic
]

{:ok, embeddings} = EmbedEx.embed_batch(
  texts,
  provider: :cohere,
  model: "embed-multilingual-v3.0"
)
```

### Code Search
**Recommended:** Voyage AI (voyage-code-3)

```elixir
{:ok, code_embeddings} = EmbedEx.embed_batch(
  code_snippets,
  provider: :voyage,
  model: "voyage-code-3"
)
```

### Cost-Sensitive Applications
**Recommended:** OpenAI (text-embedding-3-small with reduced dimensions)

```elixir
# Reduce dimensions for lower cost
{:ok, embedding} = EmbedEx.embed(
  text,
  provider: :openai,
  model: "text-embedding-3-small",
  dimensions: 512  # Instead of default 1536
)
```

### High-Throughput Batch Processing
**Recommended:** OpenAI (largest batch size: 2048)

```elixir
# Process large batches efficiently
{:ok, embeddings} = EmbedEx.embed_batch(
  large_text_list,
  provider: :openai,
  batch_size: 2048,
  concurrency: 10
)
```

---

## Performance Comparison

### Latency (p50 for single embedding)
1. OpenAI: ~50-100ms
2. Voyage AI: ~60-120ms
3. Cohere: ~80-150ms

### Throughput (embeddings/second)
1. OpenAI: ~20,000 (with max batch size)
2. Voyage AI: ~15,000
3. Cohere: ~10,000

### Quality (MTEB Benchmark)
All three providers score competitively on MTEB (Massive Text Embedding Benchmark):
- OpenAI text-embedding-3-large: 64.6
- Voyage voyage-3: 64.4
- Cohere embed-english-v3.0: 64.0

---

## Migration Between Providers

EmbedEx makes it easy to switch providers:

```elixir
# Before
{:ok, emb} = EmbedEx.embed(text, provider: :openai)

# After
{:ok, emb} = EmbedEx.embed(text, provider: :cohere)

# Or use config
config :embed_ex, default_provider: :cohere
{:ok, emb} = EmbedEx.embed(text)
```

**Note:** Embeddings from different providers are **not compatible**. You'll need to re-embed your entire corpus when switching providers.

---

## Cost Optimization Strategies

### 1. Use Caching
```elixir
config :embed_ex, :cache,
  enabled: true,
  ttl: :timer.hours(24),
  limit: 100_000
```

### 2. Reduce Dimensions (OpenAI only)
```elixir
{:ok, emb} = EmbedEx.embed(
  text,
  provider: :openai,
  dimensions: 512  # 66% cost reduction
)
```

### 3. Choose Lighter Models
```elixir
# Cohere light model
{:ok, emb} = EmbedEx.embed(
  text,
  provider: :cohere,
  model: "embed-english-light-v3.0"
)

# Voyage lite model
{:ok, emb} = EmbedEx.embed(
  text,
  provider: :voyage,
  model: "voyage-3-lite"
)
```

### 4. Deduplicate Before Embedding
```elixir
# Remove exact duplicates first
unique_texts = Enum.uniq(texts)

# Then embed
{:ok, embeddings} = EmbedEx.embed_batch(unique_texts)
```

---

## Integration with CNS

For CNS (Critic-Network Synthesis) applications:

```elixir
# Use OpenAI for general claim extraction
{:ok, claim_emb} = EmbedEx.embed(
  claim_text,
  provider: :openai,
  model: "text-embedding-3-small"
)

# Use Cohere for evidence retrieval (asymmetric search)
{:ok, query_emb} = EmbedEx.embed(
  claim_text,
  provider: :cohere,
  input_type: :search_query
)

{:ok, evidence_embs} = EmbedEx.embed_batch(
  evidence_corpus,
  provider: :cohere,
  input_type: :search_document
)

{:ok, relevant_evidence} = EmbedEx.find_similar(
  query_emb,
  evidence_embs,
  top_k: 20,
  threshold: 0.7
)
```

---

## When to Use Local Models (Bumblebee)

Consider local embedding models when:
- Privacy is critical (no data leaves your infrastructure)
- You have GPU resources available
- You need zero-latency embeddings
- Cost is prohibitive for API providers

**Note:** Local models (Bumblebee) will be available in a future release of EmbedEx.
