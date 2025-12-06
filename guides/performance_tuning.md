# Performance Tuning Guide

This guide covers optimization strategies for EmbedEx to maximize throughput, minimize latency, and reduce costs.

## Table of Contents

1. [Caching Strategies](#caching-strategies)
2. [Batch Processing](#batch-processing)
3. [Concurrency Tuning](#concurrency-tuning)
4. [Rate Limiting](#rate-limiting)
5. [Memory Optimization](#memory-optimization)
6. [Network Optimization](#network-optimization)
7. [Vector Store Performance](#vector-store-performance)
8. [Monitoring & Profiling](#monitoring--profiling)

---

## Caching Strategies

### Enable Caching

Always enable caching for production use:

```elixir
config :embed_ex, :cache,
  enabled: true,
  ttl: :timer.hours(24),
  limit: 100_000
```

### Cache Hit Optimization

```elixir
# Check cache statistics
{:ok, stats} = EmbedEx.cache_stats()
IO.inspect(stats)
# %{hits: 8500, misses: 1500, hit_rate: 0.85}

# Aim for >80% hit rate in production
```

### TTL Tuning

```elixir
# Short TTL for frequently changing content
config :embed_ex, :cache,
  ttl: :timer.hours(1)

# Long TTL for static content
config :embed_ex, :cache,
  ttl: :timer.days(7)

# No TTL (cache forever) for immutable data
config :embed_ex, :cache,
  ttl: :infinity
```

### Cache Warming

Pre-populate cache for known queries:

```elixir
# Warm cache during startup or off-peak hours
common_queries = [
  "What is machine learning?",
  "How does neural network work?",
  # ... more common queries
]

Task.async(fn ->
  Enum.each(common_queries, fn query ->
    EmbedEx.embed(query)
  end)
end)
```

---

## Batch Processing

### Optimal Batch Sizes

Use provider-specific maximum batch sizes:

```elixir
# OpenAI: 2048 (largest)
{:ok, embs} = EmbedEx.embed_batch(
  texts,
  provider: :openai,
  batch_size: 2048
)

# Cohere: 96
{:ok, embs} = EmbedEx.embed_batch(
  texts,
  provider: :cohere,
  batch_size: 96
)

# Voyage: 128
{:ok, embs} = EmbedEx.embed_batch(
  texts,
  provider: :voyage,
  batch_size: 128
)
```

### Adaptive Batching

Adjust batch size based on text length:

```elixir
defmodule MyApp.AdaptiveBatching do
  def embed_with_adaptive_batch(texts, opts \\ []) do
    avg_length = Enum.reduce(texts, 0, &(String.length(&1) + &2)) / length(texts)

    batch_size = cond do
      avg_length < 100 -> 2048   # Short texts: max batch
      avg_length < 500 -> 1024   # Medium texts: half batch
      true -> 512                # Long texts: smaller batch
    end

    EmbedEx.embed_batch(texts, Keyword.put(opts, :batch_size, batch_size))
  end
end
```

### Stream Processing for Large Corpora

Avoid loading everything into memory:

```elixir
# Process 1M documents without OOM
File.stream!("documents.jsonl")
|> Stream.map(&Jason.decode!/1)
|> Stream.map(& &1["text"])
|> Stream.chunk_every(1000)
|> Task.async_stream(
  fn batch ->
    {:ok, embs} = EmbedEx.embed_batch(batch)
    # Store embeddings
    MyApp.VectorStore.insert_batch(embs)
  end,
  max_concurrency: 5,
  timeout: :timer.minutes(5)
)
|> Stream.run()
```

---

## Concurrency Tuning

### Find Optimal Concurrency

Start low and increase until you hit rate limits or diminishing returns:

```elixir
# Test different concurrency levels
concurrency_levels = [1, 2, 4, 8, 16, 32]

results = Enum.map(concurrency_levels, fn concurrency ->
  start_time = System.monotonic_time(:millisecond)

  {:ok, _} = EmbedEx.embed_batch(
    sample_texts,
    concurrency: concurrency
  )

  duration = System.monotonic_time(:millisecond) - start_time
  {concurrency, duration}
end)

IO.inspect(results)
# [{1, 5000}, {2, 2600}, {4, 1400}, {8, 800}, {16, 750}, {32, 780}]
# Optimal: 16 (further increases don't help)
```

### Recommended Concurrency by Provider

```elixir
# OpenAI (high rate limits)
concurrency: 20

# Cohere (moderate rate limits)
concurrency: 10

# Voyage (moderate rate limits)
concurrency: 12
```

### Dynamic Concurrency

Adjust based on system load:

```elixir
defmodule MyApp.DynamicConcurrency do
  def get_concurrency do
    cpu_count = System.schedulers_online()
    current_load = :cpu_sup.avg1() / 256  # Load average

    cond do
      current_load < 0.5 -> cpu_count * 4
      current_load < 0.8 -> cpu_count * 2
      true -> cpu_count
    end
  end
end

{:ok, embs} = EmbedEx.embed_batch(
  texts,
  concurrency: MyApp.DynamicConcurrency.get_concurrency()
)
```

---

## Rate Limiting

### Using Built-in Rate Limiter

```elixir
# Start rate limiter
{:ok, _} = EmbedEx.RateLimiter.start_link(
  tokens_per_second: 100,
  burst_size: 200
)

# Wrap API calls
EmbedEx.RateLimiter.execute(fn ->
  EmbedEx.embed(text)
end)
```

### Exponential Backoff

```elixir
EmbedEx.RateLimiter.execute(
  fn -> EmbedEx.embed(text) end,
  max_retries: 5,
  initial_backoff: 1000,
  max_backoff: 60_000,
  backoff_multiplier: 2
)
```

### Provider-Specific Rate Limits

```elixir
# OpenAI
config :embed_ex, :rate_limiter,
  tokens_per_second: 200,  # Conservative
  burst_size: 400

# Cohere
config :embed_ex, :rate_limiter,
  tokens_per_second: 100,
  burst_size: 200

# Voyage
config :embed_ex, :rate_limiter,
  tokens_per_second: 150,
  burst_size: 300
```

---

## Memory Optimization

### Lazy Evaluation

Don't materialize large lists:

```elixir
# Bad: loads everything into memory
all_texts = Enum.map(1..1_000_000, &fetch_text/1)
{:ok, embs} = EmbedEx.embed_batch(all_texts)

# Good: streams through
Stream.map(1..1_000_000, &fetch_text/1)
|> Stream.chunk_every(1000)
|> Enum.each(fn batch ->
  {:ok, embs} = EmbedEx.embed_batch(batch)
  process_embeddings(embs)
end)
```

### Release Embeddings After Use

```elixir
# Convert tensors to lists when storing
embedding = %{embedding | vector: Nx.to_list(embedding.vector)}

# Or use EmbedEx.Embedding.to_list/1
embedding = EmbedEx.Embedding.to_list(embedding)
```

### Monitor Memory Usage

```elixir
defmodule MyApp.MemoryMonitor do
  def check_memory do
    memory = :erlang.memory()

    %{
      total_mb: memory[:total] / 1_024 / 1_024,
      processes_mb: memory[:processes] / 1_024 / 1_024,
      ets_mb: memory[:ets] / 1_024 / 1_024
    }
  end
end

# Monitor during batch processing
MyApp.MemoryMonitor.check_memory()
```

---

## Network Optimization

### Connection Pooling

EmbedEx uses Req which handles connection pooling automatically, but you can tune it:

```elixir
# In provider implementation (advanced)
Req.post(url,
  json: body,
  pool_timeout: 5000,
  receive_timeout: 30_000
)
```

### Compression

Enable compression for large payloads:

```elixir
# Cohere supports compression
{:ok, embs} = EmbedEx.embed_batch(
  very_long_texts,
  provider: :cohere,
  # Large batches benefit from compression
  batch_size: 96
)
```

### Retry Strategy

Configure retries for transient failures:

```elixir
# Built into providers
# Automatically retries on 429, 500, 502, 503, 504
# with exponential backoff
```

---

## Vector Store Performance

### ETS Vector Store Optimization

```elixir
# Start with appropriate table options
{:ok, _} = EmbedEx.VectorStore.ETS.start_link(
  read_concurrency: true,
  write_concurrency: true
)

# Batch inserts are faster
{:ok, ids} = EmbedEx.VectorStore.ETS.insert_batch(embeddings)

# Use metadata filtering efficiently
{:ok, results} = EmbedEx.VectorStore.ETS.search(
  query,
  filter: %{category: "tech"},  # Pre-filter
  top_k: 10
)
```

### Persistence Strategy

```elixir
# Persist periodically, not on every insert
:timer.send_interval(:timer.minutes(5), :persist_vectors)

def handle_info(:persist_vectors, state) do
  EmbedEx.VectorStore.ETS.save()
  {:noreply, state}
end
```

### Search Optimization

```elixir
# Use appropriate metric
# Cosine: best for semantic similarity
# Euclidean: better for clustered data
# Dot product: fastest for normalized vectors

{:ok, results} = EmbedEx.VectorStore.ETS.search(
  query,
  metric: :cosine,  # or :euclidean, :dot_product
  top_k: 10
)
```

---

## Monitoring & Profiling

### Track Embedding Performance

```elixir
defmodule MyApp.EmbeddingMetrics do
  def embed_with_metrics(text, opts \\ []) do
    start_time = System.monotonic_time(:microsecond)

    result = EmbedEx.embed(text, opts)

    duration = System.monotonic_time(:microsecond) - start_time

    :telemetry.execute(
      [:my_app, :embedding, :complete],
      %{duration: duration},
      %{provider: opts[:provider] || :openai}
    )

    result
  end
end
```

### Cache Hit Rate Monitoring

```elixir
# Monitor cache performance
def log_cache_stats do
  {:ok, stats} = EmbedEx.cache_stats()

  Logger.info("Cache stats: #{inspect(stats)}")

  if stats.hit_rate < 0.7 do
    Logger.warning("Cache hit rate below threshold: #{stats.hit_rate}")
  end
end

# Run periodically
:timer.send_interval(:timer.minutes(1), :log_cache_stats)
```

### Profiling with :fprof

```elixir
# Profile embedding operation
:fprof.trace([:start, {:procs, self()}])

{:ok, _} = EmbedEx.embed_batch(large_batch)

:fprof.trace(:stop)
:fprof.profile()
:fprof.analyse()
```

### Telemetry Integration

```elixir
# Attach telemetry handler
:telemetry.attach(
  "embed-ex-handler",
  [:embed_ex, :embedding, :complete],
  fn event, measurements, metadata, _config ->
    # Send to metrics system
    MyApp.Metrics.record(event, measurements, metadata)
  end,
  nil
)
```

---

## Performance Benchmarks

### Single Embedding

```
Provider    | p50    | p95    | p99
------------|--------|--------|--------
OpenAI      | 80ms   | 150ms  | 250ms
Cohere      | 120ms  | 200ms  | 350ms
Voyage      | 90ms   | 160ms  | 270ms
```

### Batch Processing (1000 texts, optimal config)

```
Provider    | Throughput  | Duration
------------|-------------|----------
OpenAI      | 20k/s       | 50s
Cohere      | 10k/s       | 100s
Voyage      | 15k/s       | 67s
```

---

## Best Practices Summary

1. **Always enable caching** with appropriate TTL
2. **Use maximum batch sizes** for your provider
3. **Tune concurrency** based on rate limits
4. **Stream large datasets** to avoid OOM
5. **Monitor cache hit rates** and tune accordingly
6. **Use rate limiting** to avoid 429 errors
7. **Choose the right similarity metric** for your use case
8. **Profile regularly** to identify bottlenecks
9. **Deduplicate** before embedding to reduce API calls
10. **Use semantic deduplication** to reduce storage

---

## Cost Optimization

### Dimension Reduction (OpenAI only)

```elixir
# 67% cost savings with minimal quality loss
{:ok, emb} = EmbedEx.embed(
  text,
  provider: :openai,
  model: "text-embedding-3-small",
  dimensions: 512  # vs default 1536
)
```

### Deduplication

```elixir
# Remove semantic duplicates before embedding
{:ok, unique_indices} = EmbedEx.Deduplication.deduplicate(
  texts,
  threshold: 0.95,
  return_indices: true
)

unique_texts = Enum.map(unique_indices, &Enum.at(texts, &1))
{:ok, embs} = EmbedEx.embed_batch(unique_texts)

# 20-40% reduction in API calls typical
```

### Smart Caching

```elixir
# Normalize text before caching to improve hit rate
defmodule MyApp.SmartCache do
  def normalize(text) do
    text
    |> String.downcase()
    |> String.trim()
    |> String.replace(~r/\s+/, " ")
  end

  def embed(text, opts \\ []) do
    normalized = normalize(text)
    EmbedEx.embed(normalized, opts)
  end
end

# Improves cache hit rate by 5-10%
```

---

## Troubleshooting

### Slow Embeddings

1. Check network latency: `ping api.openai.com`
2. Verify concurrency isn't too high
3. Check cache hit rate
4. Profile with `:fprof`

### High Memory Usage

1. Use streaming for large batches
2. Convert tensors to lists after use
3. Enable garbage collection: `:erlang.garbage_collect()`
4. Check for memory leaks in cache

### Rate Limit Errors

1. Reduce concurrency
2. Enable rate limiter
3. Increase backoff delays
4. Consider upgrading API tier

---

## Integration with CNS

For CNS-specific optimizations, see the [CNS Integration Guide](cns_integration.md).
