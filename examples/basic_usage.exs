#!/usr/bin/env elixir

# Basic usage examples for EmbedEx
# This is a demonstration script - requires OPENAI_API_KEY to be set

# Note: This example requires running from within an Elixir environment
# with EmbedEx dependencies loaded. To test:
#
# 1. Set your OpenAI API key: export OPENAI_API_KEY=sk-...
# 2. Run: mix run examples/basic_usage.exs

IO.puts("EmbedEx Basic Usage Examples\n")

# Example 1: Single embedding
IO.puts("1. Single Embedding")
IO.puts("--------------------")

case EmbedEx.embed("Hello world", provider: :openai) do
  {:ok, embedding} ->
    IO.puts("Text: #{embedding.text}")
    IO.puts("Model: #{embedding.model}")
    IO.puts("Provider: #{embedding.provider}")
    IO.puts("Dimensions: #{embedding.dimensions}")
    IO.puts("Vector (first 5): #{inspect(Enum.take(embedding.vector, 5))}")

  {:error, :missing_api_key} ->
    IO.puts("ERROR: OPENAI_API_KEY environment variable not set")
    IO.puts("Set it with: export OPENAI_API_KEY=sk-...")

  {:error, reason} ->
    IO.puts("ERROR: #{inspect(reason)}")
end

IO.puts("\n")

# Example 2: Batch embeddings
IO.puts("2. Batch Embeddings with Progress")
IO.puts("----------------------------------")

texts = [
  "Artificial intelligence is transforming technology",
  "Machine learning enables computers to learn from data",
  "Deep learning uses neural networks with multiple layers",
  "Natural language processing helps computers understand text",
  "Computer vision allows machines to interpret images"
]

case EmbedEx.embed_batch(
       texts,
       provider: :openai,
       on_progress: fn completed, total ->
         IO.write("\rProgress: #{completed}/#{total}")
       end
     ) do
  {:ok, embeddings} ->
    IO.puts("\n\nGenerated #{length(embeddings)} embeddings")

    Enum.with_index(embeddings, 1)
    |> Enum.each(fn {emb, idx} ->
      IO.puts("  #{idx}. #{String.slice(emb.text, 0..50)}... (#{emb.dimensions}d)")
    end)

  {:error, reason} ->
    IO.puts("\nERROR: #{inspect(reason)}")
end

IO.puts("\n")

# Example 3: Similarity computations
IO.puts("3. Similarity Computations")
IO.puts("--------------------------")

# Create some test embeddings
emb1 = EmbedEx.Embedding.new([1.0, 0.0, 0.0], "test", :test)
emb2 = EmbedEx.Embedding.new([0.9, 0.1, 0.0], "test", :test)
emb3 = EmbedEx.Embedding.new([0.0, 1.0, 0.0], "test", :test)

similarity_1_2 = EmbedEx.cosine_similarity(emb1, emb2)
similarity_1_3 = EmbedEx.cosine_similarity(emb1, emb3)

IO.puts("Cosine similarity between similar vectors: #{Float.round(similarity_1_2, 4)}")
IO.puts("Cosine similarity between orthogonal vectors: #{Float.round(similarity_1_3, 4)}")

distance_1_2 = EmbedEx.euclidean_distance(emb1, emb2)
distance_1_3 = EmbedEx.euclidean_distance(emb1, emb3)

IO.puts("Euclidean distance (similar): #{Float.round(distance_1_2, 4)}")
IO.puts("Euclidean distance (orthogonal): #{Float.round(distance_1_3, 4)}")

IO.puts("\n")

# Example 4: Finding similar embeddings
IO.puts("4. Finding Similar Embeddings")
IO.puts("------------------------------")

query = EmbedEx.Embedding.new([1.0, 0.0, 0.0], "test", :test)

corpus = [
  EmbedEx.Embedding.new([1.0, 0.0, 0.0], "test", :test),
  EmbedEx.Embedding.new([0.9, 0.1, 0.0], "test", :test),
  EmbedEx.Embedding.new([0.8, 0.2, 0.0], "test", :test),
  EmbedEx.Embedding.new([0.0, 1.0, 0.0], "test", :test),
  EmbedEx.Embedding.new([0.5, 0.5, 0.0], "test", :test)
]

{:ok, results} = EmbedEx.find_similar(query, corpus, top_k: 3, metric: :cosine)

IO.puts("Top 3 most similar vectors:")

Enum.each(results, fn {score, idx} ->
  IO.puts("  Index #{idx}: similarity = #{Float.round(score, 4)}")
end)

IO.puts("\n")

# Example 5: Pairwise similarity matrix
IO.puts("5. Pairwise Similarity Matrix")
IO.puts("------------------------------")

vectors = [
  EmbedEx.Embedding.new([1.0, 0.0], "test", :test),
  EmbedEx.Embedding.new([0.0, 1.0], "test", :test),
  EmbedEx.Embedding.new([1.0, 1.0], "test", :test)
]

matrix = EmbedEx.pairwise_similarity(vectors, metric: :cosine)

IO.puts("Similarity matrix (3x3):")
IO.puts(inspect(matrix))

IO.puts("\n")

# Example 6: Cache statistics
IO.puts("6. Cache Statistics")
IO.puts("-------------------")

case EmbedEx.cache_stats() do
  {:ok, stats} ->
    IO.puts("Cache statistics:")
    IO.inspect(stats, pretty: true)

  {:error, reason} ->
    IO.puts("Cache stats unavailable: #{inspect(reason)}")
end

IO.puts("\n")

# Example 7: Available providers
IO.puts("7. Available Providers")
IO.puts("----------------------")

providers = EmbedEx.providers()

IO.puts("Configured providers:")

Enum.each(providers, fn provider ->
  IO.puts("\n  #{provider.name}")
  IO.puts("    Module: #{inspect(provider.module)}")
  IO.puts("    Max batch size: #{provider.max_batch_size}")
  IO.puts("    Available models:")

  Enum.each(provider.models, fn model ->
    IO.puts("      - #{model}")
  end)
end)

IO.puts("\n\nDone! All examples completed.")
