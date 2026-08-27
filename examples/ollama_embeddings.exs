#!/usr/bin/env elixir

# Ollama embeddings examples for EmbedEx
# This demonstrates local embeddings using Ollama for PHI-safe processing
#
# Prerequisites:
# 1. Install Ollama: brew install ollama (macOS) or curl -fsSL https://ollama.ai/install.sh | sh (Linux)
# 2. Start Ollama: ollama serve
# 3. Pull embedding model: ollama pull nomic-embed-text
# 4. Run: mix run examples/ollama_embeddings.exs

alias EmbedEx.Providers.Ollama

IO.puts("EmbedEx Ollama Provider Examples\n")
IO.puts("=================================\n")

# Check if Ollama is available
IO.puts("Checking Ollama availability...")

case Ollama.health_check() do
  :ok ->
    IO.puts("✓ Ollama server is running\n")

  {:error, reason} ->
    IO.puts("✗ Ollama not available: #{inspect(reason)}")
    IO.puts("\nMake sure Ollama is installed and running:")
    IO.puts("  1. Install: brew install ollama")
    IO.puts("  2. Start server: ollama serve")
    IO.puts("  3. Pull model: ollama pull nomic-embed-text")
    System.halt(1)
end

# Example 1: List available models
IO.puts("1. Available Models")
IO.puts("-------------------")

IO.puts("\nKnown embedding models:")
Enum.each(Ollama.available_models(), fn model ->
  dims = Ollama.model_dimensions(model) || "unknown"
  IO.puts("  - #{model} (#{dims} dimensions)")
end)

IO.puts("\nModels installed in Ollama:")
case Ollama.list_embedding_models() do
  {:ok, models} when models != [] ->
    Enum.each(models, fn model ->
      IO.puts("  - #{model}")
    end)

  {:ok, []} ->
    IO.puts("  No embedding models found. Pull one with: ollama pull nomic-embed-text")

  {:error, reason} ->
    IO.puts("  Error listing models: #{inspect(reason)}")
end

IO.puts("\n")

# Example 2: Single embedding
IO.puts("2. Single Embedding")
IO.puts("-------------------")

case EmbedEx.embed("Hello world, this is a test.", provider: :ollama) do
  {:ok, embedding} ->
    IO.puts("Text: #{embedding.text}")
    IO.puts("Model: #{embedding.model}")
    IO.puts("Provider: #{embedding.provider}")
    IO.puts("Dimensions: #{embedding.dimensions}")
    IO.puts("Vector (first 5): #{inspect(Enum.take(embedding.vector, 5))}")

  {:error, reason} ->
    IO.puts("ERROR: #{inspect(reason)}")
end

IO.puts("\n")

# Example 3: Clinical text embedding (PHI-safe)
IO.puts("3. Clinical Text Embedding (PHI-Safe)")
IO.puts("--------------------------------------")

clinical_texts = [
  "Patient presents with acute chest pain radiating to left arm",
  "History of hypertension and type 2 diabetes mellitus",
  "EKG shows ST elevation in leads V1-V4",
  "Patient experiencing shortness of breath at rest"
]

IO.puts("Embedding clinical notes locally (no data leaves this machine):\n")

case EmbedEx.embed_batch(
       clinical_texts,
       provider: :ollama,
       on_progress: fn completed, total ->
         IO.write("\rProgress: #{completed}/#{total}")
       end
     ) do
  {:ok, embeddings} ->
    IO.puts("\n\nGenerated #{length(embeddings)} clinical embeddings")

    Enum.with_index(embeddings, 1)
    |> Enum.each(fn {emb, idx} ->
      truncated = String.slice(emb.text, 0..50)
      IO.puts("  #{idx}. #{truncated}... (#{emb.dimensions}d)")
    end)

  {:error, reason} ->
    IO.puts("\nERROR: #{inspect(reason)}")
end

IO.puts("\n")

# Example 4: Semantic similarity of medical terms
IO.puts("4. Semantic Similarity (Medical Terms)")
IO.puts("---------------------------------------")

medical_pairs = [
  {"heart attack", "myocardial infarction"},
  {"heart attack", "headache"},
  {"diabetes", "hyperglycemia"},
  {"diabetes", "fracture"}
]

Enum.each(medical_pairs, fn {term1, term2} ->
  with {:ok, emb1} <- EmbedEx.embed(term1, provider: :ollama),
       {:ok, emb2} <- EmbedEx.embed(term2, provider: :ollama) do
    similarity = EmbedEx.cosine_similarity(emb1, emb2)
    IO.puts("  '#{term1}' ↔ '#{term2}': #{Float.round(similarity, 4)}")
  else
    {:error, reason} ->
      IO.puts("  Error computing similarity: #{inspect(reason)}")
  end
end)

IO.puts("\n")

# Example 5: Finding similar clinical notes
IO.puts("5. Finding Similar Clinical Notes")
IO.puts("---------------------------------")

corpus = [
  "Patient with chest pain and elevated troponin levels",
  "Diabetic patient with poor glycemic control",
  "Post-surgical wound infection requiring antibiotics",
  "Acute coronary syndrome suspected, cardiology consult ordered",
  "Patient reports chronic lower back pain"
]

query = "Heart attack symptoms with cardiac markers"

IO.puts("Query: \"#{query}\"")
IO.puts("\nSearching corpus of #{length(corpus)} clinical notes...")

with {:ok, query_emb} <- EmbedEx.embed(query, provider: :ollama),
     {:ok, corpus_embs} <- EmbedEx.embed_batch(corpus, provider: :ollama),
     {:ok, results} <- EmbedEx.find_similar(query_emb, corpus_embs, top_k: 3) do
  IO.puts("\nTop 3 most similar notes:")

  Enum.each(results, fn {score, idx} ->
    text = Enum.at(corpus, idx)
    IO.puts("  #{Float.round(score, 4)}: #{text}")
  end)
else
  {:error, reason} ->
    IO.puts("Error: #{inspect(reason)}")
end

IO.puts("\n")

# Example 6: Using different models
IO.puts("6. Model Comparison")
IO.puts("-------------------")

test_text = "The patient was admitted for observation"

IO.puts("Text: \"#{test_text}\"\n")

# Try nomic-embed-text (default)
case EmbedEx.embed(test_text, provider: :ollama, model: "nomic-embed-text") do
  {:ok, emb} ->
    IO.puts("nomic-embed-text: #{emb.dimensions} dimensions")

  {:error, {:http_error, 404, _}} ->
    IO.puts("nomic-embed-text: Not installed (run: ollama pull nomic-embed-text)")

  {:error, reason} ->
    IO.puts("nomic-embed-text: Error - #{inspect(reason)}")
end

# Try mxbai-embed-large if available
case EmbedEx.embed(test_text, provider: :ollama, model: "mxbai-embed-large") do
  {:ok, emb} ->
    IO.puts("mxbai-embed-large: #{emb.dimensions} dimensions")

  {:error, {:http_error, 404, _}} ->
    IO.puts("mxbai-embed-large: Not installed (run: ollama pull mxbai-embed-large)")

  {:error, reason} ->
    IO.puts("mxbai-embed-large: Error - #{inspect(reason)}")
end

IO.puts("\n")

# Example 7: Pairwise similarity matrix for clinical terms
IO.puts("7. Clinical Term Similarity Matrix")
IO.puts("-----------------------------------")

clinical_terms = [
  "myocardial infarction",
  "heart failure",
  "diabetes mellitus",
  "pneumonia"
]

IO.puts("Terms:")
Enum.with_index(clinical_terms, 1)
|> Enum.each(fn {term, idx} ->
  IO.puts("  #{idx}. #{term}")
end)

case EmbedEx.embed_batch(clinical_terms, provider: :ollama) do
  {:ok, embeddings} ->
    matrix = EmbedEx.pairwise_similarity(embeddings, metric: :cosine)

    IO.puts("\nSimilarity Matrix:")

    # Print header
    IO.write("       ")
    Enum.with_index(clinical_terms, 1)
    |> Enum.each(fn {_, idx} -> IO.write("   #{idx}    ") end)
    IO.puts("")

    # Print matrix rows
    matrix
    |> Nx.to_list()
    |> Enum.with_index(1)
    |> Enum.each(fn {row, idx} ->
      IO.write("  #{idx}   ")
      Enum.each(row, fn val ->
        IO.write(" #{:io_lib.format("~6.3f", [val])}")
      end)
      IO.puts("")
    end)

  {:error, reason} ->
    IO.puts("Error: #{inspect(reason)}")
end

IO.puts("\n")

# Example 8: Cache demonstration
IO.puts("8. Caching Demonstration")
IO.puts("------------------------")

test_text = "This is a cached embedding test"

IO.puts("First request (not cached):")
{time1, result1} = :timer.tc(fn -> EmbedEx.embed(test_text, provider: :ollama) end)
IO.puts("  Time: #{div(time1, 1000)}ms")

IO.puts("\nSecond request (should be cached):")
{time2, _result2} = :timer.tc(fn -> EmbedEx.embed(test_text, provider: :ollama) end)
IO.puts("  Time: #{div(time2, 1000)}ms")

if time1 > time2 * 2 do
  IO.puts("\n✓ Cache working! Second request was #{div(time1, time2)}x faster")
else
  IO.puts("\n  Note: Cache may not show significant speedup for local inference")
end

IO.puts("\n")

# Example 9: Provider comparison
IO.puts("9. Provider Information")
IO.puts("-----------------------")

IO.puts("\nOllama provider details:")
IO.puts("  Default model: #{Ollama.default_model()}")
IO.puts("  Max batch size: #{Ollama.max_batch_size()}")
IO.puts("  Available models: #{inspect(Ollama.available_models())}")

IO.puts("\nAll configured providers:")
Enum.each(EmbedEx.providers(), fn provider ->
  IO.puts("  - #{provider.name}: #{length(provider.models)} models, batch size #{provider.max_batch_size}")
end)

IO.puts("\n")
IO.puts("Done! All Ollama examples completed.")
IO.puts("\nKey takeaways:")
IO.puts("  • All embeddings generated locally - safe for PHI")
IO.puts("  • No API keys or cloud services required")
IO.puts("  • Same EmbedEx API as cloud providers")
IO.puts("  • Supports batch processing and caching")
