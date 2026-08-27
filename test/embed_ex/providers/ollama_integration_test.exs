defmodule EmbedEx.Providers.OllamaIntegrationTest do
  @moduledoc """
  Integration tests for Ollama provider.

  These tests require a running Ollama server with embedding models pulled.
  Run with: mix test --include integration

  Prerequisites:
  1. Ollama installed and running: `ollama serve`
  2. Required models pulled:
     - `ollama pull nomic-embed-text`
     - `ollama pull mxbai-embed-large` (optional)
  """

  use ExUnit.Case, async: false

  alias EmbedEx.Embedding
  alias EmbedEx.Providers.Ollama

  @moduletag :integration

  setup do
    # Check if Ollama is available
    case Ollama.health_check() do
      :ok -> :ok
      {:error, reason} -> {:skip, "Ollama not available: #{inspect(reason)}"}
    end
  end

  describe "embed/2 with real Ollama" do
    test "embeds single text with default model" do
      {:ok, embedding} = Ollama.embed("Hello world, this is a test.")

      assert %Embedding{} = embedding
      assert embedding.provider == :ollama
      assert embedding.model == "nomic-embed-text"
      assert is_list(embedding.vector)
      assert length(embedding.vector) == 768
      assert embedding.text == "Hello world, this is a test."
    end

    test "embeds clinical text" do
      clinical_text = "Patient presents with acute chest pain radiating to left arm."

      {:ok, embedding} = Ollama.embed(clinical_text)

      assert %Embedding{} = embedding
      assert embedding.provider == :ollama
      assert is_list(embedding.vector)
      refute Enum.empty?(embedding.vector)
    end

    test "handles empty string" do
      {:ok, embedding} = Ollama.embed("")

      assert %Embedding{} = embedding
      assert is_list(embedding.vector)
    end

    test "handles long text" do
      long_text = String.duplicate("This is a test sentence. ", 100)

      {:ok, embedding} = Ollama.embed(long_text)

      assert %Embedding{} = embedding
      assert is_list(embedding.vector)
    end
  end

  describe "embed_batch/2 with real Ollama" do
    test "embeds multiple texts" do
      texts = [
        "First test text",
        "Second test text",
        "Third test text"
      ]

      {:ok, embeddings} = Ollama.embed_batch(texts)

      assert length(embeddings) == 3
      assert Enum.all?(embeddings, &(&1.provider == :ollama))
      assert Enum.all?(embeddings, &is_list(&1.vector))

      # Check order is preserved
      assert Enum.at(embeddings, 0).text == "First test text"
      assert Enum.at(embeddings, 1).text == "Second test text"
      assert Enum.at(embeddings, 2).text == "Third test text"
    end

    test "embeddings for similar texts have high similarity" do
      texts = [
        "The patient has diabetes",
        "The patient is diabetic",
        "The weather is sunny today"
      ]

      {:ok, embeddings} = Ollama.embed_batch(texts)

      # Compute cosine similarity between first two (similar) texts
      vec1 = Nx.tensor(Enum.at(embeddings, 0).vector)
      vec2 = Nx.tensor(Enum.at(embeddings, 1).vector)
      vec3 = Nx.tensor(Enum.at(embeddings, 2).vector)

      sim_similar = cosine_similarity(vec1, vec2) |> Nx.to_number()
      sim_different = cosine_similarity(vec1, vec3) |> Nx.to_number()

      # Similar texts should have higher similarity
      assert sim_similar > sim_different
    end

    test "handles large batch" do
      texts = for i <- 1..50, do: "Test text number #{i}"

      {:ok, embeddings} = Ollama.embed_batch(texts)

      assert length(embeddings) == 50
    end
  end

  describe "health_check/1 with real Ollama" do
    test "returns :ok when Ollama is running" do
      assert :ok = Ollama.health_check()
    end
  end

  describe "list_models/1 with real Ollama" do
    test "returns list of available models" do
      {:ok, models} = Ollama.list_models()

      assert is_list(models)
      # At minimum, we should have pulled nomic-embed-text for these tests
      # But we won't assert specific models as they depend on setup
    end
  end

  describe "list_embedding_models/1 with real Ollama" do
    test "returns list of embedding models" do
      {:ok, models} = Ollama.list_embedding_models()

      assert is_list(models)
    end
  end

  describe "EmbedEx integration" do
    test "works through main EmbedEx module" do
      {:ok, embedding} = EmbedEx.embed("Test text", provider: :ollama)

      assert %Embedding{} = embedding
      assert embedding.provider == :ollama
    end

    test "batch embedding through EmbedEx" do
      texts = ["Text 1", "Text 2", "Text 3"]

      {:ok, embeddings} = EmbedEx.embed_batch(texts, provider: :ollama)

      assert length(embeddings) == 3
      assert Enum.all?(embeddings, &(&1.provider == :ollama))
    end
  end

  # Helper functions

  defp cosine_similarity(vec1, vec2) do
    dot_product = Nx.dot(vec1, vec2)
    norm1 = Nx.sqrt(Nx.dot(vec1, vec1))
    norm2 = Nx.sqrt(Nx.dot(vec2, vec2))
    Nx.divide(dot_product, Nx.multiply(norm1, norm2))
  end
end
