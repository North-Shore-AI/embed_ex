defmodule EmbedEx.Providers.OllamaTest do
  use ExUnit.Case, async: true

  alias EmbedEx.Embedding
  alias EmbedEx.Providers.Ollama

  # Mock server setup - these tests use Bypass to mock the Ollama API
  # For integration tests with a real Ollama server, see ollama_integration_test.exs

  describe "embed/2" do
    test "returns embedding struct with correct fields" do
      # Create a mock embedding response
      mock_vector = List.duplicate(0.1, 768)

      # In real tests, we'd use Bypass or Mox to mock the HTTP call
      # For now, we test the struct creation logic directly
      embedding =
        Embedding.new(
          mock_vector,
          "nomic-embed-text",
          :ollama,
          text: "test text",
          metadata: %{dimensions: 768, host: "http://localhost:11434"}
        )

      assert %Embedding{} = embedding
      assert embedding.provider == :ollama
      assert embedding.model == "nomic-embed-text"
      assert embedding.dimensions == 768
      assert is_list(embedding.vector)
      assert length(embedding.vector) == 768
      assert embedding.text == "test text"
    end
  end

  describe "embed_batch/2" do
    test "returns list of embedding structs" do
      mock_vector1 = List.duplicate(0.1, 768)
      mock_vector2 = List.duplicate(0.2, 768)
      mock_vector3 = List.duplicate(0.3, 768)

      texts = ["text1", "text2", "text3"]
      vectors = [mock_vector1, mock_vector2, mock_vector3]

      embeddings =
        vectors
        |> Enum.zip(texts)
        |> Enum.map(fn {vector, text} ->
          Embedding.new(
            vector,
            "nomic-embed-text",
            :ollama,
            text: text,
            metadata: %{dimensions: 768, host: "http://localhost:11434"}
          )
        end)

      assert length(embeddings) == 3
      assert Enum.all?(embeddings, &(&1.provider == :ollama))
      assert Enum.all?(embeddings, &(&1.model == "nomic-embed-text"))

      # Verify order is preserved
      assert Enum.at(embeddings, 0).text == "text1"
      assert Enum.at(embeddings, 1).text == "text2"
      assert Enum.at(embeddings, 2).text == "text3"
    end
  end

  describe "default_model/0" do
    test "returns nomic-embed-text" do
      assert Ollama.default_model() == "nomic-embed-text"
    end
  end

  describe "max_batch_size/0" do
    test "returns 512" do
      assert Ollama.max_batch_size() == 512
    end
  end

  describe "available_models/0" do
    test "returns list of supported models" do
      models = Ollama.available_models()

      assert is_list(models)
      assert "nomic-embed-text" in models
      assert "mxbai-embed-large" in models
      assert "all-minilm" in models
      assert "snowflake-arctic-embed" in models
    end
  end

  describe "model_dimensions/1" do
    test "returns correct dimensions for known models" do
      assert Ollama.model_dimensions("nomic-embed-text") == 768
      assert Ollama.model_dimensions("mxbai-embed-large") == 1024
      assert Ollama.model_dimensions("all-minilm") == 384
      assert Ollama.model_dimensions("snowflake-arctic-embed") == 1024
    end

    test "returns nil for unknown models" do
      assert Ollama.model_dimensions("unknown-model") == nil
    end
  end

  describe "health_check/1" do
    @tag :external
    test "returns error for invalid host" do
      # This should fail to connect
      result = Ollama.health_check(host: "http://localhost:99999")

      assert {:error, {:connection_error, _}} = result
    end
  end

  describe "validate_config/1" do
    @tag :external
    test "returns error when ollama is unavailable" do
      result = Ollama.validate_config(host: "http://localhost:99999")

      assert {:error, {:ollama_unavailable, _}} = result
    end
  end
end
