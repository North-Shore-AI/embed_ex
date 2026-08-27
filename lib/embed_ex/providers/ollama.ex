defmodule EmbedEx.Providers.Ollama do
  @moduledoc """
  Ollama embeddings provider for local inference.

  Supports local embedding models via Ollama for privacy-safe
  clinical text processing. No data leaves the local environment.

  ## Configuration

  Set the `OLLAMA_HOST` environment variable or configure in application:

      config :embed_ex, :ollama,
        host: "http://localhost:11434",
        default_model: "nomic-embed-text"

  ## Supported Models

    * `nomic-embed-text` (default) - 768 dimensions, general purpose
    * `mxbai-embed-large` - 1024 dimensions, higher quality
    * `all-minilm` - 384 dimensions, fast and lightweight
    * `snowflake-arctic-embed` - 1024 dimensions, high performance

  ## Examples

      # Using default model
      {:ok, embedding} = EmbedEx.Providers.Ollama.embed("Hello world")

      # Using specific model
      {:ok, embedding} = EmbedEx.Providers.Ollama.embed(
        "clinical note text",
        model: "mxbai-embed-large"
      )

      # Batch embedding
      {:ok, embeddings} = EmbedEx.Providers.Ollama.embed_batch(
        ["First text", "Second text", "Third text"]
      )

      # Custom host
      {:ok, embedding} = EmbedEx.Providers.Ollama.embed(
        "Hello world",
        host: "http://custom-ollama:11434"
      )

  ## PHI-Safe Usage

  This provider is designed for processing Protected Health Information (PHI)
  because all inference happens locally. No data is sent to external servers.

      # Safe for clinical data
      {:ok, embedding} = EmbedEx.embed("Patient presents with chest pain",
        provider: :ollama,
        model: "nomic-embed-text"
      )
  """

  @behaviour EmbedEx.Provider

  alias EmbedEx.Embedding

  @default_host "http://localhost:11434"
  @default_model "nomic-embed-text"
  @max_batch_size 512
  @default_timeout 30_000

  @available_models [
    "nomic-embed-text",
    "mxbai-embed-large",
    "all-minilm",
    "snowflake-arctic-embed"
  ]

  @model_dimensions %{
    "nomic-embed-text" => 768,
    "mxbai-embed-large" => 1024,
    "all-minilm" => 384,
    "snowflake-arctic-embed" => 1024
  }

  @impl true
  @spec embed(String.t(), keyword()) :: {:ok, Embedding.t()} | {:error, term()}
  def embed(text, opts \\ []) do
    with {:ok, response} <- make_request([text], opts) do
      [embedding_vector] = response["embeddings"]
      model = opts[:model] || @default_model

      embedding =
        Embedding.new(
          embedding_vector,
          model,
          :ollama,
          text: text,
          metadata: %{
            dimensions: length(embedding_vector),
            host: get_host(opts)
          }
        )

      {:ok, embedding}
    end
  end

  @impl true
  @spec embed_batch([String.t()], keyword()) :: {:ok, [Embedding.t()]} | {:error, term()}
  def embed_batch(texts, opts \\ []) when is_list(texts) do
    with {:ok, response} <- make_request(texts, opts) do
      model = opts[:model] || @default_model
      host = get_host(opts)

      embeddings =
        response["embeddings"]
        |> Enum.zip(texts)
        |> Enum.map(fn {embedding_vector, text} ->
          Embedding.new(
            embedding_vector,
            model,
            :ollama,
            text: text,
            metadata: %{
              dimensions: length(embedding_vector),
              host: host
            }
          )
        end)

      {:ok, embeddings}
    end
  end

  @impl true
  @spec default_model() :: String.t()
  def default_model, do: @default_model

  @impl true
  @spec max_batch_size() :: pos_integer()
  def max_batch_size, do: @max_batch_size

  @impl true
  @spec available_models() :: [String.t()]
  def available_models, do: @available_models

  @impl true
  @spec validate_config(keyword()) :: :ok | {:error, term()}
  def validate_config(opts) do
    host = get_host(opts)

    case health_check(host: host) do
      :ok -> :ok
      {:error, reason} -> {:error, {:ollama_unavailable, reason}}
    end
  end

  @doc """
  Returns the expected dimensions for a model.

  Returns `nil` for unknown models (custom models may have different dimensions).

  ## Examples

      iex> EmbedEx.Providers.Ollama.model_dimensions("nomic-embed-text")
      768

      iex> EmbedEx.Providers.Ollama.model_dimensions("mxbai-embed-large")
      1024

      iex> EmbedEx.Providers.Ollama.model_dimensions("unknown-model")
      nil
  """
  @spec model_dimensions(String.t()) :: pos_integer() | nil
  def model_dimensions(model) do
    Map.get(@model_dimensions, model)
  end

  @doc """
  Checks if Ollama server is available.

  ## Options

    * `:host` - Ollama server URL (default: `http://localhost:11434`)

  ## Examples

      iex> EmbedEx.Providers.Ollama.health_check()
      :ok

      iex> EmbedEx.Providers.Ollama.health_check(host: "http://localhost:99999")
      {:error, {:connection_error, %Req.TransportError{...}}}
  """
  @spec health_check(keyword()) :: :ok | {:error, term()}
  def health_check(opts \\ []) do
    host = get_host(opts)
    url = "#{host}/api/tags"

    case Req.get(url, receive_timeout: 5_000) do
      {:ok, %{status: 200}} -> :ok
      {:ok, %{status: status}} -> {:error, {:http_error, status}}
      {:error, reason} -> {:error, {:connection_error, reason}}
    end
  end

  @doc """
  Checks if a specific model is available in Ollama.

  ## Options

    * `:host` - Ollama server URL (default: `http://localhost:11434`)

  ## Examples

      iex> EmbedEx.Providers.Ollama.model_available?("nomic-embed-text")
      true

      iex> EmbedEx.Providers.Ollama.model_available?("nonexistent-model")
      false
  """
  @spec model_available?(String.t(), keyword()) :: boolean()
  def model_available?(model, opts \\ []) do
    case list_models(opts) do
      {:ok, models} -> model in models or String.contains?(models |> Enum.join(","), model)
      {:error, _} -> false
    end
  end

  @doc """
  Lists models currently available in Ollama.

  Returns all models, not just embedding models.

  ## Options

    * `:host` - Ollama server URL (default: `http://localhost:11434`)

  ## Examples

      {:ok, models} = EmbedEx.Providers.Ollama.list_models()
      # => ["nomic-embed-text:latest", "mxbai-embed-large:latest", ...]
  """
  @spec list_models(keyword()) :: {:ok, [String.t()]} | {:error, term()}
  def list_models(opts \\ []) do
    host = get_host(opts)
    url = "#{host}/api/tags"

    case Req.get(url, receive_timeout: 10_000) do
      {:ok, %{status: 200, body: body}} ->
        models =
          body["models"]
          |> Enum.map(& &1["name"])

        {:ok, models}

      {:ok, %{status: status, body: body}} ->
        {:error, {:http_error, status, body}}

      {:error, reason} ->
        {:error, {:connection_error, reason}}
    end
  end

  @doc """
  Lists only embedding models available in Ollama.

  Filters models by known embedding model names.

  ## Options

    * `:host` - Ollama server URL (default: `http://localhost:11434`)

  ## Examples

      {:ok, models} = EmbedEx.Providers.Ollama.list_embedding_models()
      # => ["nomic-embed-text:latest", "mxbai-embed-large:latest"]
  """
  @spec list_embedding_models(keyword()) :: {:ok, [String.t()]} | {:error, term()}
  def list_embedding_models(opts \\ []) do
    case list_models(opts) do
      {:ok, models} ->
        embedding_models = Enum.filter(models, &embedding_model?/1)
        {:ok, embedding_models}

      error ->
        error
    end
  end

  # Private functions

  defp get_host(opts) do
    Keyword.get(opts, :host) ||
      get_config(:host) ||
      System.get_env("OLLAMA_HOST") ||
      @default_host
  end

  defp get_config(key) do
    Application.get_env(:embed_ex, :ollama, [])
    |> Keyword.get(key)
  end

  defp make_request(texts, opts) do
    host = get_host(opts)
    model = opts[:model] || @default_model
    timeout = opts[:timeout] || @default_timeout
    url = "#{host}/api/embed"

    body = %{
      model: model,
      input: texts
    }

    case Req.post(url,
           json: body,
           receive_timeout: timeout
         ) do
      {:ok, %{status: 200, body: body}} ->
        {:ok, body}

      {:ok, %{status: status, body: body}} ->
        {:error, {:http_error, status, body}}

      {:error, reason} ->
        {:error, {:connection_error, reason}}
    end
  end

  defp embedding_model?(name) do
    # Heuristic: embedding models typically have "embed" in name
    name_lower = String.downcase(name)

    String.contains?(name_lower, "embed") ||
      String.contains?(name_lower, "minilm") ||
      Enum.any?(@available_models, fn model ->
        String.starts_with?(name_lower, model)
      end)
  end
end
