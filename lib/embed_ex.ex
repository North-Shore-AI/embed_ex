defmodule EmbedEx do
  @moduledoc """
  EmbedEx - Vector embeddings service for the NSAI ecosystem.

  A unified interface for generating and working with text embeddings across
  multiple providers (OpenAI, Cohere, local models) with built-in caching,
  batch processing, and similarity computations.

  ## Features

    * Multiple provider support with unified API
    * Automatic caching with TTL
    * Efficient batch processing with parallelization
    * Vector similarity computations using Nx
    * GPU acceleration support (when available)

  ## Quick Start

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

      # Find similar embeddings
      {:ok, results} = EmbedEx.find_similar(
        query_embedding,
        corpus_embeddings,
        top_k: 5
      )

  ## Configuration

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
  """

  alias EmbedEx.{Batch, Cache, Embedding, Providers, Similarity}

  @type embed_opts :: [
          provider: atom() | module(),
          model: String.t(),
          use_cache: boolean(),
          api_key: String.t()
        ]

  @doc """
  Embeds a single text string.

  ## Options

    * `:provider` - Provider to use (`:openai`, `:cohere`, `:local`, or module)
      (default: configured default or `:openai`)
    * `:model` - Model to use (provider-specific)
    * `:use_cache` - Whether to use caching (default: `true`)
    * Provider-specific options (e.g., `:api_key` for OpenAI)

  ## Examples

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
  """
  @spec embed(String.t(), embed_opts()) :: {:ok, Embedding.t()} | {:error, term()}
  def embed(text, opts \\ []) when is_binary(text) do
    provider = get_provider(opts)
    use_cache = Keyword.get(opts, :use_cache, true)

    if use_cache do
      Cache.fetch(text, opts, fn ->
        case provider.embed(text, opts) do
          {:ok, embedding} ->
            Cache.put(embedding, opts)
            {:ok, embedding}

          error ->
            error
        end
      end)
    else
      provider.embed(text, opts)
    end
  end

  @doc """
  Embeds a batch of text strings.

  Automatically handles chunking, parallel processing, and caching.

  ## Options

    * `:provider` - Provider to use (default: configured default or `:openai`)
    * `:batch_size` - Maximum batch size per request (default: provider max)
    * `:concurrency` - Number of concurrent requests (default: `4`)
    * `:use_cache` - Whether to use caching (default: `true`)
    * `:on_progress` - Progress callback `(completed, total -> any())`
    * Provider-specific options

  ## Examples

      {:ok, embeddings} = EmbedEx.embed_batch([
        "First text",
        "Second text",
        "Third text"
      ])

      # With progress tracking
      {:ok, embeddings} = EmbedEx.embed_batch(
        texts,
        provider: :openai,
        on_progress: fn completed, total ->
          IO.puts("Progress: \#{completed}/\#{total}")
        end
      )

      # Higher concurrency
      {:ok, embeddings} = EmbedEx.embed_batch(
        texts,
        provider: :openai,
        concurrency: 10
      )
  """
  @spec embed_batch([String.t()], embed_opts()) :: {:ok, [Embedding.t()]} | {:error, term()}
  def embed_batch(texts, opts \\ []) when is_list(texts) do
    opts = Keyword.put_new(opts, :provider, get_provider_name(opts))
    Batch.embed_batch(texts, opts)
  end

  @doc """
  Computes cosine similarity between two embeddings.

  Returns a float between -1 and 1, where 1 means identical vectors.

  ## Examples

      similarity = EmbedEx.cosine_similarity(embedding1, embedding2)
      # => 0.87
  """
  @spec cosine_similarity(Embedding.t() | list(float()), Embedding.t() | list(float())) ::
          float()
  def cosine_similarity(emb1, emb2) do
    Similarity.cosine_similarity(emb1, emb2)
  end

  @doc """
  Computes Euclidean distance between two embeddings.

  Lower values indicate more similar embeddings.

  ## Examples

      distance = EmbedEx.euclidean_distance(embedding1, embedding2)
      # => 0.23
  """
  @spec euclidean_distance(Embedding.t() | list(float()), Embedding.t() | list(float())) ::
          float()
  def euclidean_distance(emb1, emb2) do
    Similarity.euclidean_distance(emb1, emb2)
  end

  @doc """
  Computes dot product between two embeddings.

  ## Examples

      dot = EmbedEx.dot_product(embedding1, embedding2)
      # => 32.5
  """
  @spec dot_product(Embedding.t() | list(float()), Embedding.t() | list(float())) :: float()
  def dot_product(emb1, emb2) do
    Similarity.dot_product(emb1, emb2)
  end

  @doc """
  Finds the top-k most similar embeddings to a query.

  ## Options

    * `:top_k` - Number of results to return (default: `10`)
    * `:metric` - Similarity metric (`:cosine`, `:euclidean`, `:dot_product`)
      (default: `:cosine`)
    * `:threshold` - Minimum similarity threshold (optional)

  Returns `{:ok, results}` where results is a list of `{score, index}` tuples
  sorted by similarity.

  ## Examples

      {:ok, results} = EmbedEx.find_similar(
        query_embedding,
        corpus_embeddings,
        top_k: 5,
        metric: :cosine
      )

      # => {:ok, [{0.95, 0}, {0.87, 2}, {0.82, 5}, {0.79, 1}, {0.75, 8}]}

      # With threshold
      {:ok, results} = EmbedEx.find_similar(
        query_embedding,
        corpus_embeddings,
        top_k: 10,
        threshold: 0.8
      )
      # Only returns results with similarity >= 0.8
  """
  @spec find_similar(
          Embedding.t() | list(float()),
          [Embedding.t()] | [list(float())],
          keyword()
        ) :: {:ok, [{float(), non_neg_integer()}]} | {:error, term()}
  def find_similar(query, corpus, opts \\ []) do
    Similarity.find_similar(query, corpus, opts)
  end

  @doc """
  Computes a pairwise similarity matrix for a list of embeddings.

  Returns an Nx tensor of shape {n, n}.

  ## Options

    * `:metric` - Similarity metric (`:cosine`, `:euclidean`, `:dot_product`)
      (default: `:cosine`)

  ## Examples

      matrix = EmbedEx.pairwise_similarity([emb1, emb2, emb3])
      # => #Nx.Tensor<...>
  """
  @spec pairwise_similarity([Embedding.t()] | [list(float())], keyword()) :: Nx.Tensor.t()
  def pairwise_similarity(embeddings, opts \\ []) do
    Similarity.pairwise_similarity(embeddings, opts)
  end

  @doc """
  Clears the embedding cache.

  Returns `{:ok, count}` where count is the number of items cleared.
  """
  @spec clear_cache() :: {:ok, non_neg_integer()}
  def clear_cache do
    Cache.clear()
  end

  @doc """
  Returns cache statistics.

  ## Examples

      {:ok, stats} = EmbedEx.cache_stats()
      # => {:ok, %{hits: 150, misses: 50, ...}}
  """
  @spec cache_stats() :: {:ok, map()} | {:error, term()}
  def cache_stats do
    Cache.stats()
  end

  @doc """
  Returns information about available providers.

  ## Examples

      EmbedEx.providers()
      # => [
      #   %{
      #     name: :openai,
      #     module: EmbedEx.Providers.OpenAI,
      #     models: ["text-embedding-3-small", ...],
      #     max_batch_size: 2048
      #   },
      #   ...
      # ]
  """
  def providers do
    [
      %{
        name: :openai,
        module: Providers.OpenAI,
        models: Providers.OpenAI.available_models(),
        max_batch_size: Providers.OpenAI.max_batch_size()
      },
      %{
        name: :cohere,
        module: Providers.Cohere,
        models: Providers.Cohere.available_models(),
        max_batch_size: Providers.Cohere.max_batch_size()
      },
      %{
        name: :voyage,
        module: Providers.Voyage,
        models: Providers.Voyage.available_models(),
        max_batch_size: Providers.Voyage.max_batch_size()
      }
    ]
  end

  # Private functions

  defp get_provider(opts) do
    case Keyword.get(opts, :provider) do
      nil -> get_default_provider()
      :openai -> Providers.OpenAI
      :cohere -> Providers.Cohere
      :voyage -> Providers.Voyage
      module when is_atom(module) -> module
    end
  end

  defp get_provider_name(opts) do
    case Keyword.get(opts, :provider) do
      nil -> get_default_provider_name()
      name -> name
    end
  end

  defp get_default_provider do
    case Application.get_env(:embed_ex, :default_provider, :openai) do
      :openai -> Providers.OpenAI
      module when is_atom(module) -> module
    end
  end

  defp get_default_provider_name do
    Application.get_env(:embed_ex, :default_provider, :openai)
  end
end
