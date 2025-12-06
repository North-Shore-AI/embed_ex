defmodule EmbedEx.Cache do
  @moduledoc """
  Caching layer for embeddings.

  Uses Cachex to store embeddings with TTL support. Cache keys are generated
  from a hash of the text and model configuration.

  ## Configuration

  Configure in your application config:

      config :embed_ex, :cache,
        enabled: true,
        ttl: :timer.hours(24),
        limit: 10_000

  ## Examples

      # Get from cache or compute
      EmbedEx.Cache.fetch("Hello world", model: "text-embedding-3-small", fn ->
        # This function is only called if not in cache
        {:ok, embedding} = EmbedEx.Providers.OpenAI.embed("Hello world")
        embedding
      end)

      # Manually put in cache
      EmbedEx.Cache.put(embedding, model: "text-embedding-3-small")

      # Clear cache
      EmbedEx.Cache.clear()
  """

  use GenServer
  require Logger

  alias EmbedEx.Embedding

  @cache_name :embed_ex_cache

  # Client API

  @doc """
  Starts the cache GenServer.
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Fetches an embedding from the cache, or computes it using the provided function.

  Returns the cached or newly computed embedding.
  """
  @spec fetch(String.t(), keyword(), (-> term())) :: term()
  def fetch(text, opts, compute_fn) when is_function(compute_fn, 0) do
    if cache_enabled?() do
      key = cache_key(text, opts)

      case Cachex.fetch(@cache_name, key, fn _key ->
             result = compute_fn.()
             {:commit, result}
           end) do
        {:ok, result} ->
          result

        {:commit, result} ->
          Logger.debug("Cache miss for key: #{inspect(key)}")
          result

        {:error, reason} ->
          Logger.warning("Cache error: #{inspect(reason)}, computing anyway")
          compute_fn.()
      end
    else
      compute_fn.()
    end
  end

  @doc """
  Puts an embedding in the cache.
  """
  @spec put(Embedding.t(), keyword()) :: :ok
  def put(embedding, opts) do
    if cache_enabled?() do
      key = cache_key(embedding.text, opts)
      ttl = get_ttl()

      case Cachex.put(@cache_name, key, embedding, ttl: ttl) do
        {:ok, true} ->
          :ok

        {:error, reason} ->
          Logger.warning("Failed to cache embedding: #{inspect(reason)}")
          :ok
      end
    else
      :ok
    end
  end

  @doc """
  Gets an embedding from the cache.

  Returns `{:ok, embedding}` if found, `{:error, :not_found}` otherwise.
  """
  @spec get(String.t(), keyword()) :: {:ok, Embedding.t()} | {:error, term()}
  def get(text, opts) do
    if cache_enabled?() do
      key = cache_key(text, opts)

      case Cachex.get(@cache_name, key) do
        {:ok, nil} -> {:error, :not_found}
        {:ok, embedding} -> {:ok, embedding}
        {:error, reason} -> {:error, reason}
      end
    else
      {:error, :cache_disabled}
    end
  end

  @doc """
  Clears all cached embeddings.
  """
  @spec clear() :: {:ok, non_neg_integer()}
  def clear do
    if cache_enabled?() do
      Cachex.clear(@cache_name)
    else
      {:ok, 0}
    end
  end

  @doc """
  Returns cache statistics.
  """
  @spec stats() :: {:ok, map()} | {:error, term()}
  def stats do
    if cache_enabled?() do
      {:ok, stats} = Cachex.stats(@cache_name)
      {:ok, stats}
    else
      {:error, :cache_disabled}
    end
  end

  # Server callbacks

  @impl true
  def init(_opts) do
    if cache_enabled?() do
      cache_opts = [
        limit: get_config(:limit, 10_000),
        stats: true
      ]

      case Cachex.start_link(@cache_name, cache_opts) do
        {:ok, _pid} ->
          Logger.info("EmbedEx cache started")
          {:ok, %{}}

        {:error, reason} ->
          Logger.error("Failed to start cache: #{inspect(reason)}")
          {:stop, reason}
      end
    else
      Logger.info("EmbedEx cache disabled")
      {:ok, %{}}
    end
  end

  # Private functions

  defp cache_enabled? do
    get_config(:enabled, true)
  end

  defp get_ttl do
    get_config(:ttl, :timer.hours(24))
  end

  defp get_config(key, default) do
    Application.get_env(:embed_ex, :cache, [])
    |> Keyword.get(key, default)
  end

  defp cache_key(text, opts) do
    # Create a deterministic key from text + model + provider + dimensions
    model = opts[:model] || "default"
    provider = opts[:provider] || :default
    dimensions = opts[:dimensions]

    key_data = {text, model, provider, dimensions}

    :crypto.hash(:sha256, :erlang.term_to_binary(key_data))
    |> Base.encode16(case: :lower)
  end
end
