defmodule EmbedEx.Batch do
  @moduledoc """
  Batch processing utilities for embeddings.

  Handles automatic batching, parallel processing, and progress tracking
  for large-scale embedding operations.

  ## Examples

      # Batch embed with automatic chunking
      {:ok, embeddings} = EmbedEx.Batch.embed_batch(
        texts,
        provider: :openai,
        batch_size: 100,
        concurrency: 4
      )

      # With progress callback
      {:ok, embeddings} = EmbedEx.Batch.embed_batch(
        texts,
        provider: :openai,
        on_progress: fn completed, total ->
          IO.puts("Progress: \#{completed}/\#{total}")
        end
      )
  """

  alias EmbedEx.{Cache, Embedding}

  @doc """
  Embeds a batch of texts with automatic chunking and parallel processing.

  ## Options

    * `:provider` - Provider module (required)
    * `:batch_size` - Maximum batch size per request (default: provider max)
    * `:concurrency` - Number of concurrent requests (default: 4)
    * `:on_progress` - Callback function `(completed, total -> any())` (optional)
    * `:use_cache` - Whether to use caching (default: true)
    * Any provider-specific options

  Returns `{:ok, [%Embedding{}]}` or `{:error, reason}`.
  """
  @spec embed_batch([String.t()], keyword()) :: {:ok, [Embedding.t()]} | {:error, term()}
  def embed_batch(texts, opts) when is_list(texts) do
    provider = Keyword.fetch!(opts, :provider)
    provider_module = get_provider_module(provider)

    batch_size = Keyword.get(opts, :batch_size, provider_module.max_batch_size())
    concurrency = Keyword.get(opts, :concurrency, 4)
    use_cache = Keyword.get(opts, :use_cache, true)
    on_progress = Keyword.get(opts, :on_progress)

    total = length(texts)

    # Check cache first if enabled
    {cached, to_compute} =
      if use_cache do
        partition_cached(texts, opts)
      else
        {[], Enum.with_index(texts)}
      end

    # Process uncached texts in batches
    computed =
      to_compute
      |> Enum.chunk_every(batch_size)
      |> Task.async_stream(
        fn batch ->
          batch_texts = Enum.map(batch, fn {text, _idx} -> text end)
          indices = Enum.map(batch, fn {_text, idx} -> idx end)

          case provider_module.embed_batch(batch_texts, opts) do
            {:ok, embeddings} ->
              # Cache results
              if use_cache do
                Enum.each(embeddings, &Cache.put(&1, opts))
              end

              # Return with original indices
              {:ok, Enum.zip(embeddings, indices)}

            {:error, reason} ->
              {:error, reason}
          end
        end,
        max_concurrency: concurrency,
        timeout: :timer.minutes(5)
      )
      |> Enum.reduce_while([], fn
        {:ok, {:ok, results}}, acc ->
          # Report progress
          if on_progress do
            completed = length(acc) + length(results)
            on_progress.(completed + length(cached), total)
          end

          {:cont, acc ++ results}

        {:ok, {:error, reason}}, _acc ->
          {:halt, {:error, reason}}

        {:exit, reason}, _acc ->
          {:halt, {:error, {:task_exit, reason}}}
      end)

    case computed do
      {:error, _} = error ->
        error

      results ->
        # Combine cached and computed results, sort by original index
        all_results = (cached ++ results) |> Enum.sort_by(fn {_emb, idx} -> idx end)
        embeddings = Enum.map(all_results, fn {emb, _idx} -> emb end)
        {:ok, embeddings}
    end
  end

  @doc """
  Streams embeddings for a large corpus without loading all into memory.

  Returns a `Stream` that yields `{:ok, embedding}` or `{:error, reason}`.

  ## Examples

      texts
      |> EmbedEx.Batch.stream_embed(provider: :openai, batch_size: 100)
      |> Stream.each(fn
        {:ok, embedding} -> process_embedding(embedding)
        {:error, reason} -> handle_error(reason)
      end)
      |> Stream.run()
  """
  @spec stream_embed([String.t()], keyword()) :: Enumerable.t()
  def stream_embed(texts, opts) when is_list(texts) do
    provider = Keyword.fetch!(opts, :provider)
    provider_module = get_provider_module(provider)
    batch_size = Keyword.get(opts, :batch_size, provider_module.max_batch_size())

    Stream.resource(
      fn -> {texts, 0} end,
      fn
        {[], _index} ->
          {:halt, nil}

        {remaining, index} ->
          {batch, rest} = Enum.split(remaining, batch_size)
          batch_texts = Enum.map(batch, fn {text, _} -> text end)

          case provider_module.embed_batch(batch_texts, opts) do
            {:ok, embeddings} ->
              results = Enum.map(embeddings, &{:ok, &1})
              {results, {rest, index + length(batch)}}

            {:error, reason} ->
              {[{:error, reason}], {[], index}}
          end
      end,
      fn _acc -> :ok end
    )
  end

  @doc """
  Embeds texts in parallel with fine-grained control.

  Unlike `embed_batch/2`, this processes each text individually in parallel,
  which is useful when you want maximum concurrency and don't need to batch
  requests to the provider.

  ## Options

    * `:provider` - Provider module (required)
    * `:concurrency` - Number of concurrent requests (default: 10)
    * `:timeout` - Timeout per request in milliseconds (default: 30_000)
    * `:ordered` - Whether to preserve input order (default: true)
    * Any provider-specific options

  Returns `{:ok, [%Embedding{}]}` or `{:error, reason}`.
  """
  @spec parallel_embed([String.t()], keyword()) :: {:ok, [Embedding.t()]} | {:error, term()}
  def parallel_embed(texts, opts) when is_list(texts) do
    provider = Keyword.fetch!(opts, :provider)
    provider_module = get_provider_module(provider)

    concurrency = Keyword.get(opts, :concurrency, 10)
    timeout = Keyword.get(opts, :timeout, :timer.seconds(30))
    ordered = Keyword.get(opts, :ordered, true)

    results =
      texts
      |> Enum.with_index()
      |> Task.async_stream(
        fn {text, idx} ->
          case provider_module.embed(text, opts) do
            {:ok, embedding} -> {:ok, {embedding, idx}}
            {:error, reason} -> {:error, reason}
          end
        end,
        max_concurrency: concurrency,
        timeout: timeout,
        ordered: ordered
      )
      |> Enum.reduce_while([], fn
        {:ok, {:ok, {embedding, idx}}}, acc ->
          {:cont, [{embedding, idx} | acc]}

        {:ok, {:error, reason}}, _acc ->
          {:halt, {:error, reason}}

        {:exit, reason}, _acc ->
          {:halt, {:error, {:task_exit, reason}}}
      end)

    case results do
      {:error, _} = error ->
        error

      embeddings_with_idx ->
        embeddings =
          if ordered do
            embeddings_with_idx
            |> Enum.sort_by(fn {_emb, idx} -> idx end)
            |> Enum.map(fn {emb, _idx} -> emb end)
          else
            Enum.map(embeddings_with_idx, fn {emb, _idx} -> emb end)
          end

        {:ok, embeddings}
    end
  end

  # Private functions

  defp get_provider_module(:openai), do: EmbedEx.Providers.OpenAI
  defp get_provider_module(:cohere), do: EmbedEx.Providers.Cohere
  defp get_provider_module(:voyage), do: EmbedEx.Providers.Voyage
  defp get_provider_module(module) when is_atom(module), do: module

  defp partition_cached(texts, opts) do
    texts
    |> Enum.with_index()
    |> Enum.reduce({[], []}, fn {text, idx}, {cached, to_compute} ->
      case Cache.get(text, opts) do
        {:ok, embedding} ->
          {[{embedding, idx} | cached], to_compute}

        {:error, _} ->
          {cached, [{text, idx} | to_compute]}
      end
    end)
  end
end
