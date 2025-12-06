defmodule EmbedEx.VectorStore.ETS do
  @moduledoc """
  ETS-based in-memory vector store.

  Fast in-memory storage for embeddings using Erlang Term Storage.
  Suitable for development, testing, and small-scale deployments.

  ## Features

    * Fast in-memory storage
    * No external dependencies
    * Metadata filtering support
    * Multiple similarity metrics
    * Persistence to disk (optional)

  ## Configuration

      config :embed_ex, EmbedEx.VectorStore.ETS,
        table_name: :embed_ex_vectors,
        persistence_path: "priv/vectors.ets"

  ## Examples

      # Start the store
      {:ok, _pid} = EmbedEx.VectorStore.ETS.start_link()

      # Insert an embedding
      {:ok, id} = EmbedEx.VectorStore.ETS.insert(embedding)

      # Search for similar embeddings
      {:ok, results} = EmbedEx.VectorStore.ETS.search(
        query_embedding,
        top_k: 5,
        metric: :cosine
      )

      # Persist to disk
      :ok = EmbedEx.VectorStore.ETS.save()
  """

  @behaviour EmbedEx.VectorStore

  use GenServer
  require Logger

  alias EmbedEx.{Embedding, Similarity}

  @default_table_name :embed_ex_vectors

  # Client API

  @doc """
  Starts the ETS vector store GenServer.
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl EmbedEx.VectorStore
  def insert(embedding, opts \\ []) do
    GenServer.call(__MODULE__, {:insert, embedding, opts})
  end

  @impl EmbedEx.VectorStore
  def insert_batch(embeddings, opts \\ []) do
    GenServer.call(__MODULE__, {:insert_batch, embeddings, opts})
  end

  @impl EmbedEx.VectorStore
  def get(id, opts \\ []) do
    table_name = get_table_name(opts)

    case :ets.lookup(table_name, id) do
      [{^id, embedding, _metadata, _timestamp}] -> {:ok, embedding}
      [] -> {:error, :not_found}
    end
  end

  @impl EmbedEx.VectorStore
  def get_batch(ids, opts \\ []) do
    results =
      Enum.map(ids, fn id ->
        case get(id, opts) do
          {:ok, embedding} -> embedding
          {:error, :not_found} -> nil
        end
      end)

    {:ok, results}
  end

  @impl EmbedEx.VectorStore
  def delete(id, opts \\ []) do
    table_name = get_table_name(opts)
    :ets.delete(table_name, id)
    :ok
  end

  @impl EmbedEx.VectorStore
  def delete_batch(ids, opts \\ []) do
    table_name = get_table_name(opts)

    count =
      Enum.reduce(ids, 0, fn id, acc ->
        :ets.delete(table_name, id)
        acc + 1
      end)

    {:ok, count}
  end

  @impl EmbedEx.VectorStore
  def search(query, opts \\ []) do
    GenServer.call(__MODULE__, {:search, query, opts}, :infinity)
  end

  @impl EmbedEx.VectorStore
  def count(opts \\ []) do
    table_name = get_table_name(opts)
    count = :ets.info(table_name, :size)
    {:ok, count}
  end

  @impl EmbedEx.VectorStore
  def clear(opts \\ []) do
    table_name = get_table_name(opts)
    count = :ets.info(table_name, :size)
    :ets.delete_all_objects(table_name)
    {:ok, count}
  end

  @impl EmbedEx.VectorStore
  def stats(opts \\ []) do
    table_name = get_table_name(opts)

    stats = %{
      count: :ets.info(table_name, :size),
      memory_bytes: :ets.info(table_name, :memory) * :erlang.system_info(:wordsize),
      table_name: table_name,
      type: :ets
    }

    {:ok, stats}
  end

  @doc """
  Saves the ETS table to disk.

  Returns `:ok` or `{:error, reason}`.
  """
  @spec save(String.t() | nil) :: :ok | {:error, term()}
  def save(path \\ nil) do
    GenServer.call(__MODULE__, {:save, path})
  end

  @doc """
  Loads the ETS table from disk.

  Returns `:ok` or `{:error, reason}`.
  """
  @spec load(String.t() | nil) :: :ok | {:error, term()}
  def load(path \\ nil) do
    GenServer.call(__MODULE__, {:load, path})
  end

  # Server callbacks

  @impl true
  def init(opts) do
    table_name = Keyword.get(opts, :table_name, @default_table_name)
    persistence_path = Keyword.get(opts, :persistence_path)

    # Create ETS table
    table =
      :ets.new(table_name, [
        :named_table,
        :set,
        :public,
        read_concurrency: true,
        write_concurrency: true
      ])

    state = %{
      table: table,
      table_name: table_name,
      persistence_path: persistence_path,
      next_id: 1
    }

    # Try to load from disk if persistence path is configured
    if persistence_path && File.exists?(persistence_path) do
      case do_load(table_name, persistence_path) do
        :ok ->
          Logger.info("Loaded ETS vector store from #{persistence_path}")
          # Update next_id based on loaded data
          max_id = get_max_id(table_name)
          {:ok, %{state | next_id: max_id + 1}}

        {:error, reason} ->
          Logger.warning("Failed to load ETS vector store: #{inspect(reason)}")
          {:ok, state}
      end
    else
      Logger.info("ETS vector store started")
      {:ok, state}
    end
  end

  @impl true
  def handle_call({:insert, embedding, _opts}, _from, state) do
    id = state.next_id
    timestamp = System.system_time(:second)
    metadata = embedding.metadata || %{}

    :ets.insert(state.table, {id, embedding, metadata, timestamp})

    {:reply, {:ok, id}, %{state | next_id: id + 1}}
  end

  @impl true
  def handle_call({:insert_batch, embeddings, _opts}, _from, state) do
    {ids, next_id} =
      Enum.reduce(embeddings, {[], state.next_id}, fn embedding, {ids_acc, current_id} ->
        timestamp = System.system_time(:second)
        metadata = embedding.metadata || %{}
        :ets.insert(state.table, {current_id, embedding, metadata, timestamp})
        {[current_id | ids_acc], current_id + 1}
      end)

    {:reply, {:ok, Enum.reverse(ids)}, %{state | next_id: next_id}}
  end

  @impl true
  def handle_call({:search, query, opts}, _from, state) do
    top_k = Keyword.get(opts, :top_k, 10)
    metric = Keyword.get(opts, :metric, :cosine)
    threshold = Keyword.get(opts, :threshold)
    filter = Keyword.get(opts, :filter)

    # Extract query vector
    query_vec = extract_vector(query)

    # Get all embeddings from ETS
    all_entries = :ets.tab2list(state.table)

    # Apply metadata filter if specified
    filtered_entries =
      if filter do
        Enum.filter(all_entries, fn {_id, _embedding, metadata, _timestamp} ->
          matches_filter?(metadata, filter)
        end)
      else
        all_entries
      end

    # Compute similarities
    similarities =
      Enum.map(filtered_entries, fn {id, embedding, metadata, _timestamp} ->
        score = Similarity.cosine_similarity(query_vec, embedding.vector)

        %{
          id: id,
          embedding: embedding,
          score: score,
          metadata: metadata
        }
      end)

    # Apply threshold if specified
    filtered_similarities =
      if threshold do
        Enum.filter(similarities, fn %{score: score} ->
          case metric do
            :euclidean -> score <= threshold
            _ -> score >= threshold
          end
        end)
      else
        similarities
      end

    # Sort and take top-k
    results =
      filtered_similarities
      |> sort_by_metric(metric)
      |> Enum.take(top_k)

    {:reply, {:ok, results}, state}
  end

  @impl true
  def handle_call({:save, path}, _from, state) do
    save_path = path || state.persistence_path

    if save_path do
      result = do_save(state.table_name, save_path)
      {:reply, result, state}
    else
      {:reply, {:error, :no_persistence_path}, state}
    end
  end

  @impl true
  def handle_call({:load, path}, _from, state) do
    load_path = path || state.persistence_path

    if load_path do
      result = do_load(state.table_name, load_path)

      new_state =
        if result == :ok do
          max_id = get_max_id(state.table_name)
          %{state | next_id: max_id + 1}
        else
          state
        end

      {:reply, result, new_state}
    else
      {:reply, {:error, :no_persistence_path}, state}
    end
  end

  # Private functions

  defp get_table_name(opts) do
    Keyword.get(opts, :table_name, @default_table_name)
  end

  defp extract_vector(%Embedding{vector: vector}), do: vector
  defp extract_vector(vector), do: vector

  defp matches_filter?(metadata, filter) do
    Enum.all?(filter, fn {key, value} ->
      Map.get(metadata, key) == value ||
        Map.get(metadata, Atom.to_string(key)) == value
    end)
  end

  defp sort_by_metric(results, :euclidean) do
    Enum.sort_by(results, & &1.score, :asc)
  end

  defp sort_by_metric(results, _metric) do
    Enum.sort_by(results, & &1.score, :desc)
  end

  defp do_save(table_name, path) do
    # Ensure directory exists
    Path.dirname(path) |> File.mkdir_p!()

    case :ets.tab2file(table_name, String.to_charlist(path)) do
      :ok -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  defp do_load(table_name, path) do
    case :ets.file2tab(String.to_charlist(path), verify: true) do
      {:ok, ^table_name} -> :ok
      {:ok, _other_table} -> {:error, :table_name_mismatch}
      {:error, reason} -> {:error, reason}
    end
  end

  defp get_max_id(table_name) do
    case :ets.foldl(
           fn {id, _embedding, _metadata, _timestamp}, max_id ->
             if is_integer(id) and id > max_id, do: id, else: max_id
           end,
           0,
           table_name
         ) do
      max_id when is_integer(max_id) -> max_id
      _ -> 0
    end
  end
end
