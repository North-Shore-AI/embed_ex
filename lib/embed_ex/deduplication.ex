defmodule EmbedEx.Deduplication do
  @moduledoc """
  Semantic deduplication for embeddings and text.

  Identifies and removes semantically duplicate or near-duplicate items
  based on embedding similarity.

  ## Examples

      # Deduplicate embeddings
      {:ok, unique} = EmbedEx.Deduplication.deduplicate(embeddings)

      # Deduplicate with custom threshold
      {:ok, unique} = EmbedEx.Deduplication.deduplicate(
        embeddings,
        threshold: 0.95,
        strategy: :keep_first
      )

      # Get duplicate groups
      {:ok, groups} = EmbedEx.Deduplication.find_duplicates(
        embeddings,
        threshold: 0.9
      )
  """

  alias EmbedEx.{Embedding, Similarity}

  @type dedup_strategy :: :keep_first | :keep_last | :keep_longest | :keep_shortest

  @doc """
  Deduplicates a list of embeddings based on semantic similarity.

  ## Options

    * `:threshold` - Similarity threshold for duplicates (default: 0.95)
    * `:metric` - Similarity metric to use (`:cosine`, `:euclidean`) (default: `:cosine`)
    * `:strategy` - Which duplicate to keep (`:keep_first`, `:keep_last`, `:keep_longest`, `:keep_shortest`)
      (default: `:keep_first`)
    * `:return_indices` - If true, returns indices instead of embeddings (default: false)

  Returns `{:ok, unique_embeddings}` or `{:ok, unique_indices}` if `:return_indices` is true.

  ## Examples

      embeddings = [emb1, emb2, emb3, emb2_similar, emb4]
      {:ok, unique} = EmbedEx.Deduplication.deduplicate(embeddings, threshold: 0.95)
      # Returns [emb1, emb2, emb3, emb4] (emb2_similar removed as duplicate of emb2)
  """
  @spec deduplicate([Embedding.t()] | [list(float())], keyword()) ::
          {:ok, [Embedding.t()] | [list(float())] | [non_neg_integer()]} | {:error, term()}
  def deduplicate(embeddings, opts \\ []) do
    threshold = Keyword.get(opts, :threshold, 0.95)
    metric = Keyword.get(opts, :metric, :cosine)
    strategy = Keyword.get(opts, :strategy, :keep_first)
    return_indices = Keyword.get(opts, :return_indices, false)

    # Find duplicate groups
    {:ok, duplicate_groups} = find_duplicates(embeddings, threshold: threshold, metric: metric)

    # Determine which indices to keep
    indices_to_remove =
      duplicate_groups
      |> Enum.flat_map(fn group ->
        select_duplicates_to_remove(group, embeddings, strategy)
      end)
      |> MapSet.new()

    # Filter out duplicates
    unique_indices =
      embeddings
      |> Enum.with_index()
      |> Enum.reject(fn {_emb, idx} -> MapSet.member?(indices_to_remove, idx) end)
      |> Enum.map(fn {_emb, idx} -> idx end)

    result =
      if return_indices do
        unique_indices
      else
        Enum.map(unique_indices, &Enum.at(embeddings, &1))
      end

    {:ok, result}
  end

  @doc """
  Finds groups of duplicate embeddings.

  Returns groups of indices that are semantically similar above the threshold.

  ## Options

    * `:threshold` - Similarity threshold (default: 0.95)
    * `:metric` - Similarity metric (`:cosine`, `:euclidean`) (default: `:cosine`)

  Returns `{:ok, duplicate_groups}` where each group is a list of indices.

  ## Examples

      {:ok, groups} = EmbedEx.Deduplication.find_duplicates(embeddings)
      # => [[0, 3, 7], [1, 5], [2, 4, 9]]
      # Embeddings at indices [0, 3, 7] are duplicates of each other
  """
  @spec find_duplicates([Embedding.t()] | [list(float())], keyword()) ::
          {:ok, [[non_neg_integer()]]} | {:error, term()}
  def find_duplicates(embeddings, opts \\ []) do
    threshold = Keyword.get(opts, :threshold, 0.95)
    metric = Keyword.get(opts, :metric, :cosine)

    # Compute pairwise similarity matrix
    similarity_matrix = Similarity.pairwise_similarity(embeddings, metric: metric)

    # Convert to adjacency list of duplicates
    n = length(embeddings)
    adjacency = build_adjacency_list(similarity_matrix, threshold, metric, n)

    # Find connected components (duplicate groups)
    groups = find_connected_components(adjacency, n)

    # Filter out singleton groups (no duplicates)
    duplicate_groups = Enum.filter(groups, &(length(&1) > 1))

    {:ok, duplicate_groups}
  end

  @doc """
  Computes deduplication statistics for a collection.

  Returns information about duplicate patterns in the dataset.

  ## Options

    * `:threshold` - Similarity threshold (default: 0.95)
    * `:metric` - Similarity metric (default: `:cosine`)

  Returns a map with:
    * `:total_items` - Total number of embeddings
    * `:unique_items` - Number of unique embeddings
    * `:duplicate_items` - Number of duplicates
    * `:duplicate_groups` - Number of duplicate groups
    * `:deduplication_ratio` - Ratio of unique to total items
    * `:largest_group_size` - Size of the largest duplicate group

  ## Examples

      {:ok, stats} = EmbedEx.Deduplication.statistics(embeddings)
      # => %{
      #   total_items: 100,
      #   unique_items: 75,
      #   duplicate_items: 25,
      #   duplicate_groups: 10,
      #   deduplication_ratio: 0.75,
      #   largest_group_size: 5
      # }
  """
  @spec statistics([Embedding.t()] | [list(float())], keyword()) ::
          {:ok, map()} | {:error, term()}
  def statistics(embeddings, opts \\ []) do
    total = length(embeddings)

    {:ok, groups} = find_duplicates(embeddings, opts)

    duplicate_count = Enum.reduce(groups, 0, fn group, acc -> acc + length(group) - 1 end)
    unique_count = total - duplicate_count

    largest_group =
      if Enum.empty?(groups), do: 0, else: Enum.max_by(groups, &length/1) |> length()

    stats = %{
      total_items: total,
      unique_items: unique_count,
      duplicate_items: duplicate_count,
      duplicate_groups: length(groups),
      deduplication_ratio: unique_count / total,
      largest_group_size: largest_group
    }

    {:ok, stats}
  end

  @doc """
  Creates a deduplication report with detailed information.

  Returns a list of duplicate groups with metadata.

  ## Options

    * `:threshold` - Similarity threshold (default: 0.95)
    * `:metric` - Similarity metric (default: `:cosine`)
    * `:include_similarities` - Include similarity scores (default: true)

  ## Examples

      {:ok, report} = EmbedEx.Deduplication.report(embeddings)
      # => [
      #   %{
      #     indices: [0, 3, 7],
      #     size: 3,
      #     similarities: [[1.0, 0.97, 0.96], [0.97, 1.0, 0.98], [0.96, 0.98, 1.0]],
      #     representative: 0
      #   },
      #   ...
      # ]
  """
  @spec report([Embedding.t()] | [list(float())], keyword()) ::
          {:ok, [map()]} | {:error, term()}
  def report(embeddings, opts \\ []) do
    include_similarities = Keyword.get(opts, :include_similarities, true)

    {:ok, groups} = find_duplicates(embeddings, opts)

    report_data =
      Enum.map(groups, fn group ->
        report_entry = %{
          indices: group,
          size: length(group),
          representative: hd(group)
        }

        if include_similarities do
          # Compute pairwise similarities within group
          group_embeddings = Enum.map(group, &Enum.at(embeddings, &1))
          metric = Keyword.get(opts, :metric, :cosine)
          sim_matrix = Similarity.pairwise_similarity(group_embeddings, metric: metric)

          Map.put(report_entry, :similarities, Nx.to_list(sim_matrix))
        else
          report_entry
        end
      end)

    {:ok, report_data}
  end

  # Private functions

  defp build_adjacency_list(similarity_matrix, threshold, metric, n) do
    sim_list = Nx.to_list(similarity_matrix)

    Enum.reduce(0..(n - 1), %{}, fn i, acc ->
      row = Enum.at(sim_list, i)

      neighbors =
        row
        |> Enum.with_index()
        |> Enum.filter(fn {sim, j} ->
          i != j and is_similar?(sim, threshold, metric)
        end)
        |> Enum.map(fn {_sim, j} -> j end)

      Map.put(acc, i, neighbors)
    end)
  end

  defp is_similar?(similarity, threshold, :cosine) do
    # For cosine, higher is more similar
    similarity >= threshold
  end

  defp is_similar?(distance, threshold, :euclidean) do
    # For euclidean, lower is more similar
    # Convert threshold to distance threshold (inverse relationship)
    distance <= 1.0 - threshold
  end

  defp find_connected_components(adjacency, n) do
    visited = MapSet.new()

    {components, _} =
      Enum.reduce(0..(n - 1), {[], visited}, fn node, {groups, visited_acc} ->
        if MapSet.member?(visited_acc, node) do
          {groups, visited_acc}
        else
          {component, new_visited} = explore_component(adjacency, node, visited_acc)
          {[component | groups], new_visited}
        end
      end)

    Enum.reverse(components)
  end

  defp explore_component(adjacency, start_node, visited) do
    queue = [start_node]
    visited = MapSet.put(visited, start_node)
    do_explore(adjacency, queue, visited, [start_node])
  end

  defp do_explore(_adjacency, [], visited, component), do: {Enum.reverse(component), visited}

  defp do_explore(adjacency, [node | rest], visited, component) do
    neighbors = Map.get(adjacency, node, [])

    {new_queue, new_visited, new_component} =
      Enum.reduce(neighbors, {rest, visited, component}, fn neighbor, {q, v, c} ->
        if MapSet.member?(v, neighbor) do
          {q, v, c}
        else
          {q ++ [neighbor], MapSet.put(v, neighbor), [neighbor | c]}
        end
      end)

    do_explore(adjacency, new_queue, new_visited, new_component)
  end

  defp select_duplicates_to_remove(group, _embeddings, :keep_first) do
    # Keep first, remove rest
    tl(group)
  end

  defp select_duplicates_to_remove(group, _embeddings, :keep_last) do
    # Keep last, remove all but last
    group |> Enum.reverse() |> tl() |> Enum.reverse()
  end

  defp select_duplicates_to_remove(group, embeddings, :keep_longest) do
    # Keep the one with longest text
    longest_idx =
      group
      |> Enum.max_by(fn idx ->
        emb = Enum.at(embeddings, idx)
        get_text_length(emb)
      end)

    List.delete(group, longest_idx)
  end

  defp select_duplicates_to_remove(group, embeddings, :keep_shortest) do
    # Keep the one with shortest text
    shortest_idx =
      group
      |> Enum.min_by(fn idx ->
        emb = Enum.at(embeddings, idx)
        get_text_length(emb)
      end)

    List.delete(group, shortest_idx)
  end

  defp get_text_length(%Embedding{text: text}) when is_binary(text), do: String.length(text)
  defp get_text_length(%Embedding{text: nil}), do: 0
  defp get_text_length(_), do: 0
end
