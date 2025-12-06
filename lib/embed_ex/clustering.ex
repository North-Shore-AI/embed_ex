defmodule EmbedEx.Clustering do
  @moduledoc """
  Clustering algorithms for embeddings.

  Provides k-means and DBSCAN clustering for grouping similar embeddings.

  ## Examples

      # K-means clustering
      {:ok, clusters} = EmbedEx.Clustering.kmeans(embeddings, k: 3)
      # => %{
      #   labels: [0, 0, 1, 2, 1, ...],
      #   centroids: [...],
      #   inertia: 123.45
      # }

      # DBSCAN clustering
      {:ok, clusters} = EmbedEx.Clustering.dbscan(
        embeddings,
        eps: 0.5,
        min_samples: 3
      )
      # => %{
      #   labels: [0, 0, -1, 1, 1, ...],  # -1 indicates noise
      #   n_clusters: 2
      # }
  """

  alias EmbedEx.{Embedding, Similarity}

  @doc """
  Performs k-means clustering on embeddings.

  ## Options

    * `:k` - Number of clusters (required)
    * `:max_iterations` - Maximum iterations (default: 100)
    * `:tolerance` - Convergence tolerance (default: 1.0e-4)
    * `:init` - Initialization method (`:random`, `:kmeans++`) (default: `:kmeans++`)
    * `:random_seed` - Random seed for reproducibility (optional)

  Returns a map with:
    * `:labels` - Cluster assignment for each embedding
    * `:centroids` - Cluster centroids
    * `:inertia` - Sum of squared distances to nearest centroid
    * `:iterations` - Number of iterations performed

  ## Examples

      embeddings = [emb1, emb2, emb3, emb4, emb5]
      {:ok, result} = EmbedEx.Clustering.kmeans(embeddings, k: 2)
      IO.inspect(result.labels)  # => [0, 0, 1, 1, 0]
  """
  @spec kmeans([Embedding.t()] | [list(float())], keyword()) ::
          {:ok, map()} | {:error, term()}
  def kmeans(embeddings, opts) do
    k = Keyword.fetch!(opts, :k)
    max_iterations = Keyword.get(opts, :max_iterations, 100)
    tolerance = Keyword.get(opts, :tolerance, 1.0e-4)
    init_method = Keyword.get(opts, :init, :kmeans_plus_plus)

    if length(embeddings) < k do
      {:error, :insufficient_data}
    else
      # Convert to tensor matrix
      vectors =
        embeddings
        |> Enum.map(&extract_vector/1)
        |> Enum.map(&ensure_tensor/1)

      matrix = Nx.stack(vectors)

      # Initialize centroids
      centroids = initialize_centroids(matrix, k, init_method, opts)

      # Run k-means iterations
      {final_centroids, labels, iterations} =
        kmeans_iterate(matrix, centroids, 0, max_iterations, tolerance)

      # Calculate inertia
      inertia = calculate_inertia(matrix, final_centroids, labels)

      {:ok,
       %{
         labels: Nx.to_flat_list(labels),
         centroids: Nx.to_list(final_centroids),
         inertia: Nx.to_number(inertia),
         iterations: iterations
       }}
    end
  end

  @doc """
  Performs DBSCAN (Density-Based Spatial Clustering of Applications with Noise) clustering.

  ## Options

    * `:eps` - Maximum distance between two samples (required)
    * `:min_samples` - Minimum samples in a neighborhood to form a cluster (default: 5)
    * `:metric` - Distance metric (`:euclidean`, `:cosine`) (default: `:euclidean`)

  Returns a map with:
    * `:labels` - Cluster labels (where -1 indicates noise/outliers)
    * `:n_clusters` - Number of clusters found
    * `:core_sample_indices` - Indices of core samples

  ## Examples

      {:ok, result} = EmbedEx.Clustering.dbscan(
        embeddings,
        eps: 0.3,
        min_samples: 3
      )
  """
  @spec dbscan([Embedding.t()] | [list(float())], keyword()) ::
          {:ok, map()} | {:error, term()}
  def dbscan(embeddings, opts) do
    eps = Keyword.fetch!(opts, :eps)
    min_samples = Keyword.get(opts, :min_samples, 5)
    metric = Keyword.get(opts, :metric, :euclidean)

    # Convert to tensor matrix
    vectors =
      embeddings
      |> Enum.map(&extract_vector/1)
      |> Enum.map(&ensure_tensor/1)

    matrix = Nx.stack(vectors)
    n = Nx.axis_size(matrix, 0)

    # Compute distance matrix
    distances = compute_distance_matrix(matrix, metric)

    # Initialize labels (-1 = noise, 0+ = cluster)
    labels = List.duplicate(-1, n)
    current_cluster = 0
    visited = MapSet.new()
    core_samples = []

    # Process each point
    {final_labels, final_cluster, core_samples} =
      Enum.reduce(0..(n - 1), {labels, current_cluster, core_samples}, fn idx,
                                                                          {labels_acc,
                                                                           cluster_acc,
                                                                           cores_acc} ->
        if MapSet.member?(visited, idx) do
          {labels_acc, cluster_acc, cores_acc}
        else
          neighbors = find_neighbors(distances, idx, eps)

          if length(neighbors) < min_samples do
            # Noise point
            {labels_acc, cluster_acc, cores_acc}
          else
            # Core point - expand cluster
            new_labels =
              expand_cluster(distances, labels_acc, idx, neighbors, cluster_acc, eps, min_samples)

            {new_labels, cluster_acc + 1, [idx | cores_acc]}
          end
        end
      end)

    n_clusters = final_cluster

    {:ok,
     %{
       labels: final_labels,
       n_clusters: n_clusters,
       core_sample_indices: Enum.reverse(core_samples)
     }}
  end

  @doc """
  Finds the optimal number of clusters using the elbow method.

  Runs k-means for different values of k and returns inertias.

  ## Options

    * `:k_range` - Range of k values to try (default: 2..10)
    * Other k-means options

  Returns a list of `{k, inertia}` tuples.
  """
  @spec elbow_method([Embedding.t()] | [list(float())], keyword()) ::
          {:ok, [{pos_integer(), float()}]} | {:error, term()}
  def elbow_method(embeddings, opts \\ []) do
    k_range = Keyword.get(opts, :k_range, 2..10)

    results =
      Enum.map(k_range, fn k ->
        opts_with_k = Keyword.put(opts, :k, k)

        case kmeans(embeddings, opts_with_k) do
          {:ok, %{inertia: inertia}} -> {k, inertia}
          {:error, _} -> {k, nil}
        end
      end)
      |> Enum.reject(fn {_k, inertia} -> is_nil(inertia) end)

    {:ok, results}
  end

  # Private functions

  defp extract_vector(%Embedding{vector: vector}), do: vector
  defp extract_vector(vector), do: vector

  defp ensure_tensor(%Nx.Tensor{} = tensor), do: tensor
  defp ensure_tensor(list) when is_list(list), do: Nx.tensor(list)

  defp initialize_centroids(matrix, k, :random, opts) do
    n = Nx.axis_size(matrix, 0)
    seed = Keyword.get(opts, :random_seed, :os.system_time(:microsecond))
    :rand.seed(:exsplus, {seed, seed, seed})

    indices = Enum.take_random(0..(n - 1), k)
    Nx.take(matrix, Nx.tensor(indices))
  end

  defp initialize_centroids(matrix, k, :kmeans_plus_plus, opts) do
    n = Nx.axis_size(matrix, 0)
    seed = Keyword.get(opts, :random_seed, :os.system_time(:microsecond))
    :rand.seed(:exsplus, {seed, seed, seed})

    # First centroid is random
    first_idx = :rand.uniform(n) - 1
    first_centroid = Nx.take(matrix, Nx.tensor([first_idx]))

    # Select remaining centroids with probability proportional to distance from nearest centroid
    kmeans_plus_plus_iterate(matrix, first_centroid, k - 1)
  end

  defp kmeans_plus_plus_iterate(_matrix, centroids, 0), do: centroids

  defp kmeans_plus_plus_iterate(matrix, centroids, remaining) do
    # Compute distances to nearest centroid
    distances = compute_min_distances(matrix, centroids)

    # Sample next centroid
    probs = Nx.divide(distances, Nx.sum(distances))
    cumsum = Nx.cumulative_sum(probs)
    rand_val = :rand.uniform()

    next_idx =
      cumsum
      |> Nx.to_flat_list()
      |> Enum.find_index(&(&1 >= rand_val))

    next_centroid = Nx.take(matrix, Nx.tensor([next_idx || 0]))
    new_centroids = Nx.concatenate([centroids, next_centroid])

    kmeans_plus_plus_iterate(matrix, new_centroids, remaining - 1)
  end

  defp compute_min_distances(matrix, centroids) do
    # For each point, find distance to nearest centroid
    n = Nx.axis_size(matrix, 0)
    k = Nx.axis_size(centroids, 0)

    Enum.reduce(0..(n - 1), Nx.tensor([]), fn i, acc ->
      point = Nx.take(matrix, Nx.tensor([i]))

      min_dist =
        Enum.reduce(0..(k - 1), nil, fn j, min_so_far ->
          centroid = Nx.take(centroids, Nx.tensor([j]))
          dist = euclidean_distance(point, centroid)

          case min_so_far do
            nil -> dist
            prev -> Nx.min(dist, prev)
          end
        end)

      if Nx.size(acc) == 0 do
        Nx.reshape(min_dist, {1})
      else
        Nx.concatenate([acc, Nx.reshape(min_dist, {1})])
      end
    end)
  end

  defp kmeans_iterate(matrix, centroids, iteration, max_iterations, tolerance) do
    if iteration >= max_iterations do
      labels = assign_labels(matrix, centroids)
      {centroids, labels, iteration}
    else
      # Assign points to nearest centroid
      labels = assign_labels(matrix, centroids)

      # Compute new centroids
      new_centroids = update_centroids(matrix, labels, Nx.axis_size(centroids, 0))

      # Check convergence
      diff = Nx.subtract(new_centroids, centroids)
      change = Nx.sum(Nx.abs(diff)) |> Nx.to_number()

      if change < tolerance do
        {new_centroids, labels, iteration + 1}
      else
        kmeans_iterate(matrix, new_centroids, iteration + 1, max_iterations, tolerance)
      end
    end
  end

  defp assign_labels(matrix, centroids) do
    n = Nx.axis_size(matrix, 0)
    k = Nx.axis_size(centroids, 0)

    labels =
      Enum.map(0..(n - 1), fn i ->
        point = Nx.take(matrix, Nx.tensor([i]))

        {_min_dist, label} =
          Enum.reduce(0..(k - 1), {nil, 0}, fn j, {min_dist, min_label} ->
            centroid = Nx.take(centroids, Nx.tensor([j]))
            dist = euclidean_distance(point, centroid) |> Nx.to_number()

            case min_dist do
              nil -> {dist, j}
              prev when dist < prev -> {dist, j}
              _ -> {min_dist, min_label}
            end
          end)

        label
      end)

    Nx.tensor(labels)
  end

  defp update_centroids(matrix, labels, k) do
    d = Nx.axis_size(matrix, 1)
    labels_list = Nx.to_flat_list(labels)

    centroids =
      Enum.map(0..(k - 1), fn cluster_id ->
        # Find all points in this cluster
        cluster_indices =
          labels_list
          |> Enum.with_index()
          |> Enum.filter(fn {label, _idx} -> label == cluster_id end)
          |> Enum.map(fn {_label, idx} -> idx end)

        if Enum.empty?(cluster_indices) do
          # Empty cluster - keep previous centroid (will be replaced by random point)
          List.duplicate(0.0, d)
        else
          # Compute mean of cluster points
          cluster_points = Nx.take(matrix, Nx.tensor(cluster_indices))
          Nx.mean(cluster_points, axes: [0]) |> Nx.to_flat_list()
        end
      end)

    Nx.tensor(centroids)
  end

  defp calculate_inertia(matrix, centroids, labels) do
    n = Nx.axis_size(matrix, 0)
    labels_list = Nx.to_flat_list(labels)

    total =
      Enum.reduce(0..(n - 1), 0.0, fn i, acc ->
        point = Nx.take(matrix, Nx.tensor([i]))
        label = Enum.at(labels_list, i)
        centroid = Nx.take(centroids, Nx.tensor([label]))

        dist = euclidean_distance(point, centroid) |> Nx.to_number()
        acc + dist * dist
      end)

    Nx.tensor(total)
  end

  defp euclidean_distance(vec1, vec2) do
    diff = Nx.subtract(vec1, vec2)
    squared = Nx.multiply(diff, diff)
    Nx.sqrt(Nx.sum(squared))
  end

  defp compute_distance_matrix(matrix, :euclidean) do
    Similarity.pairwise_similarity(Nx.to_list(matrix), metric: :euclidean)
  end

  defp compute_distance_matrix(matrix, :cosine) do
    # For cosine, we want distance = 1 - similarity
    sim_matrix = Similarity.pairwise_similarity(Nx.to_list(matrix), metric: :cosine)
    Nx.subtract(1.0, sim_matrix)
  end

  defp find_neighbors(distance_matrix, idx, eps) do
    row = Nx.take(distance_matrix, Nx.tensor([idx])) |> Nx.to_flat_list()

    row
    |> Enum.with_index()
    |> Enum.filter(fn {dist, _i} -> dist <= eps end)
    |> Enum.map(fn {_dist, i} -> i end)
  end

  defp expand_cluster(distances, labels, idx, neighbors, cluster_id, eps, min_samples) do
    # Mark current point as part of cluster
    updated_labels = List.replace_at(labels, idx, cluster_id)

    # Expand to neighbors
    Enum.reduce(neighbors, {updated_labels, neighbors}, fn neighbor_idx, {labels_acc, queue} ->
      if Enum.at(labels_acc, neighbor_idx) == -1 do
        # Mark neighbor as part of cluster
        new_labels = List.replace_at(labels_acc, neighbor_idx, cluster_id)

        # Check if neighbor is also a core point
        neighbor_neighbors = find_neighbors(distances, neighbor_idx, eps)

        if length(neighbor_neighbors) >= min_samples do
          # Add its neighbors to queue
          {new_labels, queue ++ neighbor_neighbors}
        else
          {new_labels, queue}
        end
      else
        {labels_acc, queue}
      end
    end)
    |> elem(0)
  end
end
