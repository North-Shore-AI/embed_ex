defmodule EmbedEx.Similarity do
  @moduledoc """
  Vector similarity computations using Nx.

  Provides efficient similarity metrics for embeddings, including:
  - Cosine similarity
  - Euclidean distance
  - Dot product
  - Finding top-k similar embeddings

  All operations leverage Nx for efficient tensor computations and
  optional GPU acceleration.

  ## Examples

      # Cosine similarity between two embeddings
      similarity = EmbedEx.Similarity.cosine_similarity(embedding1, embedding2)

      # Find most similar embeddings
      {:ok, results} = EmbedEx.Similarity.find_similar(
        query_embedding,
        corpus_embeddings,
        top_k: 5
      )

      # results => [{0.95, 0}, {0.87, 2}, {0.82, 5}, ...]
  """

  alias EmbedEx.Embedding

  @doc """
  Computes cosine similarity between two embeddings.

  Returns a float between -1 and 1, where:
  - 1 means identical vectors
  - 0 means orthogonal vectors
  - -1 means opposite vectors

  ## Examples

      iex> emb1 = EmbedEx.Embedding.new([1.0, 0.0, 0.0], "model", :provider)
      iex> emb2 = EmbedEx.Embedding.new([1.0, 0.0, 0.0], "model", :provider)
      iex> EmbedEx.Similarity.cosine_similarity(emb1, emb2)
      1.0

      iex> emb1 = EmbedEx.Embedding.new([1.0, 0.0], "model", :provider)
      iex> emb2 = EmbedEx.Embedding.new([0.0, 1.0], "model", :provider)
      iex> EmbedEx.Similarity.cosine_similarity(emb1, emb2)
      0.0
  """
  def cosine_similarity(%Embedding{} = emb1, %Embedding{} = emb2) do
    cosine_similarity(emb1.vector, emb2.vector)
  end

  def cosine_similarity(vec1, vec2) when is_list(vec1) and is_list(vec2) do
    cosine_similarity(Nx.tensor(vec1), Nx.tensor(vec2))
  end

  def cosine_similarity(%Nx.Tensor{} = vec1, %Nx.Tensor{} = vec2) do
    # Cosine similarity = (A · B) / (||A|| * ||B||)
    dot_product = Nx.dot(vec1, vec2)
    norm1 = Nx.sqrt(Nx.dot(vec1, vec1))
    norm2 = Nx.sqrt(Nx.dot(vec2, vec2))

    result = Nx.divide(dot_product, Nx.multiply(norm1, norm2))

    # Convert to scalar
    Nx.to_number(result)
  end

  @doc """
  Computes Euclidean distance between two embeddings.

  Lower values indicate more similar embeddings.

  ## Examples

      iex> emb1 = EmbedEx.Embedding.new([0.0, 0.0], "model", :provider)
      iex> emb2 = EmbedEx.Embedding.new([3.0, 4.0], "model", :provider)
      iex> EmbedEx.Similarity.euclidean_distance(emb1, emb2)
      5.0
  """
  def euclidean_distance(%Embedding{} = emb1, %Embedding{} = emb2) do
    euclidean_distance(emb1.vector, emb2.vector)
  end

  def euclidean_distance(vec1, vec2) when is_list(vec1) and is_list(vec2) do
    euclidean_distance(Nx.tensor(vec1), Nx.tensor(vec2))
  end

  def euclidean_distance(%Nx.Tensor{} = vec1, %Nx.Tensor{} = vec2) do
    diff = Nx.subtract(vec1, vec2)
    squared = Nx.multiply(diff, diff)
    sum_squared = Nx.sum(squared)

    sum_squared
    |> Nx.sqrt()
    |> Nx.to_number()
  end

  @doc """
  Computes dot product between two embeddings.

  ## Examples

      iex> emb1 = EmbedEx.Embedding.new([1.0, 2.0, 3.0], "model", :provider)
      iex> emb2 = EmbedEx.Embedding.new([4.0, 5.0, 6.0], "model", :provider)
      iex> EmbedEx.Similarity.dot_product(emb1, emb2)
      32.0
  """
  def dot_product(%Embedding{} = emb1, %Embedding{} = emb2) do
    dot_product(emb1.vector, emb2.vector)
  end

  def dot_product(vec1, vec2) when is_list(vec1) and is_list(vec2) do
    dot_product(Nx.tensor(vec1), Nx.tensor(vec2))
  end

  def dot_product(%Nx.Tensor{} = vec1, %Nx.Tensor{} = vec2) do
    vec1
    |> Nx.dot(vec2)
    |> Nx.to_number()
  end

  @doc """
  Finds the top-k most similar embeddings to a query embedding.

  ## Options

    * `:top_k` - Number of top results to return (default: 10)
    * `:metric` - Similarity metric to use (`:cosine`, `:euclidean`, `:dot_product`)
      (default: `:cosine`)
    * `:threshold` - Minimum similarity threshold (optional)

  Returns `{:ok, results}` where results is a list of `{similarity_score, index}`
  tuples sorted by similarity (highest first for cosine/dot, lowest first for euclidean).

  ## Examples

      query = EmbedEx.Embedding.new([1.0, 0.0, 0.0], "model", :provider)
      corpus = [
        EmbedEx.Embedding.new([1.0, 0.0, 0.0], "model", :provider),
        EmbedEx.Embedding.new([0.9, 0.1, 0.0], "model", :provider),
        EmbedEx.Embedding.new([0.0, 1.0, 0.0], "model", :provider)
      ]

      {:ok, results} = EmbedEx.Similarity.find_similar(query, corpus, top_k: 2)
      # => {:ok, [{1.0, 0}, {0.995, 1}]}
  """
  def find_similar(query, corpus, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, 10)
    metric = Keyword.get(opts, :metric, :cosine)
    threshold = Keyword.get(opts, :threshold)

    # Extract query vector
    query_vec = extract_vector(query)

    # Compute similarities for all corpus embeddings
    similarities =
      corpus
      |> Enum.with_index()
      |> Enum.map(fn {embedding, idx} ->
        corpus_vec = extract_vector(embedding)
        score = compute_similarity(query_vec, corpus_vec, metric)
        {score, idx}
      end)

    # Apply threshold if specified
    similarities =
      if threshold do
        Enum.filter(similarities, fn {score, _idx} ->
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
      similarities
      |> sort_by_metric(metric)
      |> Enum.take(top_k)

    {:ok, results}
  end

  @doc """
  Computes a pairwise similarity matrix for a list of embeddings.

  Returns an Nx tensor of shape {n, n} where element [i, j] is the
  similarity between embedding i and embedding j.

  ## Examples

      embeddings = [emb1, emb2, emb3]
      matrix = EmbedEx.Similarity.pairwise_similarity(embeddings)
      # => #Nx.Tensor<...>
  """
  def pairwise_similarity(embeddings, opts \\ []) do
    metric = Keyword.get(opts, :metric, :cosine)

    # Convert all embeddings to tensors and stack them
    vectors =
      embeddings
      |> Enum.map(&extract_vector/1)
      |> Enum.map(&ensure_tensor/1)

    # Stack into matrix
    matrix = Nx.stack(vectors)

    # Compute pairwise similarities
    case metric do
      :cosine -> cosine_similarity_matrix(matrix)
      :dot_product -> dot_product_matrix(matrix)
      :euclidean -> euclidean_distance_matrix(matrix)
    end
  end

  # Private functions

  defp extract_vector(%Embedding{vector: vector}), do: vector
  defp extract_vector(vector), do: vector

  defp ensure_tensor(%Nx.Tensor{} = tensor), do: tensor
  defp ensure_tensor(list) when is_list(list), do: Nx.tensor(list)

  defp compute_similarity(vec1, vec2, :cosine), do: cosine_similarity(vec1, vec2)
  defp compute_similarity(vec1, vec2, :euclidean), do: euclidean_distance(vec1, vec2)
  defp compute_similarity(vec1, vec2, :dot_product), do: dot_product(vec1, vec2)

  defp sort_by_metric(similarities, :euclidean) do
    # For Euclidean distance, lower is better
    Enum.sort_by(similarities, fn {score, _idx} -> score end, :asc)
  end

  defp sort_by_metric(similarities, _metric) do
    # For cosine and dot product, higher is better
    Enum.sort_by(similarities, fn {score, _idx} -> score end, :desc)
  end

  defp cosine_similarity_matrix(matrix) do
    # Normalize each row
    norms = Nx.sqrt(Nx.sum(Nx.pow(matrix, 2), axes: [1], keep_axes: true))
    normalized = Nx.divide(matrix, norms)

    # Compute dot product matrix
    Nx.dot(normalized, Nx.transpose(normalized))
  end

  defp dot_product_matrix(matrix) do
    Nx.dot(matrix, Nx.transpose(matrix))
  end

  defp euclidean_distance_matrix(matrix) do
    # Compute pairwise Euclidean distances using broadcasting
    n = Nx.axis_size(matrix, 0)

    # Expand dimensions for broadcasting
    matrix_i = Nx.reshape(matrix, {n, 1, :auto})
    matrix_j = Nx.reshape(matrix, {1, n, :auto})

    # Compute squared differences and sum
    diff = Nx.subtract(matrix_i, matrix_j)
    squared = Nx.pow(diff, 2)
    sum_squared = Nx.sum(squared, axes: [2])

    Nx.sqrt(sum_squared)
  end
end
