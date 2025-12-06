defmodule EmbedEx.SimilarityTest do
  use ExUnit.Case, async: true

  alias EmbedEx.{Embedding, Similarity}

  describe "cosine_similarity/2" do
    test "returns 1.0 for identical vectors" do
      emb1 = Embedding.new([1.0, 0.0, 0.0], "model", :provider)
      emb2 = Embedding.new([1.0, 0.0, 0.0], "model", :provider)

      assert_in_delta Similarity.cosine_similarity(emb1, emb2), 1.0, 0.001
    end

    test "returns 0.0 for orthogonal vectors" do
      emb1 = Embedding.new([1.0, 0.0], "model", :provider)
      emb2 = Embedding.new([0.0, 1.0], "model", :provider)

      assert_in_delta Similarity.cosine_similarity(emb1, emb2), 0.0, 0.001
    end

    test "returns -1.0 for opposite vectors" do
      emb1 = Embedding.new([1.0, 0.0, 0.0], "model", :provider)
      emb2 = Embedding.new([-1.0, 0.0, 0.0], "model", :provider)

      assert_in_delta Similarity.cosine_similarity(emb1, emb2), -1.0, 0.001
    end

    test "computes correct similarity for arbitrary vectors" do
      emb1 = Embedding.new([1.0, 2.0, 3.0], "model", :provider)
      emb2 = Embedding.new([4.0, 5.0, 6.0], "model", :provider)

      # cos(θ) = (1*4 + 2*5 + 3*6) / (sqrt(14) * sqrt(77))
      # = 32 / (3.7417 * 8.7750) = 32 / 32.8329 ≈ 0.9746
      similarity = Similarity.cosine_similarity(emb1, emb2)
      assert_in_delta similarity, 0.9746, 0.001
    end

    test "works with raw lists" do
      similarity = Similarity.cosine_similarity([1.0, 0.0], [1.0, 0.0])
      assert_in_delta similarity, 1.0, 0.001
    end

    test "works with Nx tensors" do
      vec1 = Nx.tensor([1.0, 0.0, 0.0])
      vec2 = Nx.tensor([1.0, 0.0, 0.0])

      similarity = Similarity.cosine_similarity(vec1, vec2)
      assert_in_delta similarity, 1.0, 0.001
    end
  end

  describe "euclidean_distance/2" do
    test "returns 0.0 for identical vectors" do
      emb1 = Embedding.new([1.0, 2.0, 3.0], "model", :provider)
      emb2 = Embedding.new([1.0, 2.0, 3.0], "model", :provider)

      assert_in_delta Similarity.euclidean_distance(emb1, emb2), 0.0, 0.001
    end

    test "computes correct distance" do
      emb1 = Embedding.new([0.0, 0.0], "model", :provider)
      emb2 = Embedding.new([3.0, 4.0], "model", :provider)

      # sqrt((3-0)^2 + (4-0)^2) = sqrt(9 + 16) = sqrt(25) = 5.0
      distance = Similarity.euclidean_distance(emb1, emb2)
      assert_in_delta distance, 5.0, 0.001
    end

    test "works with raw lists" do
      distance = Similarity.euclidean_distance([0.0, 0.0], [3.0, 4.0])
      assert_in_delta distance, 5.0, 0.001
    end
  end

  describe "dot_product/2" do
    test "computes correct dot product" do
      emb1 = Embedding.new([1.0, 2.0, 3.0], "model", :provider)
      emb2 = Embedding.new([4.0, 5.0, 6.0], "model", :provider)

      # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
      dot = Similarity.dot_product(emb1, emb2)
      assert_in_delta dot, 32.0, 0.001
    end

    test "returns 0.0 for orthogonal vectors" do
      emb1 = Embedding.new([1.0, 0.0], "model", :provider)
      emb2 = Embedding.new([0.0, 1.0], "model", :provider)

      assert_in_delta Similarity.dot_product(emb1, emb2), 0.0, 0.001
    end
  end

  describe "find_similar/3" do
    test "finds top-k similar embeddings with cosine similarity" do
      query = Embedding.new([1.0, 0.0, 0.0], "model", :provider)

      corpus = [
        Embedding.new([1.0, 0.0, 0.0], "model", :provider),
        Embedding.new([0.9, 0.1, 0.0], "model", :provider),
        Embedding.new([0.0, 1.0, 0.0], "model", :provider),
        Embedding.new([0.8, 0.2, 0.0], "model", :provider)
      ]

      {:ok, results} = Similarity.find_similar(query, corpus, top_k: 2, metric: :cosine)

      assert length(results) == 2
      [{score1, idx1}, {score2, idx2}] = results

      # First should be exact match (index 0)
      assert idx1 == 0
      assert_in_delta score1, 1.0, 0.001

      # Second should be [0.9, 0.1, 0.0] (index 1)
      assert idx2 == 1
      assert score2 > 0.99
    end

    test "finds similar embeddings with threshold" do
      query = Embedding.new([1.0, 0.0, 0.0], "model", :provider)

      corpus = [
        Embedding.new([1.0, 0.0, 0.0], "model", :provider),
        Embedding.new([0.9, 0.1, 0.0], "model", :provider),
        Embedding.new([0.0, 1.0, 0.0], "model", :provider)
      ]

      {:ok, results} =
        Similarity.find_similar(query, corpus, top_k: 10, metric: :cosine, threshold: 0.95)

      # Only first two should pass threshold
      assert length(results) == 2
    end

    test "works with euclidean distance metric" do
      query = Embedding.new([0.0, 0.0], "model", :provider)

      corpus = [
        Embedding.new([1.0, 0.0], "model", :provider),
        Embedding.new([3.0, 4.0], "model", :provider),
        Embedding.new([0.0, 0.0], "model", :provider)
      ]

      {:ok, results} = Similarity.find_similar(query, corpus, top_k: 2, metric: :euclidean)

      # For Euclidean, lower is better
      [{score1, idx1}, {score2, idx2}] = results

      # First should be identical (distance 0, index 2)
      assert idx1 == 2
      assert_in_delta score1, 0.0, 0.001

      # Second should be [1.0, 0.0] (distance 1, index 0)
      assert idx2 == 0
      assert_in_delta score2, 1.0, 0.001
    end

    test "works with raw lists" do
      query = [1.0, 0.0, 0.0]
      corpus = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

      {:ok, results} = Similarity.find_similar(query, corpus, top_k: 1)

      assert length(results) == 1
      [{score, idx}] = results
      assert idx == 0
      assert_in_delta score, 1.0, 0.001
    end
  end

  describe "pairwise_similarity/2" do
    test "computes cosine similarity matrix" do
      embeddings = [
        Embedding.new([1.0, 0.0], "model", :provider),
        Embedding.new([0.0, 1.0], "model", :provider),
        Embedding.new([1.0, 0.0], "model", :provider)
      ]

      matrix = Similarity.pairwise_similarity(embeddings, metric: :cosine)

      assert Nx.shape(matrix) == {3, 3}

      # Diagonal should be 1.0 (self-similarity)
      assert_in_delta Nx.to_number(matrix[0][0]), 1.0, 0.001
      assert_in_delta Nx.to_number(matrix[1][1]), 1.0, 0.001
      assert_in_delta Nx.to_number(matrix[2][2]), 1.0, 0.001

      # [1,0] and [0,1] should be orthogonal (similarity 0)
      assert_in_delta Nx.to_number(matrix[0][1]), 0.0, 0.001
      assert_in_delta Nx.to_number(matrix[1][0]), 0.0, 0.001

      # [1,0] and [1,0] should be identical (similarity 1)
      assert_in_delta Nx.to_number(matrix[0][2]), 1.0, 0.001
      assert_in_delta Nx.to_number(matrix[2][0]), 1.0, 0.001
    end

    test "computes euclidean distance matrix" do
      embeddings = [
        Embedding.new([0.0, 0.0], "model", :provider),
        Embedding.new([3.0, 4.0], "model", :provider)
      ]

      matrix = Similarity.pairwise_similarity(embeddings, metric: :euclidean)

      assert Nx.shape(matrix) == {2, 2}

      # Diagonal should be 0.0 (distance to self)
      assert_in_delta Nx.to_number(matrix[0][0]), 0.0, 0.001
      assert_in_delta Nx.to_number(matrix[1][1]), 0.0, 0.001

      # Distance between [0,0] and [3,4] should be 5.0
      assert_in_delta Nx.to_number(matrix[0][1]), 5.0, 0.001
      assert_in_delta Nx.to_number(matrix[1][0]), 5.0, 0.001
    end
  end
end
