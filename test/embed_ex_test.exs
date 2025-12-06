defmodule EmbedExTest do
  use ExUnit.Case
  doctest EmbedEx

  alias EmbedEx.Embedding

  test "providers/0 returns available providers" do
    providers = EmbedEx.providers()
    assert is_list(providers)
    assert length(providers) >= 1

    [openai | _] = providers
    assert openai.name == :openai
    assert is_list(openai.models)
    assert openai.max_batch_size > 0
  end

  test "cosine_similarity/2 delegates to Similarity module" do
    emb1 = Embedding.new([1.0, 0.0, 0.0], "model", :provider)
    emb2 = Embedding.new([1.0, 0.0, 0.0], "model", :provider)

    similarity = EmbedEx.cosine_similarity(emb1, emb2)
    assert_in_delta similarity, 1.0, 0.001
  end

  test "euclidean_distance/2 delegates to Similarity module" do
    emb1 = Embedding.new([0.0, 0.0], "model", :provider)
    emb2 = Embedding.new([3.0, 4.0], "model", :provider)

    distance = EmbedEx.euclidean_distance(emb1, emb2)
    assert_in_delta distance, 5.0, 0.001
  end

  test "dot_product/2 delegates to Similarity module" do
    emb1 = Embedding.new([1.0, 2.0, 3.0], "model", :provider)
    emb2 = Embedding.new([4.0, 5.0, 6.0], "model", :provider)

    dot = EmbedEx.dot_product(emb1, emb2)
    assert_in_delta dot, 32.0, 0.001
  end

  test "find_similar/3 delegates to Similarity module" do
    query = Embedding.new([1.0, 0.0], "model", :provider)
    corpus = [Embedding.new([1.0, 0.0], "model", :provider)]

    {:ok, results} = EmbedEx.find_similar(query, corpus, top_k: 1)
    assert length(results) == 1
  end
end
