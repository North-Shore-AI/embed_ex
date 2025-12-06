defmodule EmbedEx.CacheTest do
  use ExUnit.Case, async: false

  alias EmbedEx.{Cache, Embedding}

  setup do
    # Clear cache before each test
    Cache.clear()
    :ok
  end

  describe "fetch/3" do
    test "computes value on cache miss" do
      result =
        Cache.fetch("test text", [model: "test-model"], fn ->
          Embedding.new([0.1, 0.2], "test-model", :test, text: "test text")
        end)

      assert %Embedding{} = result
    end

    test "returns cached value on cache hit" do
      # First call - cache miss
      call_count = :counters.new(1, [])

      Cache.fetch("test text", [model: "test-model"], fn ->
        :counters.add(call_count, 1, 1)
        Embedding.new([0.1, 0.2], "test-model", :test, text: "test text")
      end)

      # Second call - should be cache hit
      Cache.fetch("test text", [model: "test-model"], fn ->
        :counters.add(call_count, 1, 1)
        Embedding.new([0.1, 0.2], "test-model", :test, text: "test text")
      end)

      # Function should only be called once
      assert :counters.get(call_count, 1) == 1
    end

    test "different texts have different cache keys" do
      Cache.fetch("text1", [model: "test-model"], fn ->
        Embedding.new([0.1], "test-model", :test, text: "text1")
      end)

      Cache.fetch("text2", [model: "test-model"], fn ->
        Embedding.new([0.2], "test-model", :test, text: "text2")
      end)

      {:ok, emb1} = Cache.get("text1", model: "test-model")
      {:ok, emb2} = Cache.get("text2", model: "test-model")

      assert emb1.text == "text1"
      assert emb2.text == "text2"
    end

    test "different models have different cache keys" do
      Cache.fetch("text", [model: "model1"], fn ->
        Embedding.new([0.1], "model1", :test, text: "text")
      end)

      Cache.fetch("text", [model: "model2"], fn ->
        Embedding.new([0.2], "model2", :test, text: "text")
      end)

      {:ok, emb1} = Cache.get("text", model: "model1")
      {:ok, emb2} = Cache.get("text", model: "model2")

      assert emb1.model == "model1"
      assert emb2.model == "model2"
    end
  end

  describe "put/2" do
    test "stores embedding in cache" do
      embedding = Embedding.new([0.1, 0.2], "test-model", :test, text: "hello")

      :ok = Cache.put(embedding, model: "test-model")

      {:ok, cached} = Cache.get("hello", model: "test-model")
      assert cached.vector == [0.1, 0.2]
    end
  end

  describe "get/2" do
    test "returns error when key not in cache" do
      assert {:error, :not_found} = Cache.get("nonexistent", model: "test-model")
    end

    test "returns cached embedding when present" do
      embedding = Embedding.new([0.1, 0.2], "test-model", :test, text: "hello")
      Cache.put(embedding, model: "test-model")

      {:ok, cached} = Cache.get("hello", model: "test-model")
      assert cached.vector == [0.1, 0.2]
    end
  end

  describe "clear/0" do
    test "removes all cached embeddings" do
      Cache.put(Embedding.new([0.1], "model", :test, text: "text1"), model: "model")
      Cache.put(Embedding.new([0.2], "model", :test, text: "text2"), model: "model")

      {:ok, _} = Cache.clear()

      assert {:error, :not_found} = Cache.get("text1", model: "model")
      assert {:error, :not_found} = Cache.get("text2", model: "model")
    end
  end

  describe "stats/0" do
    test "returns cache statistics" do
      {:ok, stats} = Cache.stats()
      assert is_map(stats)
    end
  end
end
