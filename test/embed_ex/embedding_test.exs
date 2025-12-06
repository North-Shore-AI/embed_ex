defmodule EmbedEx.EmbeddingTest do
  use ExUnit.Case, async: true

  alias EmbedEx.Embedding

  describe "new/4" do
    test "creates embedding with vector list" do
      vector = [0.1, 0.2, 0.3]
      embedding = Embedding.new(vector, "test-model", :test_provider)

      assert embedding.vector == vector
      assert embedding.model == "test-model"
      assert embedding.provider == :test_provider
      assert embedding.dimensions == 3
      assert embedding.metadata == %{}
    end

    test "creates embedding with options" do
      vector = [0.1, 0.2, 0.3]

      embedding =
        Embedding.new(vector, "test-model", :test_provider,
          text: "hello",
          metadata: %{foo: "bar"}
        )

      assert embedding.text == "hello"
      assert embedding.metadata == %{foo: "bar"}
    end

    test "calculates dimensions correctly" do
      vector = Enum.to_list(1..100)
      embedding = Embedding.new(vector, "test-model", :test_provider)

      assert embedding.dimensions == 100
    end
  end

  describe "to_tensor/1" do
    test "converts list vector to tensor" do
      vector = [0.1, 0.2, 0.3]
      embedding = Embedding.new(vector, "test-model", :test_provider)

      tensor_embedding = Embedding.to_tensor(embedding)

      assert %Nx.Tensor{} = tensor_embedding.vector

      # Compare with delta for floating point precision
      result = Nx.to_list(tensor_embedding.vector)

      Enum.zip(result, vector)
      |> Enum.each(fn {actual, expected} ->
        assert_in_delta actual, expected, 0.001
      end)
    end

    test "leaves tensor vector unchanged" do
      vector = Nx.tensor([0.1, 0.2, 0.3])
      embedding = %Embedding{vector: vector, model: "test", provider: :test, dimensions: 3}

      tensor_embedding = Embedding.to_tensor(embedding)

      assert tensor_embedding.vector == vector
    end
  end

  describe "to_list/1" do
    test "converts tensor vector to list" do
      vector = Nx.tensor([0.1, 0.2, 0.3])
      embedding = %Embedding{vector: vector, model: "test", provider: :test, dimensions: 3}

      list_embedding = Embedding.to_list(embedding)

      assert is_list(list_embedding.vector)

      # Compare with delta for floating point precision
      expected = [0.1, 0.2, 0.3]

      Enum.zip(list_embedding.vector, expected)
      |> Enum.each(fn {actual, expected} ->
        assert_in_delta actual, expected, 0.001
      end)
    end

    test "leaves list vector unchanged" do
      vector = [0.1, 0.2, 0.3]
      embedding = Embedding.new(vector, "test-model", :test_provider)

      list_embedding = Embedding.to_list(embedding)

      assert list_embedding.vector == vector
    end
  end
end
