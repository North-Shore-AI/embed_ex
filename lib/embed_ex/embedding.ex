defmodule EmbedEx.Embedding do
  @moduledoc """
  Represents a text embedding with metadata.

  ## Fields

    * `:text` - The original text that was embedded
    * `:vector` - The embedding vector (list of floats or Nx tensor)
    * `:model` - The model used to generate the embedding
    * `:provider` - The provider that generated the embedding (:openai, :cohere, :local, etc.)
    * `:dimensions` - Number of dimensions in the vector
    * `:metadata` - Additional metadata map
  """

  @type t :: %__MODULE__{
          text: String.t(),
          vector: list(float()) | Nx.Tensor.t(),
          model: String.t(),
          provider: atom(),
          dimensions: pos_integer(),
          metadata: map()
        }

  @enforce_keys [:vector, :model, :provider]
  defstruct [
    :text,
    :vector,
    :model,
    :provider,
    :dimensions,
    metadata: %{}
  ]

  @doc """
  Creates a new embedding struct.

  ## Examples

      iex> EmbedEx.Embedding.new([0.1, 0.2, 0.3], "text-embedding-3-small", :openai)
      %EmbedEx.Embedding{
        vector: [0.1, 0.2, 0.3],
        model: "text-embedding-3-small",
        provider: :openai,
        dimensions: 3
      }
  """
  @spec new(list(float()) | Nx.Tensor.t(), String.t(), atom(), keyword()) :: t()
  def new(vector, model, provider, opts \\ []) do
    dimensions = if is_list(vector), do: length(vector), else: Nx.size(vector)

    %__MODULE__{
      text: opts[:text],
      vector: vector,
      model: model,
      provider: provider,
      dimensions: dimensions,
      metadata: opts[:metadata] || %{}
    }
  end

  @doc """
  Converts the vector to an Nx tensor if it isn't already.

  ## Examples

      iex> embedding = EmbedEx.Embedding.new([0.1, 0.2, 0.3], "model", :provider)
      iex> tensor_embedding = EmbedEx.Embedding.to_tensor(embedding)
      iex> Nx.to_list(tensor_embedding.vector)
      [0.1, 0.2, 0.3]
  """
  @spec to_tensor(t()) :: t()
  def to_tensor(%__MODULE__{vector: vector} = embedding) when is_list(vector) do
    %{embedding | vector: Nx.tensor(vector)}
  end

  def to_tensor(%__MODULE__{} = embedding), do: embedding

  @doc """
  Converts the vector to a list if it isn't already.
  """
  @spec to_list(t()) :: t()
  def to_list(%__MODULE__{vector: %Nx.Tensor{} = vector} = embedding) do
    %{embedding | vector: Nx.to_list(vector)}
  end

  def to_list(%__MODULE__{} = embedding), do: embedding
end
