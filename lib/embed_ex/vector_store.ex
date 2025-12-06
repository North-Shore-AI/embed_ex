defmodule EmbedEx.VectorStore do
  @moduledoc """
  Behaviour for vector storage backends.

  Defines a common interface for storing and retrieving embeddings from various
  vector databases and storage systems.
  """

  alias EmbedEx.Embedding

  @type id :: String.t() | integer()
  @type filter :: map()
  @type search_result :: %{
          id: id(),
          embedding: Embedding.t(),
          score: float(),
          metadata: map()
        }

  @doc """
  Inserts an embedding into the store.

  Returns `{:ok, id}` where id is the assigned identifier, or `{:error, reason}`.
  """
  @callback insert(embedding :: Embedding.t(), opts :: keyword()) ::
              {:ok, id()} | {:error, term()}

  @doc """
  Inserts multiple embeddings into the store.

  Returns `{:ok, ids}` where ids is a list of assigned identifiers, or `{:error, reason}`.
  """
  @callback insert_batch(embeddings :: [Embedding.t()], opts :: keyword()) ::
              {:ok, [id()]} | {:error, term()}

  @doc """
  Retrieves an embedding by ID.

  Returns `{:ok, embedding}` or `{:error, :not_found}`.
  """
  @callback get(id :: id(), opts :: keyword()) :: {:ok, Embedding.t()} | {:error, term()}

  @doc """
  Retrieves multiple embeddings by IDs.

  Returns `{:ok, embeddings}` where embeddings is a list in the same order as ids.
  Missing embeddings are represented as `nil`.
  """
  @callback get_batch(ids :: [id()], opts :: keyword()) ::
              {:ok, [Embedding.t() | nil]} | {:error, term()}

  @doc """
  Deletes an embedding by ID.

  Returns `:ok` or `{:error, reason}`.
  """
  @callback delete(id :: id(), opts :: keyword()) :: :ok | {:error, term()}

  @doc """
  Deletes multiple embeddings by IDs.

  Returns `{:ok, count}` where count is the number of deleted embeddings.
  """
  @callback delete_batch(ids :: [id()], opts :: keyword()) ::
              {:ok, non_neg_integer()} | {:error, term()}

  @doc """
  Searches for similar embeddings.

  ## Options

    * `:top_k` - Number of results to return (default: 10)
    * `:metric` - Similarity metric (`:cosine`, `:euclidean`, `:dot_product`)
    * `:filter` - Metadata filter to apply
    * `:threshold` - Minimum similarity threshold

  Returns `{:ok, results}` where results is a list of search result maps.
  """
  @callback search(
              query :: Embedding.t() | list(float()),
              opts :: keyword()
            ) :: {:ok, [search_result()]} | {:error, term()}

  @doc """
  Returns the total number of embeddings in the store.
  """
  @callback count(opts :: keyword()) :: {:ok, non_neg_integer()} | {:error, term()}

  @doc """
  Clears all embeddings from the store.

  Returns `{:ok, count}` where count is the number of deleted embeddings.
  """
  @callback clear(opts :: keyword()) :: {:ok, non_neg_integer()} | {:error, term()}

  @doc """
  Returns store statistics and metadata.
  """
  @callback stats(opts :: keyword()) :: {:ok, map()} | {:error, term()}
end
