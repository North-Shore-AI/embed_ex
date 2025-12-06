defmodule EmbedEx.Providers.Cohere do
  @moduledoc """
  Cohere embeddings provider.

  Supports Cohere's embedding models via their API.

  ## Configuration

  Set the `COHERE_API_KEY` environment variable or pass `:api_key` in options.

  ## Supported Models

    * `embed-english-v3.0` (default) - English text embeddings
    * `embed-multilingual-v3.0` - Multilingual embeddings (100+ languages)
    * `embed-english-light-v3.0` - Lighter, faster English model
    * `embed-multilingual-light-v3.0` - Lighter multilingual model

  ## Input Types

  Cohere supports different input types that optimize the embeddings:
    * `:search_document` - For documents in a search corpus
    * `:search_query` - For search queries
    * `:classification` - For classification tasks
    * `:clustering` - For clustering tasks

  ## Examples

      # Using default model
      {:ok, embedding} = EmbedEx.Providers.Cohere.embed("Hello world")

      # Using specific model and input type
      {:ok, embedding} = EmbedEx.Providers.Cohere.embed(
        "Hello world",
        model: "embed-multilingual-v3.0",
        input_type: :search_query
      )

      # Batch embedding with truncation
      {:ok, embeddings} = EmbedEx.Providers.Cohere.embed_batch(
        ["First text", "Second text", "Third text"],
        input_type: :search_document,
        truncate: :end
      )
  """

  @behaviour EmbedEx.Provider

  alias EmbedEx.Embedding

  @default_model "embed-english-v3.0"
  @max_batch_size 96
  @api_url "https://api.cohere.ai/v1/embed"

  @available_models [
    "embed-english-v3.0",
    "embed-multilingual-v3.0",
    "embed-english-light-v3.0",
    "embed-multilingual-light-v3.0"
  ]

  @impl true
  def embed(text, opts \\ []) do
    with {:ok, api_key} <- get_api_key(opts),
         {:ok, response} <- make_request([text], api_key, opts) do
      [embedding_vector] = response["embeddings"]

      embedding =
        Embedding.new(
          embedding_vector,
          opts[:model] || @default_model,
          :cohere,
          text: text,
          metadata: %{
            id: response["id"],
            response_type: response["response_type"],
            texts: response["texts"]
          }
        )

      {:ok, embedding}
    end
  end

  @impl true
  def embed_batch(texts, opts \\ []) when is_list(texts) do
    with {:ok, api_key} <- get_api_key(opts),
         {:ok, response} <- make_request(texts, api_key, opts) do
      embeddings =
        response["embeddings"]
        |> Enum.zip(texts)
        |> Enum.map(fn {embedding_vector, text} ->
          Embedding.new(
            embedding_vector,
            opts[:model] || @default_model,
            :cohere,
            text: text,
            metadata: %{
              id: response["id"],
              response_type: response["response_type"]
            }
          )
        end)

      {:ok, embeddings}
    end
  end

  @impl true
  def default_model, do: @default_model

  @impl true
  def max_batch_size, do: @max_batch_size

  @impl true
  def available_models, do: @available_models

  @impl true
  def validate_config(opts) do
    case get_api_key(opts) do
      {:ok, _} -> :ok
      {:error, _} = error -> error
    end
  end

  # Private functions

  defp get_api_key(opts) do
    case Keyword.get(opts, :api_key) || System.get_env("COHERE_API_KEY") do
      nil -> {:error, :missing_api_key}
      key when is_binary(key) -> {:ok, key}
    end
  end

  defp make_request(texts, api_key, opts) do
    model = opts[:model] || @default_model
    input_type = opts[:input_type] || :search_document
    truncate = opts[:truncate] || :end

    body = %{
      texts: texts,
      model: model,
      input_type: Atom.to_string(input_type),
      truncate: Atom.to_string(truncate)
    }

    # Add optional embedding types parameter (for v3 models)
    body =
      case opts[:embedding_types] do
        nil -> body
        types when is_list(types) -> Map.put(body, :embedding_types, types)
      end

    case Req.post(@api_url,
           json: body,
           headers: [
             {"Authorization", "Bearer #{api_key}"},
             {"Content-Type", "application/json"}
           ],
           retry: :transient,
           max_retries: 3
         ) do
      {:ok, %{status: 200, body: body}} ->
        {:ok, body}

      {:ok, %{status: status, body: body}} ->
        {:error, {:http_error, status, body}}

      {:error, reason} ->
        {:error, reason}
    end
  end
end
