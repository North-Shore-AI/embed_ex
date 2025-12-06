defmodule EmbedEx.Providers.OpenAI do
  @moduledoc """
  OpenAI embeddings provider.

  Supports OpenAI's embedding models via their API.

  ## Configuration

  Set the `OPENAI_API_KEY` environment variable or pass `:api_key` in options.

  ## Supported Models

    * `text-embedding-3-small` (default) - 1536 dimensions, cost-effective
    * `text-embedding-3-large` - 3072 dimensions, highest quality
    * `text-embedding-ada-002` - 1536 dimensions, legacy model

  ## Examples

      # Using default model
      {:ok, embedding} = EmbedEx.Providers.OpenAI.embed("Hello world")

      # Using specific model
      {:ok, embedding} = EmbedEx.Providers.OpenAI.embed(
        "Hello world",
        model: "text-embedding-3-large"
      )

      # Batch embedding
      {:ok, embeddings} = EmbedEx.Providers.OpenAI.embed_batch([
        "First text",
        "Second text",
        "Third text"
      ])
  """

  @behaviour EmbedEx.Provider

  alias EmbedEx.Embedding

  @default_model "text-embedding-3-small"
  @max_batch_size 2048
  @api_url "https://api.openai.com/v1/embeddings"

  @available_models [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002"
  ]

  @impl true
  def embed(text, opts \\ []) do
    with {:ok, api_key} <- get_api_key(opts),
         {:ok, response} <- make_request([text], api_key, opts) do
      [embedding_data] = response["data"]

      embedding =
        Embedding.new(
          embedding_data["embedding"],
          opts[:model] || @default_model,
          :openai,
          text: text,
          metadata: %{
            index: embedding_data["index"],
            usage: response["usage"]
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
        response["data"]
        |> Enum.sort_by(& &1["index"])
        |> Enum.zip(texts)
        |> Enum.map(fn {embedding_data, text} ->
          Embedding.new(
            embedding_data["embedding"],
            opts[:model] || @default_model,
            :openai,
            text: text,
            metadata: %{
              index: embedding_data["index"],
              usage: response["usage"]
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
    case Keyword.get(opts, :api_key) || System.get_env("OPENAI_API_KEY") do
      nil -> {:error, :missing_api_key}
      key when is_binary(key) -> {:ok, key}
    end
  end

  defp make_request(texts, api_key, opts) do
    model = opts[:model] || @default_model

    body = %{
      input: texts,
      model: model
    }

    # Add optional dimensions parameter for newer models
    body =
      case opts[:dimensions] do
        nil -> body
        dims when is_integer(dims) -> Map.put(body, :dimensions, dims)
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
