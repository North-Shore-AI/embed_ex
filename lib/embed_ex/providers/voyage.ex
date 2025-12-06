defmodule EmbedEx.Providers.Voyage do
  @moduledoc """
  Voyage AI embeddings provider.

  Supports Voyage AI's high-performance embedding models.

  ## Configuration

  Set the `VOYAGE_API_KEY` environment variable or pass `:api_key` in options.

  ## Supported Models

    * `voyage-3` (default) - Latest general-purpose model, 1024 dimensions
    * `voyage-3-lite` - Faster, lighter model, 512 dimensions
    * `voyage-code-3` - Optimized for code, 1024 dimensions
    * `voyage-finance-2` - Domain-specific for finance
    * `voyage-law-2` - Domain-specific for legal text
    * `voyage-multilingual-2` - Multilingual support

  ## Input Types

  Voyage supports input types to optimize embeddings:
    * `:document` - For documents to be retrieved
    * `:query` - For search queries

  ## Examples

      # Using default model
      {:ok, embedding} = EmbedEx.Providers.Voyage.embed("Hello world")

      # Using specific model and input type
      {:ok, embedding} = EmbedEx.Providers.Voyage.embed(
        "Hello world",
        model: "voyage-code-3",
        input_type: :query
      )

      # Batch embedding with truncation
      {:ok, embeddings} = EmbedEx.Providers.Voyage.embed_batch(
        ["First text", "Second text"],
        input_type: :document,
        truncation: true
      )
  """

  @behaviour EmbedEx.Provider

  alias EmbedEx.Embedding

  @default_model "voyage-3"
  @max_batch_size 128
  @api_url "https://api.voyageai.com/v1/embeddings"

  @available_models [
    "voyage-3",
    "voyage-3-lite",
    "voyage-code-3",
    "voyage-finance-2",
    "voyage-law-2",
    "voyage-multilingual-2"
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
          :voyage,
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
            :voyage,
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
    case Keyword.get(opts, :api_key) || System.get_env("VOYAGE_API_KEY") do
      nil -> {:error, :missing_api_key}
      key when is_binary(key) -> {:ok, key}
    end
  end

  defp make_request(texts, api_key, opts) do
    model = opts[:model] || @default_model
    input_type = opts[:input_type]
    truncation = opts[:truncation]

    body = %{
      input: texts,
      model: model
    }

    # Add optional parameters
    body =
      body
      |> maybe_add(:input_type, input_type)
      |> maybe_add(:truncation, truncation)

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

  defp maybe_add(map, _key, nil), do: map

  defp maybe_add(map, key, value) when is_atom(value) do
    Map.put(map, key, Atom.to_string(value))
  end

  defp maybe_add(map, key, value) do
    Map.put(map, key, value)
  end
end
