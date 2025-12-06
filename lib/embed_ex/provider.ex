defmodule EmbedEx.Provider do
  @moduledoc """
  Behaviour for embedding providers.

  All embedding providers must implement this behaviour to be used by EmbedEx.
  """

  alias EmbedEx.Embedding

  @doc """
  Embeds a single text string.

  ## Options

    * `:model` - The model to use (provider-specific)
    * Any other provider-specific options

  Returns `{:ok, %Embedding{}}` or `{:error, reason}`.
  """
  @callback embed(text :: String.t(), opts :: keyword()) ::
              {:ok, Embedding.t()} | {:error, term()}

  @doc """
  Embeds a batch of text strings.

  ## Options

    * `:model` - The model to use (provider-specific)
    * `:batch_size` - Maximum batch size (provider-specific)
    * Any other provider-specific options

  Returns `{:ok, [%Embedding{}]}` or `{:error, reason}`.
  """
  @callback embed_batch(texts :: [String.t()], opts :: keyword()) ::
              {:ok, [Embedding.t()]} | {:error, term()}

  @doc """
  Returns the default model for this provider.
  """
  @callback default_model() :: String.t()

  @doc """
  Returns the maximum batch size supported by this provider.
  """
  @callback max_batch_size() :: pos_integer()

  @doc """
  Returns the list of available models for this provider.
  """
  @callback available_models() :: [String.t()]

  @doc """
  Validates provider-specific configuration.

  Returns `:ok` or `{:error, reason}`.
  """
  @callback validate_config(opts :: keyword()) :: :ok | {:error, term()}
end
