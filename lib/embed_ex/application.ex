defmodule EmbedEx.Application do
  @moduledoc """
  OTP Application for EmbedEx.

  Starts the cache GenServer and any other required services.
  """

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      # Start the cache GenServer
      {EmbedEx.Cache, []}
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: EmbedEx.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
