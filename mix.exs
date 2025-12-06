defmodule EmbedEx.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/North-Shore-AI/embed_ex"

  def project do
    [
      app: :embed_ex,
      version: @version,
      name: "EmbedEx",
      source_url: @source_url,
      homepage_url: @source_url,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs(),
      description: description(),
      package: package()
    ]
  end

  def version, do: @version

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {EmbedEx.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.7"},
      {:req, "~> 0.4"},
      {:jason, "~> 1.4"},
      {:cachex, "~> 3.6"},
      {:ex_doc, "~> 0.31", only: :dev, runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:stream_data, "~> 1.1", only: :test}
    ]
  end

  defp description do
    "Vector embeddings service for the NSAI ecosystem"
  end

  defp docs do
    [
      main: "readme",
      source_ref: "v#{@version}",
      source_url: @source_url,
      assets: %{"assets" => "assets"},
      logo: "assets/embed_ex.svg",
      extras: ["README.md", "LICENSE"]
    ]
  end

  defp package do
    [
      name: "embed_ex",
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      files: ~w(lib mix.exs README.md LICENSE assets)
    ]
  end
end
