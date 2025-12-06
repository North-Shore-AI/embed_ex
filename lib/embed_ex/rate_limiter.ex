defmodule EmbedEx.RateLimiter do
  @moduledoc """
  Rate limiting and exponential backoff for API requests.

  Implements token bucket algorithm with exponential backoff for failed requests.

  ## Configuration

      config :embed_ex, :rate_limiter,
        # Tokens per second
        tokens_per_second: 10,
        # Maximum burst size
        burst_size: 20,
        # Initial backoff delay (milliseconds)
        initial_backoff: 1000,
        # Maximum backoff delay (milliseconds)
        max_backoff: 60_000,
        # Backoff multiplier
        backoff_multiplier: 2

  ## Examples

      # Execute with rate limiting
      EmbedEx.RateLimiter.execute(fn ->
        make_api_request()
      end)

      # Execute with custom retry policy
      EmbedEx.RateLimiter.execute(
        fn -> make_api_request() end,
        max_retries: 5,
        backoff_multiplier: 1.5
      )
  """

  use GenServer
  require Logger

  @type state :: %{
          tokens: float(),
          last_refill: integer(),
          tokens_per_second: float(),
          burst_size: pos_integer()
        }

  # Client API

  @doc """
  Starts the rate limiter GenServer.
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Executes a function with rate limiting and exponential backoff.

  ## Options

    * `:max_retries` - Maximum number of retries (default: 3)
    * `:initial_backoff` - Initial backoff delay in ms (default: 1000)
    * `:max_backoff` - Maximum backoff delay in ms (default: 60_000)
    * `:backoff_multiplier` - Backoff multiplier (default: 2)
    * `:retryable_errors` - List of error patterns to retry (default: [:rate_limit, :timeout, :server_error])

  Returns the result of the function or `{:error, reason}` after max retries.
  """
  @spec execute((-> term()), keyword()) :: term() | {:error, term()}
  def execute(fun, opts \\ []) when is_function(fun, 0) do
    max_retries = Keyword.get(opts, :max_retries, 3)
    do_execute(fun, opts, 0, max_retries)
  end

  @doc """
  Acquires a token from the rate limiter.

  Blocks until a token is available. Returns `:ok` when a token is acquired.
  """
  @spec acquire() :: :ok
  def acquire do
    GenServer.call(__MODULE__, :acquire, :infinity)
  end

  @doc """
  Tries to acquire a token without blocking.

  Returns `:ok` if a token is available, `{:error, :rate_limited}` otherwise.
  """
  @spec try_acquire() :: :ok | {:error, :rate_limited}
  def try_acquire do
    GenServer.call(__MODULE__, :try_acquire)
  end

  @doc """
  Returns the current number of available tokens.
  """
  @spec available_tokens() :: float()
  def available_tokens do
    GenServer.call(__MODULE__, :available_tokens)
  end

  # Server callbacks

  @impl true
  def init(opts) do
    config = get_config()
    tokens_per_second = Keyword.get(opts, :tokens_per_second, config[:tokens_per_second])
    burst_size = Keyword.get(opts, :burst_size, config[:burst_size])

    state = %{
      tokens: burst_size,
      last_refill: System.monotonic_time(:millisecond),
      tokens_per_second: tokens_per_second,
      burst_size: burst_size
    }

    Logger.info("Rate limiter started: #{tokens_per_second} tokens/sec, burst: #{burst_size}")
    {:ok, state}
  end

  @impl true
  def handle_call(:acquire, _from, state) do
    state = refill_tokens(state)

    if state.tokens >= 1 do
      new_state = %{state | tokens: state.tokens - 1}
      {:reply, :ok, new_state}
    else
      # Calculate wait time
      wait_time = trunc(1000 / state.tokens_per_second)
      Process.sleep(wait_time)
      handle_call(:acquire, nil, refill_tokens(state))
    end
  end

  @impl true
  def handle_call(:try_acquire, _from, state) do
    state = refill_tokens(state)

    if state.tokens >= 1 do
      new_state = %{state | tokens: state.tokens - 1}
      {:reply, :ok, new_state}
    else
      {:reply, {:error, :rate_limited}, state}
    end
  end

  @impl true
  def handle_call(:available_tokens, _from, state) do
    state = refill_tokens(state)
    {:reply, state.tokens, state}
  end

  # Private functions

  defp do_execute(fun, opts, retry_count, max_retries) do
    :ok = acquire()

    case fun.() do
      {:ok, _} = success ->
        success

      {:error, reason} = error ->
        if should_retry?(reason, retry_count, max_retries, opts) do
          backoff_delay = calculate_backoff(retry_count, opts)

          Logger.warning(
            "Request failed (#{inspect(reason)}), retrying in #{backoff_delay}ms (attempt #{retry_count + 1}/#{max_retries})"
          )

          Process.sleep(backoff_delay)
          do_execute(fun, opts, retry_count + 1, max_retries)
        else
          error
        end

      other ->
        other
    end
  end

  defp should_retry?(reason, retry_count, max_retries, opts) do
    if retry_count >= max_retries do
      false
    else
      retryable_errors =
        Keyword.get(opts, :retryable_errors, [:rate_limit, :timeout, :server_error])

      error_retryable?(reason, retryable_errors)
    end
  end

  defp error_retryable?({:http_error, status, _}, _retryable)
       when status in [429, 500, 502, 503, 504] do
    true
  end

  defp error_retryable?(reason, retryable_errors) when is_atom(reason) do
    reason in retryable_errors
  end

  defp error_retryable?({reason, _}, retryable_errors) when is_atom(reason) do
    reason in retryable_errors
  end

  defp error_retryable?(_, _), do: false

  defp calculate_backoff(retry_count, opts) do
    initial = Keyword.get(opts, :initial_backoff, 1000)
    max_backoff = Keyword.get(opts, :max_backoff, 60_000)
    multiplier = Keyword.get(opts, :backoff_multiplier, 2)

    backoff = trunc(initial * :math.pow(multiplier, retry_count))
    min(backoff, max_backoff)
  end

  defp refill_tokens(state) do
    now = System.monotonic_time(:millisecond)
    elapsed = now - state.last_refill
    tokens_to_add = elapsed * state.tokens_per_second / 1000
    new_tokens = min(state.tokens + tokens_to_add, state.burst_size)

    %{state | tokens: new_tokens, last_refill: now}
  end

  defp get_config do
    Application.get_env(:embed_ex, :rate_limiter, [])
    |> Keyword.put_new(:tokens_per_second, 10)
    |> Keyword.put_new(:burst_size, 20)
  end
end
