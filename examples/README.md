# EmbedEx Examples

This directory contains example scripts demonstrating EmbedEx functionality.

## Available Examples

### basic_usage.exs

Demonstrates core EmbedEx functionality using the OpenAI provider:
- Single embeddings
- Batch embeddings with progress tracking
- Similarity computations (cosine, euclidean)
- Finding similar embeddings
- Pairwise similarity matrices
- Cache statistics
- Provider information

**Prerequisites:**
- `OPENAI_API_KEY` environment variable set

**Run:**
```bash
export OPENAI_API_KEY=sk-...
mix run examples/basic_usage.exs
```

### ollama_embeddings.exs

Demonstrates local embeddings using the Ollama provider for PHI-safe processing:
- Ollama health check and model listing
- Single and batch embeddings
- Clinical text embedding (PHI-safe)
- Medical term similarity
- Finding similar clinical notes
- Model comparison (nomic-embed-text vs mxbai-embed-large)
- Pairwise similarity matrices
- Cache demonstration

**Prerequisites:**
1. Install Ollama: `brew install ollama` (macOS) or `curl -fsSL https://ollama.ai/install.sh | sh` (Linux)
2. Start server: `ollama serve`
3. Pull embedding model: `ollama pull nomic-embed-text`

**Run:**
```bash
mix run examples/ollama_embeddings.exs
```

## Running All Examples

Use the provided shell script to run all examples:

```bash
./examples/run_all.sh
```

This will run examples with available providers (skipping those that require unavailable services).

## Writing Your Own Examples

Examples should:
1. Use `#!/usr/bin/env elixir` shebang
2. Include clear prerequisites in comments
3. Handle errors gracefully
4. Print informative output
5. Demonstrate practical use cases
