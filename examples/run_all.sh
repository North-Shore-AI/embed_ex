#!/bin/bash

# Run all EmbedEx examples
# This script runs examples with available providers, skipping unavailable ones

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "========================================"
echo "EmbedEx Examples Runner"
echo "========================================"
echo ""

# Check for Ollama
OLLAMA_AVAILABLE=false
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    OLLAMA_AVAILABLE=true
    echo "✓ Ollama is available"
else
    echo "✗ Ollama not available (install and run: ollama serve)"
fi

# Check for OpenAI API key
OPENAI_AVAILABLE=false
if [ -n "$OPENAI_API_KEY" ]; then
    OPENAI_AVAILABLE=true
    echo "✓ OpenAI API key is set"
else
    echo "✗ OpenAI API key not set (export OPENAI_API_KEY=sk-...)"
fi

echo ""
echo "========================================"
echo ""

# Run Ollama examples (local, no API key needed)
if [ "$OLLAMA_AVAILABLE" = true ]; then
    echo "Running: ollama_embeddings.exs"
    echo "----------------------------------------"
    mix run examples/ollama_embeddings.exs
    echo ""
    echo "========================================"
    echo ""
else
    echo "Skipping ollama_embeddings.exs (Ollama not available)"
    echo ""
fi

# Run basic usage examples (requires OpenAI)
if [ "$OPENAI_AVAILABLE" = true ]; then
    echo "Running: basic_usage.exs"
    echo "----------------------------------------"
    mix run examples/basic_usage.exs
    echo ""
    echo "========================================"
    echo ""
else
    echo "Skipping basic_usage.exs (OpenAI API key not set)"
    echo ""
fi

echo ""
echo "All available examples completed!"
echo ""
echo "To run more examples, ensure the required services are available:"
if [ "$OLLAMA_AVAILABLE" = false ]; then
    echo "  - Ollama: brew install ollama && ollama serve && ollama pull nomic-embed-text"
fi
if [ "$OPENAI_AVAILABLE" = false ]; then
    echo "  - OpenAI: export OPENAI_API_KEY=sk-..."
fi
