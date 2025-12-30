# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work. This project does **not** accept AI-generated pull requests - AI tools may only be used in an assistive capacity.

## Build Commands

```bash
# CPU-only build
cmake -B build
cmake --build build --config Release

# Faster parallel compilation
cmake --build build --config Release -j 8

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Common GPU backends
cmake -B build -DGGML_CUDA=ON              # NVIDIA CUDA
cmake -B build -DGGML_HIP=ON               # AMD ROCm
cmake -B build -DGGML_VULKAN=ON            # Vulkan
cmake -B build -DGGML_METAL=ON             # macOS Metal (enabled by default on Mac)

# Static build
cmake -B build -DBUILD_SHARED_LIBS=OFF
```

## Running Tests

```bash
# Run full CI locally
mkdir tmp
bash ./ci/run.sh ./tmp/results ./tmp/mnt

# With CUDA support
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# Run CTest after building
ctest --test-dir build

# Server tests (pytest-based)
cd tools/server/tests
pip install -r requirements.txt
./tests.sh                                    # Run all tests
./tests.sh unit/test_chat_completion.py -v    # Run specific test file
./tests.sh unit/test_chat_completion.py::test_name  # Run single test
DEBUG=1 ./tests.sh -s -v -x                   # Verbose debugging
```

## Key Binaries

After building, binaries are in `build/bin/`:
- `llama-cli` - Interactive CLI for chat/completions
- `llama-server` - OpenAI-compatible HTTP server
- `llama-bench` - Performance benchmarking
- `llama-perplexity` - Model quality evaluation
- `llama-quantize` - Model quantization

## Architecture Overview

### Core Libraries

- **ggml** (`ggml/`) - Low-level tensor library handling computation across backends (CPU, CUDA, Metal, Vulkan, etc.). Each backend in `ggml/src/ggml-*/` implements the same tensor operations.

- **llama** (`src/`, `include/llama.h`) - High-level LLM inference library. Key components:
  - `llama-model.*` - Model loading and tensor management
  - `llama-context.*` - Inference context and state
  - `llama-kv-cache.*` - Key-value cache management
  - `llama-vocab.*` - Tokenization
  - `llama-sampling.*` - Token sampling strategies
  - `llama-grammar.*` - Grammar-constrained generation
  - `llama-arch.*` - Model architecture definitions

- **common** (`common/`) - Shared utilities for tools/examples:
  - `common.*` - General utilities
  - `arg.*` - Argument parsing
  - `sampling.*` - High-level sampling interface
  - `chat.*` - Chat template handling

### Main Tools

Located in `tools/`:
- `server/` - HTTP server with OpenAI-compatible API (SvelteKit WebUI in `webui/`)
- `cli/` - Interactive command-line interface
- `quantize/` - Model quantization
- `perplexity/` - Model evaluation
- `llama-bench/` - Performance benchmarks
- `mtmd/` - Multimodal support library

### Model Conversion

Python scripts in root directory:
- `convert_hf_to_gguf.py` - Convert HuggingFace models to GGUF format
- GGUF format definitions in `gguf-py/`

## Coding Guidelines

- Use `snake_case` for functions, variables, and types
- Use sized integer types (`int32_t`, `size_t`) in public API
- Avoid third-party dependencies
- Use basic constructs over fancy STL; keep it simple
- 4 spaces indentation, brackets on same line
- Pointer/reference style: `void * ptr`, `int & a`
- Vertical alignment for readability

### Naming Conventions

- Pattern: `<class>_<method>` where method is `<action>_<noun>`
  - Example: `llama_model_init()`, `llama_sampler_get_seed()`
- Enum values: uppercase, prefixed with enum name
  - Example: `LLAMA_VOCAB_TYPE_BPE`
- Optimize for longest common prefix in related names

### Matrix Operations

ggml uses unconventional matrix multiplication: `C = ggml_mul_mat(ctx, A, B)` computes C^T = A B^T (equivalently C = B A^T).

## Adding New Model Support

See `docs/development/HOWTO-add-model.md`. Key steps:
1. Add GGUF conversion in `convert_hf_to_gguf.py`
2. Define architecture enum in `src/llama-arch.h`
3. Add tensor mappings in `src/llama-arch.cpp` and `gguf-py/gguf/constants.py`
4. Implement model graph in `src/llama-model.cpp`

## Related Documentation

- [Build documentation](docs/build.md)
- [Server development](tools/server/README-dev.md)
- [Contributing guidelines](CONTRIBUTING.md)
