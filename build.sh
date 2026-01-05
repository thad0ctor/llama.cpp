#!/bin/bash
set -e

# Explicitly set the CUDA compiler path
export CUDACXX=/usr/local/cuda/bin/nvcc

# Force CUDA graph usage for flash attention
export GGML_CUDA_GRAPH_FORCE=1

# Use CMake 4.2.1 for 120f architecture support
CMAKE=/opt/cmake-4.2.1-linux-x86_64/bin/cmake

# Create timestamped output directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$SCRIPT_DIR/logs/build/$TIMESTAMP"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/build.log"

# Use a writable out-of-source build directory
BUILD_DIR="$SCRIPT_DIR/build"

# Redirect all output to both console and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Build started at $(date)"
echo "Log file: $LOG_FILE"
echo "Build dir: $BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring CMake with CUDA support (Blackwell optimized)..."
# Pass CMAKE_CUDA_COMPILER explicitly to avoid detection issues
# CMAKE_CUDA_ARCHITECTURES=120 targets Blackwell (sm_120/RTX 5090)
# - Automatically converted to 120a for PTX compatibility
# - Enables --maxrregcount=224 optimization (see ggml/src/ggml-cuda/CMakeLists.txt)
# - Enables Blackwell-specific TMA and warp specialization paths
$CMAKE .. \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FORCE_MMQ=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DCMAKE_CUDA_COMPILER=$CUDACXX \
    -DCMAKE_CUDA_ARCHITECTURES="120a-real;86-real" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_FLAGS="--use_fast_math -O3" \
    -DLLAMA_BUILD_TESTS=OFF \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_LTO=ON \
    -DGGML_CUDA_COMPRESSION_MODE=speed \
    -DGGML_CUDA_PEER_MAX_BATCH_SIZE=512 \
    -DGGML_CUDA_NO_PEER_COPY=OFF \
    -DGGML_CUDA_NO_VMM=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_STATIC=OFF 

echo "Building..."
$CMAKE --build . --config Release -j $(nproc) --target llama-server

echo "Build finished at $(date)"
echo "Script finished."