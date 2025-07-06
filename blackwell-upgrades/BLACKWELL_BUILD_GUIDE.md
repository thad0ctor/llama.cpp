# Blackwell RTX 5090 Optimization Build Guide
*llama.cpp Enhanced Performance for NVIDIA Blackwell Architecture*

> **Target Hardware**: NVIDIA RTX 5090 (Blackwell architecture)  
> **CUDA Version**: 12.8+ (required for cluster features)  
> **Build System**: CMake 3.18+  
> **Performance Goal**: 2-4x memory bandwidth improvement, 128K+ context support

---

## Prerequisites & System Requirements

### Hardware Requirements

**Primary Target**:
- **NVIDIA RTX 5090** (Blackwell architecture, compute capability 12.0)
- **Memory**: 32GB+ system RAM, 32GB VRAM
- **Storage**: NVMe SSD recommended for model loading

**Multi-GPU Setup** (Optional):
- **2-4x RTX 5090** for tensor parallelism
- **NVLink/NVSwitch**: For optimal multi-GPU communication
- **System RAM**: 64GB+ for multi-GPU configurations

**Compatibility Matrix**:
| GPU Model | Compute Capability | Cluster Support | HBM3 Optimization | Status |
|-----------|-------------------|-----------------|-------------------|--------|
| RTX 5090 | 12.0 | ✅ Full | ✅ Full | Primary Target |
| RTX 4090 | 8.9 | ❌ No | ⚠️ Limited | Fallback Mode |
| RTX 3090 | 8.6 | ❌ No | ❌ No | Legacy Mode |

### Software Requirements

**CUDA Toolkit**:
- **Version**: 12.8 or later (required for Blackwell cluster features)
- **Components**: CUDA Runtime, cuBLAS, NCCL (for multi-GPU)
- **Download**: https://developer.nvidia.com/cuda-toolkit

**Build Dependencies**:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    cmake \
    build-essential \
    libcublas-dev \
    libnccl-dev \
    python3-dev \
    python3-pip

# CentOS/RHEL
sudo yum install -y \
    cmake3 \
    gcc-c++ \
    cuda-cublas-devel \
    nccl-devel \
    python3-devel
```

**Python Dependencies** (for conversion/testing):
```bash
pip3 install torch numpy sentencepiece transformers
```

---

## Build Configuration

### Basic Build (Single GPU)

```bash
# Clone and setup
git clone https://github.com/your-repo/llama.cpp.git
cd llama.cpp
git checkout blackwell-development

# Create build directory
mkdir build && cd build

# Configure with Blackwell optimizations
cmake .. \
    -DLLAMA_CUDA=ON \
    -DLLAMA_BLACKWELL_OPTIMIZATIONS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="89;90;12" \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8

# Build
make -j$(nproc)
```

### Advanced Build (Multi-GPU + All Features)

```bash
# Full-featured build with all optimizations
cmake .. \
    -DLLAMA_CUDA=ON \
    -DLLAMA_BLACKWELL_OPTIMIZATIONS=ON \
    -DLLAMA_BLACKWELL_CLUSTER_GEMM=ON \
    -DLLAMA_BLACKWELL_FLASH_ATTENTION=ON \
    -DLLAMA_BLACKWELL_HBM3_OPTIMIZATIONS=ON \
    -DLLAMA_BLACKWELL_TENSOR_PARALLEL=ON \
    -DLLAMA_NCCL=ON \
    -DLLAMA_FLASH_ATTENTION=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="89;90;12" \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 \
    -DCMAKE_CUDA_FLAGS="-O3 -use_fast_math -maxrregcount=255"

# Build with maximum parallelism
make -j$(nproc)
```

### CMake Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `LLAMA_BLACKWELL_OPTIMIZATIONS` | OFF | Enable all Blackwell optimizations |
| `LLAMA_BLACKWELL_CLUSTER_GEMM` | OFF | Enable cluster-based GEMM kernels |
| `LLAMA_BLACKWELL_FLASH_ATTENTION` | OFF | Enable L2-optimized Flash Attention |
| `LLAMA_BLACKWELL_HBM3_OPTIMIZATIONS` | OFF | Enable HBM3e bandwidth optimizations |
| `LLAMA_BLACKWELL_TENSOR_PARALLEL` | OFF | Enable enhanced tensor parallelism |
| `LLAMA_BLACKWELL_DEBUG` | OFF | Enable debug logging and validation |
| `LLAMA_BLACKWELL_FORCE_ENABLE` | OFF | Force enable on non-Blackwell GPUs |

---

## Environment Configuration

### CUDA Environment Variables

```bash
# Set in ~/.bashrc or ~/.zshrc
export CUDA_PATH=/usr/local/cuda-12.8
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Blackwell-specific optimizations
export GGML_CUDA_FORCE_BLACKWELL=1              # Force Blackwell optimizations
export GGML_CUDA_CLUSTER_SIZE=4                 # Cluster size for multi-CTA kernels
export GGML_CUDA_HBM3_COALESCING=16            # Memory coalescing factor
export GGML_CUDA_L2_CACHE_SIZE=134217728       # L2 cache size (128MB)
export GGML_CUDA_TENSOR_SPLIT=1.0              # Single GPU tensor split
export GGML_CUDA_ASYNC_PIPELINE=1              # Enable async pipeline
export GGML_CUDA_PEER_ACCESS=1                 # Enable P2P for multi-GPU
```

### Multi-GPU Configuration

```bash
# For 2-4 GPU setups
export GGML_CUDA_TENSOR_SPLIT=0.25,0.25,0.25,0.25  # 4-way split
export GGML_CUDA_MAIN_GPU=0                         # Primary GPU
export GGML_CUDA_PEER_ACCESS=1                      # Enable P2P
export NCCL_DEBUG=INFO                              # NCCL debugging
export NCCL_IB_DISABLE=1                           # Disable InfiniBand if not available
```

---

## Compilation Flags & Tuning

### CUDA Compiler Flags

```bash
# High-performance flags
-O3                    # Maximum optimization
-use_fast_math         # Fast math operations
-maxrregcount=255      # Maximum register usage
-lineinfo              # Line info for profiling
-Xptxas -v             # Verbose PTX assembly
-Xptxas -dlcm=ca       # L1 cache as cache (not shared memory)

# Blackwell-specific flags
-arch=sm_90            # Ada Lovelace compatibility
-arch=sm_12            # Blackwell native (if available)
-gencode=arch=compute_89,code=sm_89   # RTX 4090 fallback
-gencode=arch=compute_90,code=sm_90   # RTX 5090 optimization
```

### Memory Management Tuning

```bash
# GPU memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0                    # Async kernel launches
export CUDA_DEVICE_ORDER=PCI_BUS_ID            # Consistent GPU ordering
export CUDA_VISIBLE_DEVICES=0,1,2,3            # Visible GPUs
```

---

## Testing & Validation

### Basic Functionality Test

```bash
# Test Blackwell optimizations
./llama-cli \
    --model models/llama-2-7b-chat.Q4_K_M.gguf \
    --prompt "Hello, world!" \
    --n-predict 100 \
    --ctx-size 2048 \
    --batch-size 512 \
    --threads 1 \
    --n-gpu-layers 35

# Expected output should include:
# [BLACKWELL] Using Blackwell-optimized kernels
# [BLACKWELL] Cluster GEMM enabled for device 0
# [BLACKWELL] HBM3 bandwidth optimizations active
```

### Performance Benchmarking

```bash
# Memory bandwidth test
./llama-bench \
    --model models/llama-2-7b-chat.Q4_K_M.gguf \
    --batch-size 512 \
    --ctx-size 4096 \
    --n-gen 512 \
    --n-gpu-layers 35

# Flash Attention performance test
./llama-bench \
    --model models/llama-2-7b-chat.Q4_K_M.gguf \
    --batch-size 1 \
    --ctx-size 32768 \
    --n-gen 128 \
    --n-gpu-layers 35 \
    --flash-attention
```

### Long Context Testing

```bash
# Test 128K context handling
./llama-cli \
    --model models/llama-2-7b-chat.Q4_K_M.gguf \
    --prompt "$(cat long_context.txt)" \
    --ctx-size 131072 \
    --batch-size 512 \
    --n-predict 100 \
    --n-gpu-layers 35

# Monitor memory usage
nvidia-smi -l 1
```

---

## Troubleshooting

### Common Issues & Solutions

**1. "Cluster GEMM not supported" Error**
```bash
# Check CUDA version
nvcc --version  # Should be 12.8+

# Check GPU compute capability
./llama-cli --help | grep -i blackwell

# Force enable for testing
export GGML_CUDA_FORCE_BLACKWELL=1
```

**2. Out of Memory Errors**
```bash
# Reduce batch size
./llama-cli --batch-size 256

# Use model sharding
export GGML_CUDA_TENSOR_SPLIT=0.5,0.5

# Enable KV cache quantization
./llama-cli --kv-cache-quantization int8
```

**3. Poor Performance**
```bash
# Check GPU utilization
nvidia-smi dmon -s pucvmet

# Verify P2P access (multi-GPU)
./llama-cli --check-p2p

# Profile memory access patterns
export GGML_CUDA_DEBUG=1
```

**4. Gibberish Output (CRITICAL)**
```bash
# Check if RTX 5090 is properly detected
./llama-cli --verbose 2>&1 | grep -i blackwell
# Should show: "[BLACKWELL] Detected RTX 5090 with 32GB VRAM - enabling all optimizations"

# If not detected, force enable Blackwell mode
export GGML_CUDA_FORCE_BLACKWELL=1

# Check for numerical instability (common cause of gibberish)
export GGML_CUDA_DISABLE_FAST_MATH=1

# Use higher precision if fast math causes issues
./llama-cli --precision fp16

# Verify model integrity
./llama-cli --check-model

# Check for matrix dimension issues (leading cause of corruption)
export GGML_CUDA_DEBUG=1
./llama-cli --verbose 2>&1 | grep -E "(Invalid matrix|GEMM)"

# If still getting gibberish, disable Blackwell optimizations temporarily
export GGML_CUDA_DISABLE_BLACKWELL_OPTIMIZATIONS=1
```

### Debug Builds

```bash
# Debug build for troubleshooting
cmake .. \
    -DLLAMA_CUDA=ON \
    -DLLAMA_BLACKWELL_OPTIMIZATIONS=ON \
    -DLLAMA_BLACKWELL_DEBUG=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CUDA_FLAGS="-g -G -O0"

make -j$(nproc)

# Run with debugging
./llama-cli --debug --verbose
```

### Performance Profiling

```bash
# NVIDIA Nsight Systems profiling
nsys profile --trace=cuda,nvtx ./llama-cli \
    --model models/llama-2-7b-chat.Q4_K_M.gguf \
    --prompt "Test prompt" \
    --n-predict 100

# NVIDIA Nsight Compute profiling
ncu --set full ./llama-cli \
    --model models/llama-2-7b-chat.Q4_K_M.gguf \
    --prompt "Test prompt" \
    --n-predict 10
```

---

## Performance Optimization Tips

### Memory Optimization

```bash
# Optimal memory configuration for RTX 5090
export GGML_CUDA_L2_CACHE_SIZE=134217728      # 128MB L2 cache
export GGML_CUDA_HBM3_COALESCING=16          # 16-way coalescing
export GGML_CUDA_SHARED_MEM_SIZE=233472      # 228KB shared memory

# KV cache optimization
export GGML_CUDA_KV_CACHE_QUANTIZATION=int8  # INT8 quantization
export GGML_CUDA_KV_CACHE_COMPRESSION=1      # Enable compression
```

### Compute Optimization

```bash
# Tensor parallelism tuning
export GGML_CUDA_TENSOR_SPLIT=0.25,0.25,0.25,0.25  # 4-way split
export GGML_CUDA_ASYNC_PIPELINE=1                   # Async pipeline
export GGML_CUDA_CLUSTER_SIZE=4                     # Optimal cluster size

# Flash Attention tuning
export GGML_CUDA_FLASH_ATTENTION_THRESHOLD=2048     # Use FA for 2K+ contexts
export GGML_CUDA_FLASH_ATTENTION_BLOCK_SIZE=128     # Optimal block size
```

---

## Version Compatibility

### Supported CUDA Versions

| CUDA Version | Blackwell Support | Cluster Features | Notes |
|--------------|-------------------|------------------|--------|
| 12.8+ | ✅ Full | ✅ Full | Recommended |
| 12.6-12.7 | ⚠️ Limited | ❌ No | Missing cluster APIs |
| 12.0-12.5 | ⚠️ Limited | ❌ No | Basic Blackwell only |
| 11.x | ❌ No | ❌ No | Legacy mode only |

### Driver Compatibility

```bash
# Check driver version
nvidia-smi

# Minimum driver versions:
# RTX 5090: 560.x+ (for full Blackwell support)
# RTX 4090: 525.x+ (for basic compatibility)
```

---

## Known Issues & Fixes

### Critical Issues Fixed (January 2025)

**1. Thread Mapping Bug in GEMM Kernels**
- **Issue**: Incorrect thread-to-output mapping could cause memory corruption
- **Fix**: Reduced accumulator size from 8x8 to 4x4 per thread, fixed thread indexing
- **Symptoms**: Random gibberish output, memory access violations

**2. RTX 5090 Detection Failure**
- **Issue**: VRAM threshold was set to 24GB instead of 30GB for RTX 5090 (32GB)
- **Fix**: Updated detection threshold to properly identify RTX 5090
- **Symptoms**: Blackwell optimizations not enabled despite having RTX 5090

**3. Leading Dimension Mismatch**
- **Issue**: Matrix leading dimensions didn't match llama.cpp's expected layout
- **Fix**: Corrected leading dimensions to use tensor dimensions directly
- **Symptoms**: Incorrect matrix multiplication results, model degradation

**4. Matrix Access Pattern Errors**
- **Issue**: Shared memory indexing didn't match global memory layout
- **Fix**: Aligned shared memory access patterns with tensor layouts
- **Symptoms**: Numerical instability, incorrect computations

### Verification Commands

```bash
# Verify RTX 5090 detection
./llama-cli --verbose 2>&1 | grep "Detected RTX 5090"

# Check Blackwell optimization status
./llama-cli --verbose 2>&1 | grep "Blackwell.*enabled"

# Monitor for matrix dimension errors
export GGML_CUDA_DEBUG=1
./llama-cli --verbose 2>&1 | grep -E "(Invalid|dimension|corruption)"

# Performance validation
./llama-bench --model your-model.gguf --batch-size 512 --ctx-size 4096
```

---

## Support & Resources

### Documentation
- [NVIDIA Blackwell Architecture Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [llama.cpp CUDA Backend Documentation](../docs/backend/CUDA.md)
- [Flash Attention Implementation Guide](../docs/flash-attention.md)

### Community
- GitHub Issues: Report bugs and request features
- Discord: Real-time community support
- Reddit: r/LocalLLaMA performance discussions

### Benchmarks
- [RTX 5090 Performance Results](../benchmarks/rtx5090-results.md)
- [Multi-GPU Scaling Results](../benchmarks/multi-gpu-scaling.md)
- [Long Context Performance](../benchmarks/long-context-performance.md)

---

*Last updated: January 2025*
*Build Guide Version: 1.0* 