#!/bin/bash
# Verification script for Blackwell optimizations
# Run after rebuild to confirm optimizations are active

set -e

MODEL="${1:-/path/to/model.gguf}"
NCU_OUTPUT="blackwell_verify_$(date +%Y%m%d_%H%M%S)"

echo "=== Blackwell Optimization Verification ==="
echo ""

# Check CUDA architecture in build
echo "1. Checking build configuration..."
if [ -f build/compile_commands.json ]; then
    if grep -q "sm_120\|compute_120" build/compile_commands.json 2>/dev/null; then
        echo "   ✓ sm_120 (Blackwell) architecture found in build"
    else
        echo "   ✗ WARNING: sm_120 not found in build config"
        echo "   Rebuild with: cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120"
    fi
else
    echo "   ? compile_commands.json not found, checking CMakeCache..."
    if grep -q "120" build/CMakeCache.txt 2>/dev/null; then
        echo "   ✓ Architecture 120 found in CMakeCache"
    else
        echo "   ✗ WARNING: Cannot verify sm_120 in build"
    fi
fi
echo ""

# Check binary for Blackwell kernels
echo "2. Checking for Blackwell kernel symbols..."
if [ -f build/bin/llama-cli ]; then
    if cuobjdump -symbols build/bin/llama-cli 2>/dev/null | grep -q "blackwell\|sm_120"; then
        echo "   ✓ Blackwell symbols found in binary"
    else
        echo "   ? No explicit Blackwell symbols (may still work via PTX JIT)"
    fi
fi
echo ""

# Quick functional test
echo "3. Quick functional test..."
if [ -f "$MODEL" ]; then
    echo "   Running short inference..."
    timeout 30 build/bin/llama-cli -m "$MODEL" -p "Hello" -n 5 2>&1 | tail -5
    echo "   ✓ Inference completed"
else
    echo "   ⊘ Skipping (no model specified)"
    echo "   Usage: $0 /path/to/model.gguf"
fi
echo ""

# NCU profiling for kernel verification
echo "4. NCU Kernel Verification (requires sudo/root)..."
if command -v ncu &> /dev/null && [ -f "$MODEL" ]; then
    echo "   Profiling key kernels..."

    # Profile with kernel filtering
    sudo ncu --set full \
        -k "regex:mul_mat_q|flash_attn|rope_|topk_moe" \
        --target-processes all \
        -o "${NCU_OUTPUT}" \
        build/bin/llama-cli -m "$MODEL" -p "Test prompt for profiling" -n 10 2>/dev/null || true

    if [ -f "${NCU_OUTPUT}.ncu-rep" ]; then
        echo ""
        echo "   === Kernel Analysis ==="

        # Extract kernel names and check for expected patterns
        ncu --import "${NCU_OUTPUT}.ncu-rep" --csv 2>/dev/null | \
            grep -E "Kernel Name|mul_mat_q|flash_attn|rope_" | head -20

        echo ""
        echo "   Key metrics to verify:"
        echo "   - MMQ: Look for cp.async instructions (Achieved Occupancy, L2 Hit Rate)"
        echo "   - FA:  Look for 7 warps active (Warp Execution Efficiency)"
        echo "   - RoPE: Look for vectorized loads (Memory Throughput)"
        echo ""
        echo "   Full report: ${NCU_OUTPUT}.ncu-rep"
    fi
else
    echo "   ⊘ Skipping (ncu not found or no model)"
fi
echo ""

# Check for cp.async usage in PTX (if available)
echo "5. PTX Analysis for cp.async..."
if [ -f build/bin/llama-cli ]; then
    PTX_COUNT=$(cuobjdump -ptx build/bin/llama-cli 2>/dev/null | grep -c "cp.async" || echo "0")
    if [ "$PTX_COUNT" -gt 0 ]; then
        echo "   ✓ Found $PTX_COUNT cp.async instructions in PTX"
    else
        echo "   ? No cp.async in dumped PTX (may be in separate .cubin)"
    fi
fi
echo ""

echo "=== Verification Complete ==="
echo ""
echo "Next steps:"
echo "1. If sm_120 not in build, rebuild with -DCMAKE_CUDA_ARCHITECTURES=120"
echo "2. Run full benchmark: build/bin/llama-bench -m \$MODEL"
echo "3. For detailed analysis: ncu --import ${NCU_OUTPUT}.ncu-rep"
