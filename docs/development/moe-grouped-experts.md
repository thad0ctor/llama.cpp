# MoE Grouped Expert Quantization

## Overview

This feature enables per-group quantization of MoE (Mixture of Experts) expert
tensors. Instead of a single merged tensor per projection per layer (where all
experts share the same quantization type), experts are split into groups that
can be independently quantized (e.g., hot/warm/cold tiers).

Primary target architecture: **MiniMax M2** (`minimax-m2`).

## Metadata Format

### GGUF Keys

| Key | Type | Description |
|-----|------|-------------|
| `{arch}.moe_quant_group_count` | uint32 | Number of expert groups (e.g., 3) |
| `{arch}.moe_quant_expert_group_map.{bid}` | uint32[] | Per-layer mapping: expert_id → group_id |
| `{arch}.moe_quant_expert_index_map.{bid}` | uint32[] | Per-layer mapping: expert_id → index within group |

### Tensor Naming

Grouped expert tensors follow this naming convention:

```
blk.{layer}.ffn_gate_exps.g{group_id}.weight
blk.{layer}.ffn_down_exps.g{group_id}.weight
blk.{layer}.ffn_up_exps.g{group_id}.weight
```

Each group tensor is a 3D tensor with shape `[dim_in, dim_out, n_experts_in_group]`.

### Example: 12 experts, 3 groups (round-robin)

- Group 0 ("hot"): experts 0, 3, 6, 9 → `blk.0.ffn_gate_exps.g0.weight` (shape [n_embd, n_ff, 4])
- Group 1 ("warm"): experts 1, 4, 7, 10 → `blk.0.ffn_gate_exps.g1.weight` (shape [n_embd, n_ff, 4])
- Group 2 ("cold"): experts 2, 5, 8, 11 → `blk.0.ffn_gate_exps.g2.weight` (shape [n_embd, n_ff, 4])

## Routing Algorithm

At inference time, the grouped MoE FFN works as follows:

1. **Gating**: Standard top-k expert selection (softmax/sigmoid + top-k) produces global expert IDs and weights.
2. **Remap**: Static lookup tables (from metadata) map each selected expert to its group tensor and local index within that group.
3. **Per-group FFN**: For each group, `ggml_mul_mat_id` is called on the group tensor. Experts not belonging to the current group are masked (weight zeroed).
4. **Accumulate**: Results from all groups are summed.

This is slightly slower than the fully-merged path due to the per-group iteration
(n_groups separate matmul calls instead of 1), but preserves per-group quantization
in memory. Expected overhead: ~15-30% vs merged, but significantly better quality
control than uniform quantization.

## Backward Compatibility

- Existing GGUFs (merged format): Fully supported, no changes needed.
- New grouped GGUFs: Require runtime support. If `moe_quant_group_count` is present
  but the runtime lacks grouped MoE support, tensor loading will fail with a clear
  error (expected tensors not found).
- Default converter behavior: Unchanged. Grouped export requires `--moe-grouped-experts`.

## Conversion Examples

### Basic grouped conversion (3 groups, round-robin):

```bash
python convert_hf_to_gguf.py /path/to/minimax-m2 \
    --outtype f16 \
    --moe-grouped-experts \
    --moe-group-count 3
```

### Custom group assignment:

```bash
# Create a group map JSON file
cat > group_map.json << 'EOF'
{
  "0": {"0": 0, "1": 0, "2": 0, "3": 1, "4": 1, "5": 2},
  "1": {"0": 0, "1": 0, "2": 0, "3": 1, "4": 1, "5": 2}
}
EOF

python convert_hf_to_gguf.py /path/to/minimax-m2 \
    --outtype f16 \
    --moe-grouped-experts \
    --moe-group-count 3 \
    --moe-group-map group_map.json
```

### Per-group quantization with tensor_types.txt:

```bash
# Create tensor type overrides targeting groups
cat > tensor_types.txt << 'EOF'
# Hot experts (group 0) - high quality
\.g0\.weight q6_k

# Warm experts (group 1) - medium quality
\.g1\.weight q4_k_m

# Cold experts (group 2) - aggressive compression
\.g2\.weight q2_k
EOF

# Quantize with per-group types
./llama-quantize \
    minimax-m2-f16.gguf \
    minimax-m2-grouped.gguf \
    q4_k_m \
    --tensor-type-file tensor_types.txt
```

### Equivalence verification:

To verify correctness, convert with grouped experts where all groups use the
same quantization. The output should match the merged baseline within floating
point tolerance:

```bash
# Grouped (all same quant)
python convert_hf_to_gguf.py model --outtype f16 --moe-grouped-experts --moe-group-count 3
./llama-quantize grouped-f16.gguf grouped-q4.gguf q4_k_m

# Merged baseline
python convert_hf_to_gguf.py model --outtype f16
./llama-quantize merged-f16.gguf merged-q4.gguf q4_k_m

# Compare outputs (should be identical within tolerance)
./llama-cli -m grouped-q4.gguf -p "test prompt" --seed 42
./llama-cli -m merged-q4.gguf -p "test prompt" --seed 42
```

## Tradeoffs

| Aspect | Merged | Grouped |
|--------|--------|---------|
| Memory | Single quant for all experts | Per-group quant control |
| Speed | Fastest (single mul_mat_id call) | ~15-30% slower (per-group calls) |
| Quality | Uniform precision loss | Hot experts preserved, cold experts compressed |
| File size | Limited by worst-case expert | Smaller: aggressive on cold, generous on hot |
| Compatibility | Universal | Requires grouped MoE runtime support |

## File Changes

- `gguf-py/gguf/constants.py`: New keys, tensor types, tensor names
- `gguf-py/gguf/gguf_writer.py`: Writer methods for grouped metadata
- `convert_hf_to_gguf.py`: CLI flags, MiniMaxM2Model grouped export
- `src/llama-arch.h`: New KV keys, tensor enum entries
- `src/llama-arch.cpp`: Tensor/KV name mappings
- `src/llama-hparams.h`: Remap table storage
- `src/llama-model.h`: Grouped tensor pointers in layer struct
- `src/llama-model.cpp`: Loader for grouped tensors + metadata
- `src/llama-graph.h/cpp`: `build_grouped_moe_ffn` implementation
- `src/models/minimax-m2.cpp`: Runtime routing dispatch
