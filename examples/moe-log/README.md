# llama.cpp/examples/moe-log

Collects MoE expert activation counts by logging the `ffn_moe_topk-*` tensors during eval.

## Usage

```bash
llama-moe-log -m /path/to/model.gguf --dataset /path/to/text.txt --out experts.json
```

Options:
- `--ctx-size N`
- `--threads N`
- `--n-gpu-layers N`
- `--max-tokens N`

Output JSON includes per-layer expert counts and totals.
