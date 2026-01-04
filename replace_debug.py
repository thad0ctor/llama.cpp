import os
import re

files = [
    "ggml/src/ggml-cuda/binbcast.cu",
    "ggml/src/ggml-cuda/mmq.cu",
    "ggml/src/ggml-cuda/fattn-mma-f16.cuh",
    "ggml/src/ggml-cuda/fattn-common.cuh",
    "ggml/src/ggml-cuda/mmq.cuh",
    "ggml/src/ggml-cuda/softmax.cu",
    "ggml/src/ggml-cuda/ggml-cuda.cu",
    "ggml/src/ggml-cuda/quantize.cu",
    "ggml/src/ggml-cuda/norm.cu",
    "ggml/src/ggml-cuda/topk-moe.cu",
    "ggml/src/ggml-cuda/mma.cuh",
]

def process_file(filepath):
    if not os.path.exists(filepath):
        print(f"Skipping {filepath} (not found)")
        return

    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern 1: fprintf(stderr, "[\n]..."
    # We match fprintf(stderr, followed by whitespace, then a string literal starting with [ or \n[
    # Regex: fprintf\(stderr,\s*"(?:\\n)?\[
    # Replacement: DEBUG_PRINT("...
    
    # Using a function for replacement to handle the exact string start
    def repl_fprintf(match):
        return f'DEBUG_PRINT("{match.group(1)}'

    content = re.sub(r'fprintf\(stderr,\s*"((\\n)?\[)\'', repl_fprintf, content)

    # Pattern 2: printf("[\n]..."
    def repl_printf(match):
        return f'DEBUG_PRINT("{match.group(1)}'

    content = re.sub(r'printf\("((?:\\n)?\[)\'', repl_printf, content)

    # Special cases
    content = content.replace('printf("cuda pool', 'DEBUG_PRINT("cuda pool')
    content = content.replace('printf("topk_opt', 'DEBUG_PRINT("topk_opt')

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Processed {filepath}")

for f in files:
    process_file(f)
