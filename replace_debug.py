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
    # Replacement: DEBUG_PRINT("..."
    # We want to replace 'fprintf(stderr, "' with 'DEBUG_PRINT("'
    # ONLY IF the string starts with optional \n then [
    
    # Regex breakdown:
    # fprintf\(stderr,\s*"  : matches fprintf(stderr, "
    # (                     : start capture group 1
    #   (?:\\n)?            : optional literal \n
    #   \[                  : literal [
    # )                     : end capture group 1
    
    def repl_fprintf(match):
        return f'DEBUG_PRINT("{match.group(1)}'

    content = re.sub(r'fprintf\(stderr,\s*"((\\n)?\[)', repl_fprintf, content)

    # Pattern 2: printf("[\n]..."
    def repl_printf(match):
        return f'DEBUG_PRINT("{match.group(1)}'

    content = re.sub(r'printf\("((?:\\n)?\[)', repl_printf, content)

    # Special cases
    content = content.replace('printf("cuda pool', 'DEBUG_PRINT("cuda pool')
    content = content.replace('printf("topk_opt', 'DEBUG_PRINT("topk_opt')

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Processed {filepath}")

for f in files:
    process_file(f)