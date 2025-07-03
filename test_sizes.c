#include "ggml/include/ggml.h"
#include <stdio.h>

int main() {
    printf("GGML_OBJECT_SIZE = %zu bytes\n", sizeof(struct ggml_object));
    printf("GGML_TENSOR_SIZE = %zu bytes\n", sizeof(struct ggml_tensor));
    printf("ggml_tensor_overhead() = %zu bytes\n", ggml_tensor_overhead());
    
    printf("\nStruct components:\n");
    printf("sizeof(enum ggml_type) = %zu\n", sizeof(enum ggml_type));
    printf("sizeof(struct ggml_backend_buffer *) = %zu\n", sizeof(struct ggml_backend_buffer *));
    printf("sizeof(int64_t[4]) = %zu\n", sizeof(int64_t[4]));
    printf("sizeof(size_t[4]) = %zu\n", sizeof(size_t[4]));
    printf("sizeof(enum ggml_op) = %zu\n", sizeof(enum ggml_op));
    printf("sizeof(int32_t[16]) = %zu\n", sizeof(int32_t[16])); // GGML_MAX_OP_PARAMS / 4
    printf("sizeof(int32_t) = %zu\n", sizeof(int32_t));
    printf("sizeof(struct ggml_tensor *[10]) = %zu\n", sizeof(struct ggml_tensor *[10]));
    printf("sizeof(struct ggml_tensor *) = %zu\n", sizeof(struct ggml_tensor *));
    printf("sizeof(size_t) = %zu\n", sizeof(size_t));
    printf("sizeof(void *) = %zu\n", sizeof(void *));
    printf("sizeof(char[64]) = %zu\n", sizeof(char[64]));
    printf("sizeof(void *) = %zu\n", sizeof(void *));
    printf("sizeof(char[8]) = %zu\n", sizeof(char[8]));
    
    return 0;
}
