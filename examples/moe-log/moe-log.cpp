#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

struct moe_log_state {
    std::regex filter{R"(^ffn_moe_topk-(\d+)$)"};
    int n_expert = 0;
    std::vector<uint8_t> buf;
    std::unordered_map<int, std::vector<uint64_t>> counts_by_layer;
};

static void print_usage(const char * prog) {
    printf("Usage: %s -m MODEL.gguf --dataset FILE.txt [--out FILE.json] [--ctx-size N] [--threads N] [--n-gpu-layers N] [--max-tokens N]\n", prog);
    printf("Common perf flags supported: --threads-batch, --batch-size, --ubatch-size, --tensor-split, --flash-attn, --fit, --temp, --min-p\n");
    printf("Server-only flags are accepted and ignored (e.g., --host, --port, --no-host, --jinja, --parallel, --kv-unified).\n");
}

static void parse_tensor_split(const std::string & value, float * out, size_t n_max) {
    for (size_t i = 0; i < n_max; ++i) {
        out[i] = 0.0f;
    }
    std::stringstream ss(value);
    std::string item;
    size_t idx = 0;
    while (std::getline(ss, item, ',') && idx < n_max) {
        out[idx++] = std::strtof(item.c_str(), nullptr);
    }
}

static bool moe_log_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (moe_log_state *) user_data;

    if (ask) {
        return std::regex_search(t->name, st->filter);
    }

    std::cmatch match;
    if (!std::regex_search(t->name, match, st->filter)) {
        return true;
    }

    if (t->type != GGML_TYPE_I32) {
        return true;
    }

    const int layer = std::atoi(match[1].str().c_str());
    const int64_t ne0 = t->ne[0]; // k (n_expert_used)
    const int64_t ne1 = t->ne[1]; // n_tokens
    const size_t  nb0 = t->nb[0]; // index stride in bytes
    const size_t  nb1 = t->nb[1]; // token stride in bytes

    const size_t n_bytes = ggml_nbytes(t);
    st->buf.resize(n_bytes);
    ggml_backend_tensor_get(t, st->buf.data(), 0, n_bytes);

    auto & counts = st->counts_by_layer[layer];
    if (counts.empty()) {
        counts.assign(st->n_expert, 0);
    }

    for (int64_t row = 0; row < ne1; ++row) {
        for (int64_t col = 0; col < ne0; ++col) {
            int32_t id = -1;
            std::memcpy(&id, st->buf.data() + row * nb1 + col * nb0, sizeof(id));
            if (id >= 0 && id < st->n_expert) {
                counts[id] += 1;
            }
        }
    }

    return true;
}

static std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open file: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static int get_expert_count(const llama_model * model) {
    char arch[128] = {0};
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) <= 0) {
        return 0;
    }

    std::string key = std::string(arch) + ".expert_count";
    char val[64] = {0};
    if (llama_model_meta_val_str(model, key.c_str(), val, sizeof(val)) <= 0) {
        return 0;
    }

    return std::atoi(val);
}

static bool run_dataset(
    llama_context * ctx,
    const std::vector<llama_token> & tokens,
    size_t chunk_size,
    size_t max_total) {
    if (tokens.empty()) {
        return false;
    }

    size_t total = tokens.size();
    if (max_total > 0 && max_total < total) {
        total = max_total;
    }

    if (chunk_size == 0) {
        return false;
    }

    const size_t n_chunks = (total + chunk_size - 1) / chunk_size;
    for (size_t offset = 0; offset < total; offset += chunk_size) {
        const size_t n = std::min(chunk_size, total - offset);
        std::vector<llama_token> tmp(tokens.begin() + offset, tokens.begin() + offset + n);

        // Reset memory each chunk to keep positions valid and avoid n_batch assertions.
        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch batch = llama_batch_get_one(tmp.data(), (int32_t) tmp.size());
        if (llama_decode(ctx, batch) != 0) {
            return false;
        }

        const size_t chunk_idx = offset / chunk_size;
        if (chunk_idx == 0 || (chunk_idx % 100) == 0) {
            LOG_INF("%s : processed %zu/%zu chunks\n", __func__, chunk_idx + 1, n_chunks);
        }
    }

    return true;
}

static void write_json(
    std::ostream & out,
    const std::string & model_path,
    const std::string & dataset_path,
    int n_expert,
    const std::unordered_map<int, std::vector<uint64_t>> & counts_by_layer
) {
    std::vector<uint64_t> total(n_expert, 0);
    for (const auto & kv : counts_by_layer) {
        const auto & counts = kv.second;
        for (int i = 0; i < n_expert && i < (int) counts.size(); ++i) {
            total[i] += counts[i];
        }
    }

    out << "{\n";
    out << "  \"model\": \"" << model_path << "\",\n";
    out << "  \"dataset\": \"" << dataset_path << "\",\n";
    out << "  \"n_expert\": " << n_expert << ",\n";
    out << "  \"counts\": {\n";

    std::vector<int> layers;
    layers.reserve(counts_by_layer.size());
    for (const auto & kv : counts_by_layer) {
        layers.push_back(kv.first);
    }
    std::sort(layers.begin(), layers.end());

    for (size_t li = 0; li < layers.size(); ++li) {
        const int layer = layers[li];
        const auto & counts = counts_by_layer.at(layer);
        out << "    \"" << layer << "\": [";
        for (int i = 0; i < n_expert; ++i) {
            if (i < (int) counts.size()) {
                out << counts[i];
            } else {
                out << 0;
            }
            if (i + 1 < n_expert) {
                out << ", ";
            }
        }
        out << "]";
        if (li + 1 < layers.size()) {
            out << ",";
        }
        out << "\n";
    }

    out << "  },\n";
    out << "  \"total\": [";
    for (int i = 0; i < n_expert; ++i) {
        out << total[i];
        if (i + 1 < n_expert) {
            out << ", ";
        }
    }
    out << "]\n";
    out << "}\n";
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string dataset_path;
    std::string out_path;
    int max_tokens = -1;

    common_params params;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help" || arg == "--usage") {
            print_usage(argv[0]);
            return 0;
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--dataset" && i + 1 < argc) {
            dataset_path = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if ((arg == "-c" || arg == "--ctx-size") && i + 1 < argc) {
            params.n_ctx = std::atoi(argv[++i]);
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            params.cpuparams.n_threads = std::atoi(argv[++i]);
            params.cpuparams_batch.n_threads = params.cpuparams.n_threads;
        } else if (arg == "--threads-batch" && i + 1 < argc) {
            params.cpuparams_batch.n_threads = std::atoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            params.n_batch = std::atoi(argv[++i]);
        } else if ((arg == "-b" || arg == "--n-batch") && i + 1 < argc) {
            params.n_batch = std::atoi(argv[++i]);
        } else if (arg == "--ubatch-size" && i + 1 < argc) {
            params.n_ubatch = std::atoi(argv[++i]);
        } else if ((arg == "-ngl" || arg == "--n-gpu-layers") && i + 1 < argc) {
            params.n_gpu_layers = std::atoi(argv[++i]);
        } else if (arg == "--tensor-split" && i + 1 < argc) {
            parse_tensor_split(argv[++i], params.tensor_split, 128);
        } else if (arg == "--flash-attn" && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v == "on" || v == "true" || v == "1") {
                params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            } else if (v == "off" || v == "false" || v == "0") {
                params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            } else {
                params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
            }
        } else if (arg == "--fit" && i + 1 < argc) {
            const std::string v = argv[++i];
            params.fit_params = (v == "on" || v == "true" || v == "1");
        } else if (arg == "--temp" && i + 1 < argc) {
            params.sampling.temp = std::strtof(argv[++i], nullptr);
        } else if (arg == "--min-p" && i + 1 < argc) {
            params.sampling.min_p = std::strtof(argv[++i], nullptr);
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
        } else if (arg == "--no-host" || arg == "--jinja" || arg == "--kv-unified" || arg == "--no-warmup") {
            // server-only flags (accepted but ignored)
        } else if ((arg == "--host" || arg == "--port" || arg == "--parallel") && i + 1 < argc) {
            // server-only flags with value (accepted but ignored)
            ++i;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty() || dataset_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    moe_log_state state;

    params.model.path = model_path;
    params.cb_eval = moe_log_cb_eval;
    params.cb_eval_user_data = &state;
    params.warmup = false;

    common_init();

    ggml_backend_load_all();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init model\n", __func__);
        return 1;
    }

    const int n_expert = get_expert_count(model);
    if (n_expert <= 0) {
        LOG_ERR("%s : model does not report expert_count metadata\n", __func__);
        return 1;
    }
    state.n_expert = n_expert;

    const bool add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(model));
    const std::string text = read_file(dataset_path);
    std::vector<llama_token> tokens = common_tokenize(ctx, text, add_bos, true);

    const int n_ctx = llama_n_ctx(ctx);

    if (tokens.empty()) {
        LOG_ERR("%s : dataset produced no tokens\n", __func__);
        return 1;
    }

    size_t chunk_size = (size_t) params.n_batch;
    if (chunk_size == 0) {
        chunk_size = (size_t) n_ctx;
    }
    if ((size_t) n_ctx < chunk_size) {
        chunk_size = (size_t) n_ctx;
    }
    if (max_tokens > 0 && (size_t) max_tokens < chunk_size) {
        chunk_size = (size_t) max_tokens;
    }

    LOG_INF("%s : dataset tokens total = %zu, chunk_size = %zu\n", __func__, tokens.size(), chunk_size);

    if (!run_dataset(ctx, tokens, chunk_size, max_tokens > 0 ? (size_t) max_tokens : 0)) {
        LOG_ERR("%s : failed to evaluate dataset\n", __func__);
        return 1;
    }

    if (out_path.empty()) {
        write_json(std::cout, model_path, dataset_path, n_expert, state.counts_by_layer);
    } else {
        std::ofstream out(out_path);
        if (!out) {
            LOG_ERR("%s : failed to open output file %s\n", __func__, out_path.c_str());
            return 1;
        }
        write_json(out, model_path, dataset_path, n_expert, state.counts_by_layer);
    }

    llama_backend_free();

    return 0;
}
