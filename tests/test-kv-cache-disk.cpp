// Test for KV cache disk save/restore via llama_state_seq_save_file / llama_state_seq_load_file
// Requires a real GGUF model (fixture: test-download-model)

#include "arg.h"
#include "common.h"
#include "get-model.h"
#include "llama-cpp.h"
#include "log.h"

#include <unistd.h>

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

static const char * test_tag = "kv-cache-disk";
static int          g_failed = 0;

static void assert_true(int val, const char * msg) {
    if (!val) {
        fprintf(stderr, "  FAIL: %s\n", msg);
        g_failed++;
    }
}

// Read file header (magic 4 + version 4 + n_token_count 4 = 12 bytes)
static bool read_header(const char * path, uint32_t * magic, uint32_t * version, uint32_t * n_token_count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    uint8_t hdr[12];
    f.read(reinterpret_cast<char *>(hdr), 12);
    if (f.gcount() != 12) {
        return false;
    }
    *magic         = *(uint32_t *) (hdr + 0);
    *version       = *(uint32_t *) (hdr + 4);
    *n_token_count = *(uint32_t *) (hdr + 8);
    return true;
}

static bool test_lcp_computation() {
    fprintf(stderr, "\n=== test_lcp_computation ===\n");

    auto lcp = [](const std::vector<int32_t> & a, const std::vector<int32_t> & b) -> float {
        if (a.empty() || b.empty()) {
            return 0.0f;
        }
        size_t common = 0;
        size_t m      = (a.size() < b.size()) ? a.size() : b.size();
        for (size_t i = 0; i < m; ++i) {
            if (a[i] == b[i]) {
                common++;
            } else {
                break;
            }
        }
        return (float) common / (float) b.size();
    };

    std::vector<int32_t> a = { 1, 2, 3, 4, 5 };
    std::vector<int32_t> b = { 1, 2, 3, 4, 5 };
    std::vector<int32_t> c = { 1, 2, 3, 100, 200 };
    std::vector<int32_t> d = { 100, 200, 300, 400, 500 };

    assert_true(lcp(a, b) >= 0.99f, "identical LCP should be ~1.0");
    assert_true(lcp(a, c) >= 0.58f, "partial LCP should be ~0.6");
    assert_true(lcp(a, d) == 0.0f, "no-match LCP should be 0.0");

    return (g_failed == 0);
}

static bool test_save_restore(llama_context * ctx, const std::vector<int32_t> & tokens, const char * cache_dir) {
    fprintf(stderr, "\n=== test_save_restore ===\n");

    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/test.bin", cache_dir);

    size_t written = llama_state_seq_save_file(ctx, filepath, 0, tokens.data(), (size_t) tokens.size());
    assert_true(written > 0, "save should produce data");

    // Verify header
    uint32_t magic = 0, version = 0, n_tok = 0;
    assert_true(read_header(filepath, &magic, &version, &n_tok), "header readable");
    assert_true(magic == LLAMA_STATE_SEQ_MAGIC, "magic correct");
    assert_true(n_tok == (uint32_t) tokens.size(), "token count matches");
    fprintf(stderr, "  header: magic=0x%08x, ver=%u, n_tok=%u\n", magic, version, n_tok);

    // Restore
    std::vector<int32_t> buf(tokens.size());
    size_t               n_out = 0;
    size_t loaded = llama_state_seq_load_file(ctx, filepath, -1, buf.data(), (size_t) tokens.size(), &n_out);
    assert_true(loaded > 0, "restore should produce data");
    fprintf(stderr, "  restored %zu bytes\n", loaded);

    return (g_failed == 0);
}

static bool test_restart_restore(llama_model * model, const std::vector<int32_t> & tokens, const char * cache_dir) {
    fprintf(stderr, "\n=== test_restart_restore ===\n");

    // Save with context A
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = 256;
    auto ctx_a              = llama_context_ptr{ llama_init_from_model(model, cp) };

    // Decode prompt
    llama_batch batch = llama_batch_init((int) tokens.size(), 0, 1);
    for (int i = 0; i < (int) tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], i, { 0 }, true);
    }
    assert_true(llama_decode(ctx_a.get(), batch) == 0, "decode should succeed");
    llama_batch_free(batch);

    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/restart.bin", cache_dir);

    size_t written = llama_state_seq_save_file(ctx_a.get(), filepath, 0, tokens.data(), (size_t) tokens.size());
    assert_true(written > 0, "save should produce data");

    // Simulate restart: new context B
    auto ctx_b = llama_context_ptr{ llama_init_from_model(model, cp) };

    std::vector<int32_t> buf(tokens.size());
    size_t               n_out = 0;
    size_t loaded = llama_state_seq_load_file(ctx_b.get(), filepath, -1, buf.data(), (size_t) tokens.size(), &n_out);
    assert_true(loaded > 0, "restore to new context should succeed");
    fprintf(stderr, "  restart restore OK: %zu bytes\n", loaded);

    return (g_failed == 0);
}

int main(int argc, char ** argv) {
    common_params params;
    params.sampling.seed = 1234;
    params.n_ctx         = 512;
    params.n_predict     = 0;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    const char * model_path = get_model_or_exit(argc, argv);
    if (!model_path) {
        return 1;
    }

    auto llama_init = common_init_from_params(params);
    if (!llama_init || !llama_init->model() || !llama_init->context()) {
        fprintf(stderr, "%s: failed to init model\n", test_tag);
        return 1;
    }

    llama_context *      ctx    = llama_init->context();
    llama_model *        model  = llama_init->model();
    const llama_vocab *  vocab  = llama_model_get_vocab(model);
    // Encode prompt
    std::string          prompt = "What is 2+2? Briefly.";
    std::vector<int32_t> prompt_tokens(prompt.size() + 8, 0);
    int32_t              n_tok = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), prompt_tokens.data(),
                                                (int32_t) prompt_tokens.size(), true, true);
    if (n_tok < 0) {
        fprintf(stderr, "%s: tokenize failed\n", test_tag);
        return 1;
    }
    prompt_tokens.resize(n_tok);

    // Decode prompt so KV cache is populated
    llama_batch batch = llama_batch_init(n_tok, 0, 1);
    for (int i = 0; i < n_tok; ++i) {
        common_batch_add(batch, prompt_tokens[i], i, { 0 }, true);
    }
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s: decode failed\n", test_tag);
        llama_batch_free(batch);
        return 1;
    }
    llama_batch_free(batch);

    fprintf(stderr, "Prompt: %d tokens\n", n_tok);

    // Create temp cache dir
    char cache_dir[256];
    snprintf(cache_dir, sizeof(cache_dir), "/tmp/kv-cache-test-cpp-%d", getpid());
    std::filesystem::create_directories(cache_dir);

    int all_ok = 0;

    // Run each test
    if (test_lcp_computation()) {
        all_ok++;
    }
    if (test_save_restore(ctx, prompt_tokens, cache_dir)) {
        all_ok++;
    }
    if (test_restart_restore(model, prompt_tokens, cache_dir)) {
        all_ok++;
    }

    // Cleanup
    std::filesystem::remove_all(cache_dir);

    fprintf(stderr, "\n=== RESULTS: %d passed, %d failed ===\n", all_ok, 3 - all_ok);
    return (3 - all_ok) > 0 ? 1 : 0;
}
