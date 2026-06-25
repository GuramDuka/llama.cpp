#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

struct test_ctx {
    common_params       params;
    llama_model *       model   = nullptr;
    const llama_vocab * vocab   = nullptr;
    int                 n_vocab = 0;
};

static llama_context * make_ctx(llama_model * model, uint32_t n_rs_seq) {
    auto cparams      = llama_context_default_params();
    cparams.n_ctx     = 256;
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = n_rs_seq;
    cparams.n_batch   = std::max(cparams.n_batch, std::max(n_rs_seq + 1, (uint32_t) 64));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, std::max(n_rs_seq + 1, (uint32_t) 64));
    return llama_init_from_model(model, cparams);
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens) {
    llama_batch batch = llama_batch_init((int) tokens.size(), 0, 1);
    for (int pos = 0; pos < (int) tokens.size(); ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, true);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool compare_logits_eps(const float * a, const float * b, int n, float eps, const char * label, int vocab_size) {
    for (int i = 0; i < n; ++i) {
        if (std::fabs(a[i] - b[i]) > eps) {
            fprintf(stderr, "%s : logits mismatch at token %d (%g != %g)\n", label, i, (double) a[i], (double) b[i]);
            return false;
        }
    }
    return true;
}

// Test 1: Decode → seq_rm → save checkpoint → restore → decode → compare
// This tests the in-memory checkpoint path.
static bool test_checkpoint_rollback(test_ctx & tctx, uint32_t n_rs_seq_override) {
    fprintf(stderr, "\n--- Test 1: checkpoint rollback (n_rs_seq=%u) ---\n", n_rs_seq_override);

    llama_context * ctx_src = make_ctx(tctx.model, n_rs_seq_override);
    llama_context * ctx_dst = make_ctx(tctx.model, n_rs_seq_override);
    if (ctx_src == nullptr || ctx_dst == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return false;
    }

    if (llama_n_rs_seq(ctx_src) == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return true;
    }

    const uint32_t           n_rs_seq = llama_n_rs_seq(ctx_src);
    std::vector<llama_token> tokens   = common_tokenize(ctx_src, "The quick brown fox jumps", true);
    if (tokens.size() > n_rs_seq + 1) {
        tokens.resize(n_rs_seq + 1);
    }
    if (tokens.size() < 2) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return false;
    }
    const uint32_t    n_tokens = tokens.size();
    const llama_token last_tok = tokens.back();
    const llama_pos   last_pos = (llama_pos) n_tokens - 2;

    // Decode the full prompt, then roll back the last position.
    if (!decode_tokens(ctx_src, tokens)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return false;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, last_pos, -1)) {
        fprintf(stderr, "%s : rollback failed\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return false;
    }

    // Save and restore via checkpoint
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);
    ckpt.load_tgt(ctx_dst, 0, 0);

    // Replay on both and compare
    if (!decode_one(ctx_src, last_tok, last_pos) || !decode_one(ctx_dst, last_tok, last_pos)) {
        fprintf(stderr, "%s : replay failed\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return false;
    }

    const float * logits_src = llama_get_logits_ith(ctx_src, 0);
    const float * logits_dst = llama_get_logits_ith(ctx_dst, 0);
    if (logits_src == nullptr || logits_dst == nullptr) {
        fprintf(stderr, "%s : missing logits\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return false;
    }

    constexpr float eps = 1e-5f;
    if (!compare_logits_eps(logits_src, logits_dst, tctx.n_vocab, eps, __func__, tctx.n_vocab)) {
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return false;
    }

    // Dirty context test: load checkpoint into a context that already has
    // its own rollback state with a DIFFERENT prompt.
    llama_context * ctx_dirty = make_ctx(tctx.model, n_rs_seq_override);
    if (ctx_dirty == nullptr) {
        fprintf(stderr, "%s : failed to init dirty ctx\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return false;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % tctx.n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty, noise)) {
        fprintf(stderr, "%s : dirty prompt decode failed\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        llama_free(ctx_dirty);
        return false;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, last_pos, -1)) {
        fprintf(stderr, "%s : dirty rollback failed\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        llama_free(ctx_dirty);
        return false;
    }

    ckpt.load_tgt(ctx_dirty, 0, 0);

    if (!decode_one(ctx_dirty, last_tok, last_pos)) {
        fprintf(stderr, "%s : dirty replay failed\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        llama_free(ctx_dirty);
        return false;
    }

    const float * logits_dirty = llama_get_logits_ith(ctx_dirty, 0);
    if (logits_dirty == nullptr) {
        fprintf(stderr, "%s : missing dirty logits\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        llama_free(ctx_dirty);
        return false;
    }

    if (!compare_logits_eps(logits_src, logits_dirty, tctx.n_vocab, eps, __func__, tctx.n_vocab)) {
        llama_free(ctx_src);
        llama_free(ctx_dst);
        llama_free(ctx_dirty);
        return false;
    }

    fprintf(stderr, "%s : recurrent rollback checkpoint restored successfully\n", __func__);
    llama_free(ctx_src);
    llama_free(ctx_dst);
    llama_free(ctx_dirty);
    return true;
}

// Test 2 (MTP/L3 scenario): Decode → save state → restore state → seq_rm → decode → compare
// This simulates what happens when:
//   1. Server saves KV cache to disk (L3)
//   2. Server restores from disk
//   3. Server calls seq_rm at the last cached position (TAG_PROMPT_LOGITS)
//   4. Server decodes the last token
// With MTP, n_rs_seq is set to draft.n_max (e.g., 3) via need_n_rs_seq().
static bool test_restore_seq_rm(test_ctx & tctx, uint32_t n_rs_seq_override) {
    fprintf(stderr, "\n--- Test 2: MTP/L3 restore → seq_rm → decode (n_rs_seq=%u) ---\n", n_rs_seq_override);

    llama_context * ctx_src = make_ctx(tctx.model, n_rs_seq_override);
    if (ctx_src == nullptr) {
        fprintf(stderr, "%s : failed to init src ctx\n", __func__);
        return false;
    }

    if (llama_n_rs_seq(ctx_src) == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        llama_free(ctx_src);
        return true;
    }

    const uint32_t           n_rs_seq = llama_n_rs_seq(ctx_src);
    std::vector<llama_token> tokens   = common_tokenize(ctx_src, "The quick brown fox jumps", true);
    if (tokens.size() > n_rs_seq + 1) {
        tokens.resize(n_rs_seq + 1);
    }
    if (tokens.size() < 2) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        llama_free(ctx_src);
        return false;
    }
    const uint32_t    n_tokens = tokens.size();
    const llama_token last_tok = tokens.back();
    const llama_pos   last_pos = (llama_pos) n_tokens - 2;

    // Decode the full prompt on ctx_src
    if (!decode_tokens(ctx_src, tokens)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        llama_free(ctx_src);
        return false;
    }

    // Save full state to an in-memory buffer (simulating L3 save)
    const size_t state_size = llama_state_seq_get_size_ext(ctx_src, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (state_size == 0) {
        fprintf(stderr, "%s : state is empty\n", __func__);
        llama_free(ctx_src);
        return false;
    }
    std::vector<uint8_t> state_buf(state_size);
    const size_t         saved =
        llama_state_seq_get_data_ext(ctx_src, state_buf.data(), state_size, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (saved != state_size) {
        fprintf(stderr, "%s : save size mismatch (%zu != %zu)\n", __func__, saved, state_size);
        llama_free(ctx_src);
        return false;
    }

    // On ctx_src: call seq_rm at the last position (like server does for TAG_PROMPT_LOGITS)
    // This triggers rollback. ctx_src's rollback planes were populated during the
    // initial decode, so this produces correct logits.
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, last_pos, -1)) {
        fprintf(stderr, "%s : src rollback failed\n", __func__);
        llama_free(ctx_src);
        return false;
    }
    if (!decode_one(ctx_src, last_tok, last_pos)) {
        fprintf(stderr, "%s : src replay failed\n", __func__);
        llama_free(ctx_src);
        return false;
    }
    const float * logits_ref = llama_get_logits_ith(ctx_src, 0);
    if (logits_ref == nullptr) {
        fprintf(stderr, "%s : missing src logits\n", __func__);
        llama_free(ctx_src);
        return false;
    }

    // Now create ctx_dst and restore the state from the buffer (simulating L3 restore)
    llama_context * ctx_dst = make_ctx(tctx.model, n_rs_seq_override);
    if (ctx_dst == nullptr) {
        fprintf(stderr, "%s : failed to init dst ctx\n", __func__);
        llama_free(ctx_src);
        return false;
    }

    const size_t loaded =
        llama_state_seq_set_data_ext(ctx_dst, state_buf.data(), state_size, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (loaded != state_size) {
        fprintf(stderr, "%s : load size mismatch (%zu != %zu)\n", __func__, loaded, state_size);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return false;
    }

    // Same seq_rm as on ctx_src — after L3 restore the rollback planes are
    // zero-initialized. With our fix, they are copies of the base plane,
    // so the rollback → decode produces the same logits as ctx_src.
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, last_pos, -1)) {
        fprintf(stderr, "%s : dst rollback failed\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return false;
    }
    if (!decode_one(ctx_dst, last_tok, last_pos)) {
        fprintf(stderr, "%s : dst replay failed\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return false;
    }
    const float * logits_dst = llama_get_logits_ith(ctx_dst, 0);
    if (logits_dst == nullptr) {
        fprintf(stderr, "%s : missing dst logits\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return false;
    }

    constexpr float eps = 1e-5f;
    if (!compare_logits_eps(logits_ref, logits_dst, tctx.n_vocab, eps, __func__, tctx.n_vocab)) {
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return false;
    }

    fprintf(stderr, "%s : restore → seq_rm → decode matches reference\n", __func__);
    llama_free(ctx_src);
    llama_free(ctx_dst);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict     = 1;

    // Extract --n-rs-seq before common_params_parse (which rejects unknown args)
    uint32_t n_rs_seq_test = 8;
    {
        int write_idx = 1;
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "--n-rs-seq") == 0 && i + 1 < argc) {
                n_rs_seq_test = (uint32_t) std::max(0, atoi(argv[i + 1]));
                i++;  // skip the value
                continue;
            }
            argv[write_idx++] = argv[i];
        }
        argc       = write_idx;
        argv[argc] = nullptr;
    }

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model *          model      = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    test_ctx tctx = { params, model, vocab, n_vocab };

    bool all_pass = true;

    all_pass = test_checkpoint_rollback(tctx, n_rs_seq_test) && all_pass;
    all_pass = test_restore_seq_rm(tctx, n_rs_seq_test) && all_pass;

    if (all_pass) {
        fprintf(stderr, "\n%s : ALL TESTS PASSED\n", __func__);
        return 0;
    } else {
        fprintf(stderr, "\n%s : SOME TESTS FAILED\n", __func__);
        return 1;
    }
}
