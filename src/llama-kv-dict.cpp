#include "llama-kv-dict.h"

#include "llama-impl.h"  // for LLAMA_LOG_INFO, LLAMA_LOG_WARN

#include <cstring>

#ifdef LLAMA_HAS_ZSTD

void llama_kv_dict::add_sample(const uint8_t * data, size_t size) {
    const size_t chunk_size = 16 * 1024;  // 16 KB per chunk == one "sample" for ZDICT
    size_t       offset     = 0;
    while (offset < size) {
        size_t sz = std::min(chunk_size, size - offset);
        samples.emplace_back(data + offset, data + offset + sz);
        offset += sz;
    }
}

bool llama_kv_dict::train(int level) {
    if (samples.empty()) {
        return false;
    }

    // Prepare sample buffer: concatenate all samples
    std::vector<uint8_t> all_data;
    std::vector<size_t>  sample_sizes;
    for (const auto & s : samples) {
        all_data.insert(all_data.end(), s.begin(), s.end());
        sample_sizes.push_back(s.size());
    }

    // Target dictionary size: ~5% of total training data, capped at 64 KB.
    // ZDICT fastCover needs >=10 samples, so add_sample() splits data into 16KB chunks.
    const size_t target_dict_size = std::min((size_t) (64 * 1024), std::max((size_t) (4 * 1024), all_data.size() / 20));
    dict.resize(target_dict_size);

    ZDICT_fastCover_params_t params;
    memset(&params, 0, sizeof(params));
    params.zParams.compressionLevel  = level;
    params.zParams.notificationLevel = 0;  // no stderr output
    params.k                         = 128;
    params.d                         = 6;
    params.f                         = 15;
    params.steps                     = 1;  // single pass for speed
    params.splitPoint                = 0.0;
    params.accel                     = 2;  // skip every other dmer for 2x speed

    size_t dict_size = ZDICT_trainFromBuffer_fastCover(dict.data(), dict.size(), all_data.data(), sample_sizes.data(),
                                                       (unsigned) sample_sizes.size(), params);

    if (ZDICT_isError(dict_size)) {
        dict.clear();
        LLAMA_LOG_WARN("%s: dictionary training failed: %s\n", __func__, ZDICT_getErrorName(dict_size));
        return false;
    }

    dict.resize(dict_size);
    trained = true;

    LLAMA_LOG_INFO("%s: dictionary trained, size=%zu bytes from %zu samples (total %zu bytes)\n", __func__, dict_size,
                   samples.size(), all_data.size());

    // Reduce memory: keep only the dictionary, not the samples
    samples.clear();
    samples.shrink_to_fit();

    return true;
}

std::unique_ptr<llama_kv_dict> create_kv_dict() {
    return std::unique_ptr<llama_kv_dict>(new llama_kv_dict());
}

#endif  // LLAMA_HAS_ZSTD
