#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#ifdef LLAMA_HAS_ZSTD

#    define ZDICT_STATIC_LINKING_ONLY
#    include <zdict.h>

// KV cache dictionary trainer/manager.
// Trains a zstd dictionary from saved KV cache data using ZDICT_fastCover.
class llama_kv_dict {
  public:
    llama_kv_dict()  = default;
    ~llama_kv_dict() = default;

    // Add a sample (raw KV cache data) for dictionary training.
    // Splits the data into 16 KB chunks so ZDICT fastCover (which needs >=10 samples) works.
    void add_sample(const uint8_t * data, size_t size);

    // Train the dictionary from accumulated samples.
    // Returns true if training succeeded.
    bool train(int level = 3);

    bool is_trained() const { return trained; }

    const void * data() const { return dict.data(); }

    size_t size() const { return dict.size(); }

  private:
    std::vector<std::vector<uint8_t>> samples;
    std::vector<uint8_t>              dict;
    bool                              trained = false;
};

// Factory function — defined after the complete class definition in llama-kv-dict.cpp
std::unique_ptr<llama_kv_dict> create_kv_dict();

#endif  // LLAMA_HAS_ZSTD
