#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#ifdef LLAMA_HAS_ZSTD

#    include <zstd.h>

// Need base IO interfaces, llama_file, and other upstream types
#    include "llama-compress-stats.h"
#    include "llama-io.h"    // llama_io_write_i, llama_io_read_i
#    include "llama-mmap.h"  // llama_file

// =============================================================================
// Buffer-backed IO writer
// =============================================================================
class llama_io_write_buf : public llama_io_write_i {
  public:
    llama_io_write_buf() = default;

    void   write(const void * src, size_t size) override;
    void   write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override;
    size_t n_bytes() override;

    const uint8_t * data() const { return buf.data(); }

    size_t size() const { return buf.size(); }

  private:
    std::vector<uint8_t> buf;
    std::vector<uint8_t> temp_buffer;
};

// =============================================================================
// zstd streaming compressor (llama_io_write_i wrapper)
// =============================================================================
class llama_io_write_zstd : public llama_io_write_i {
  public:
    llama_io_write_zstd(llama_file * f, int level, const void * dict = nullptr, size_t dict_size = 0);
    ~llama_io_write_zstd() override;

    void   write(const void * src, size_t size) override;
    void   write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override;
    size_t n_bytes() override;

  private:
    void flush_stream(ZSTD_EndDirective end);

    int64_t              t_start_us         = 0;
    uint64_t             uncompressed_bytes = 0;
    llama_file *         file               = nullptr;
    ZSTD_CCtx *          cctx               = nullptr;
    ZSTD_CDict *         cdidt              = nullptr;
    std::vector<uint8_t> in_buf;
    std::vector<uint8_t> out_buf;
    std::vector<uint8_t> temp_buffer;
    size_t               size_written = 0;
};

// =============================================================================
// zstd streaming decompressor (llama_io_read_i wrapper)
// =============================================================================
class llama_io_read_zstd : public llama_io_read_i {
  public:
    llama_io_read_zstd(llama_file * f, const void * dict = nullptr, size_t dict_size = 0);
    ~llama_io_read_zstd() override;

    void   read(void * dst, size_t size) override;
    void   read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override;
    size_t n_bytes() override;

  private:
    void decompress_chunk();

    int64_t              t_start_us = 0;
    llama_file *         file       = nullptr;
    ZSTD_DCtx *          dctx       = nullptr;
    ZSTD_DDict *         ddct       = nullptr;
    std::vector<uint8_t> compressed_data;
    const uint8_t *      in_data = nullptr;
    size_t               in_size = 0;
    size_t               in_pos  = 0;
    std::vector<uint8_t> out_buf;
    size_t               out_size = 0;
    size_t               out_pos  = 0;
    std::vector<uint8_t> temp_buffer;
    size_t               size_read = 0;
};

#endif  // LLAMA_HAS_ZSTD
