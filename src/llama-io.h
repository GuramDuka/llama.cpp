#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;

class llama_io_write_i {
  public:
    llama_io_write_i()          = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size)                           = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
  public:
    llama_io_read_i()          = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size)                                 = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes read so far
    virtual size_t n_bytes() = 0;

    // Flush any deferred tensor writes (no-op for synchronous IO).
    // Must be called before reading tensor data back after all read_tensor calls.
    virtual void flush() {}

    void read_string(std::string & str);
};
