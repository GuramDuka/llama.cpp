#include "llama-zstd-io.h"

#include "ggml-backend.h"  // ggml_backend_tensor_get, ggml_backend_tensor_set
#include "llama-impl.h"    // for LLAMA_LOG_ERROR

#include <cstring>
#include <stdexcept>

#ifdef LLAMA_HAS_ZSTD

//
// llama_io_write_buf
//

void llama_io_write_buf::write(const void * src, size_t size) {
    const uint8_t * data = static_cast<const uint8_t *>(src);
    buf.insert(buf.end(), data, data + size);
}

void llama_io_write_buf::write_tensor(ggml_tensor * tensor, size_t offset, size_t size) {
    temp_buffer.resize(size);
    ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
    write(temp_buffer.data(), temp_buffer.size());
}

size_t llama_io_write_buf::n_bytes() {
    return buf.size();
}

//
// llama_io_write_zstd
//

llama_io_write_zstd::llama_io_write_zstd(llama_file * f, int level, const void * dict, size_t dict_size) :
    t_start_us(ggml_time_us()),
    file(f) {
    cctx = ZSTD_createCCtx();
    if (!cctx) {
        throw std::runtime_error("ZSTD_createCCtx failed");
    }
    size_t err;
    if (dict && dict_size > 0) {
        cdidt = ZSTD_createCDict(dict, dict_size, level);
        if (!cdidt) {
            ZSTD_freeCCtx(cctx);
            throw std::runtime_error("ZSTD_createCDict failed");
        }
        err = ZSTD_CCtx_refCDict(cctx, cdidt);
    } else {
        err = ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);
    }
    if (ZSTD_isError(err)) {
        ZSTD_freeCCtx(cctx);
        ZSTD_freeCDict(cdidt);
        throw std::runtime_error("ZSTD_CCtx_setParameter: " + std::string(ZSTD_getErrorName(err)));
    }
    in_buf.reserve(1024 * 1024);
    out_buf.resize(ZSTD_CStreamOutSize());
}

llama_io_write_zstd::~llama_io_write_zstd() {
    if (cctx) {
        // flush any remaining data
        const ZSTD_EndDirective end = ZSTD_e_end;
        ZSTD_outBuffer          out = { out_buf.data(), out_buf.size(), 0 };
        ZSTD_inBuffer           in  = { in_buf.data(), in_buf.size(), 0 };
        while (true) {
            const size_t ret = ZSTD_compressStream2(cctx, &out, &in, end);
            if (out.pos > 0) {
                file->write_raw(out_buf.data(), out.pos);
                size_written += out.pos;
                out.pos = 0;
            }
            if (ret == 0) {
                break;
            }
            if (ZSTD_isError(ret)) {
                break;
            }
        }
        ZSTD_freeCCtx(cctx);
    }
    ZSTD_freeCDict(cdidt);
    g_comp_stats.total_compress_us += ggml_time_us() - t_start_us;
    g_comp_stats.total_uncompressed += uncompressed_bytes;
    g_comp_stats.total_compressed += size_written;
    g_comp_stats.compress_count++;
}

void llama_io_write_zstd::write(const void * src, size_t size) {
    const uint8_t * data = static_cast<const uint8_t *>(src);
    in_buf.insert(in_buf.end(), data, data + size);
    uncompressed_bytes += size;
    flush_stream(ZSTD_e_continue);
}

void llama_io_write_zstd::write_tensor(ggml_tensor * tensor, size_t offset, size_t size) {
    temp_buffer.resize(size);
    ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
    write(temp_buffer.data(), temp_buffer.size());
}

size_t llama_io_write_zstd::n_bytes() {
    return size_written;
}

void llama_io_write_zstd::flush_stream(ZSTD_EndDirective end) {
    ZSTD_outBuffer out = { out_buf.data(), out_buf.size(), 0 };
    ZSTD_inBuffer  in  = { in_buf.data(), in_buf.size(), 0 };
    while (in.pos < in_buf.size()) {
        const size_t ret = ZSTD_compressStream2(cctx, &out, &in, end);
        if (ZSTD_isError(ret)) {
            throw std::runtime_error("ZSTD_compressStream2: " + std::string(ZSTD_getErrorName(ret)));
        }
        if (out.pos > 0) {
            file->write_raw(out_buf.data(), out.pos);
            size_written += out.pos;
            out.pos = 0;
        }
        if (ret == 0 && end == ZSTD_e_continue) {
            break;
        }
    }
    // move remaining unprocessed data to front of buffer
    if (in.pos < in_buf.size()) {
        size_t remaining = in_buf.size() - in.pos;
        memmove(in_buf.data(), in_buf.data() + in.pos, remaining);
        in_buf.resize(remaining);
    } else {
        in_buf.clear();
    }
}

//
// llama_io_read_zstd
//

llama_io_read_zstd::llama_io_read_zstd(llama_file * f, const void * dict, size_t dict_size) :
    t_start_us(ggml_time_us()),
    file(f) {
    dctx = ZSTD_createDCtx();
    if (!dctx) {
        throw std::runtime_error("ZSTD_createDCtx failed");
    }
    if (dict && dict_size > 0) {
        ddct = ZSTD_createDDict(dict, dict_size);
        if (!ddct) {
            ZSTD_freeDCtx(dctx);
            throw std::runtime_error("ZSTD_createDDict failed");
        }
        size_t err = ZSTD_DCtx_refDDict(dctx, ddct);
        if (ZSTD_isError(err)) {
            ZSTD_freeDCtx(dctx);
            ZSTD_freeDDict(ddct);
            throw std::runtime_error("ZSTD_DCtx_refDDict: " + std::string(ZSTD_getErrorName(err)));
        }
    }
    // Determine remaining data size from the current file position
    size_t file_size       = f->size();
    size_t cur_pos         = f->tell();
    size_t compressed_size = (file_size > cur_pos) ? (file_size - cur_pos) : 0;
    if (compressed_size == 0) {
        ZSTD_freeDCtx(dctx);
        ZSTD_freeDDict(ddct);
        throw std::runtime_error("zstd: empty compressed stream");
    }

    // Pre-read all remaining compressed data at once
    compressed_data.resize(compressed_size);
    f->read_raw(compressed_data.data(), compressed_size);
    in_data = compressed_data.data();
    in_size = compressed_data.size();
    in_pos  = 0;

    out_buf.resize(ZSTD_DStreamOutSize());
}

llama_io_read_zstd::~llama_io_read_zstd() {
    ZSTD_freeDCtx(dctx);
    ZSTD_freeDDict(ddct);
    // Record decompression metrics.
    g_comp_stats.total_decompress_us += ggml_time_us() - t_start_us;
    g_comp_stats.total_compressed += in_size;
    g_comp_stats.total_uncompressed += size_read;
    g_comp_stats.decompress_count++;
}

void llama_io_read_zstd::read(void * dst, size_t size) {
    uint8_t * pos = static_cast<uint8_t *>(dst);
    while (size > 0) {
        if (out_pos < out_size) {
            size_t to_copy = std::min(size, out_size - out_pos);
            memcpy(pos, out_buf.data() + out_pos, to_copy);
            pos += to_copy;
            size -= to_copy;
            out_pos += to_copy;
            size_read += to_copy;
            continue;
        }
        decompress_chunk();
    }
}

void llama_io_read_zstd::read_tensor(ggml_tensor * tensor, size_t offset, size_t size) {
    temp_buffer.resize(size);
    read(temp_buffer.data(), size);
    ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
}

size_t llama_io_read_zstd::n_bytes() {
    return size_read;
}

void llama_io_read_zstd::decompress_chunk() {
    out_size = 0;
    out_pos  = 0;

    ZSTD_outBuffer out = { out_buf.data(), out_buf.size(), 0 };
    ZSTD_inBuffer  in  = { in_data, in_size, in_pos };

    size_t ret = ZSTD_decompressStream(dctx, &out, &in);
    if (ZSTD_isError(ret)) {
        throw std::runtime_error("ZSTD_decompressStream: " + std::string(ZSTD_getErrorName(ret)));
    }

    out_size = out.pos;
    in_pos   = in.pos;
}

#endif  // LLAMA_HAS_ZSTD
