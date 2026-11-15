#ifndef SCISSOR_TYPES_HPP
#define SCISSOR_TYPES_HPP

// ======================================================
// scissor/types.hpp
//
// Public value types shared across the scissor API.
// No dependencies on liburing or CUDA headers.
// ======================================================

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace scissor {

// ======================================================
// TransferType: describes where data lands after a load.
//   FileToHost    -- DMA from storage to CPU (pinned) memory
//   FileToDevice  -- DMA from storage to GPU memory
//   HostToDevice  -- copy already-staged host buffer to GPU
//   HostPinned    -- allocate page-locked host memory (CUDA pinned)
// ======================================================
enum class TransferType {
    FileToHost,
    FileToDevice,
    HostToDevice,
    HostPinned,
};

// ======================================================
// FileSrc: storage tier from which files are read.
// Used by the Regressor to select the correct regression
// model (io_params coefficients) from models.conf.
//   PFS    -- parallel file system (Lustre, GPFS)
//   SSD    -- local NVMe / SSD scratch
//   Hybrid -- burst buffer or mixed topology
// ======================================================
enum class FileSrc {
    PFS,
    SSD,
    Hybrid,
};

// ======================================================
// Config: runtime configuration for a Loader instance.
// Fields may be overridden by environment variables
// (see Loader constructor documentation).
// ======================================================
struct Config {
    bool   use_gpu;           // currently unused
    size_t max_staging_bytes; // 0 means unlimited. Loader clamps batches if set
    size_t max_open_fds;      // LRU FD cache capacity (0: SCISSOR_MAX_OPEN_FDS or 32768)
};

// ======================================================
// FileRequest: one file's contribution to a load call.
// offsets are logical chunk indices, not byte offsets.
// The Loader multiplies by seg_size to get byte offsets.
// ======================================================
struct FileRequest {
    std::string          path;    // absolute or relative file path
    std::vector<size_t>  offsets; // logical chunk offsets to read
};

// ======================================================
// LoadHandle: opaque handle returned by Loader::load().
// Pass to Loader::next() to retrieve completed batches.
// ======================================================
struct LoadHandle {
    int id; // internal slot index, negative means invalid
};

// ======================================================
// LoadParams: coalescing gap (in chunks) and batch size
// (bytes of requested chunks) a load actually used, i.e.
// the Regressor's choice when nthreads > 1.
// ======================================================
struct LoadParams {
    uint32_t gap;
    size_t   block_size;
};

} // namespace scissor

#endif // SCISSOR_TYPES_HPP
