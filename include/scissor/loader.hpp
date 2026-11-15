#ifndef SCISSOR_LOADER_HPP
#define SCISSOR_LOADER_HPP

// ======================================================
// scissor/loader.hpp
//
// Loader: high-level I/O pipeline.
//
// The Loader wraps an IReader backend and provides a
// load() / next() API for staging data from one or more
// files into host (or device) memory via the HostCache
// pipeline.
//
// Public API:
//   LoadHandle load(requests, seg_size, tier, src, ...)
//     -- enqueue a multi-file scatter read
//   LoadHandle load(file, total_size, seg_size, tier)
//     -- enqueue a sequential single-file read
//   LoadHandle load(file0, file1, offsets, ...)
//     -- convenience: two-file comparison pattern
//   Batch* next(handle, tier)
//     -- block until the next batch is ready, returns nullptr when done
//   void close(handle)
//     -- release resources for a completed load
//   size_t wasted_bytes(handle), iop_count(handle)
//     -- I/O efficiency statistics
//   LoadParams params(handle)
//     -- gap and batch size the load used
//
// Thread safety: a Loader is not thread-safe.  Calls on one
// Loader must come from one thread or be serialized by the
// caller.  Several handles may be open at once and their
// next() calls interleaved.  For parallel loading, use one
// Loader and one reader per thread.
//
// Errors: load() throws std::invalid_argument for invalid
// arguments and std::system_error when a file cannot be
// opened. next() throws std::system_error when a read
// fails.  After an exception from next(), close the handle.
// ======================================================

#include <scissor/io_reader.hpp>
#include <scissor/types.hpp>
#include <scissor/aggregator.hpp>
#include <scissor/regressor.hpp>
#include <scissor/batch.hpp>   // common::Batch: public because next() returns common::Batch*

#include <cstddef>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration (internal implementation only)
namespace scissor { namespace common { class HostCache; } }

namespace scissor {

class Loader {
public:
    // ======================================================
    // Constructor.
    //
    // 'reader' must outlive this Loader.  The Loader calls
    // reader.enqueue_reads() and reader.wait_all() internally.
    //
    // Config field defaults (overridden by env vars if zero):
    //   cfg.max_open_fds    -- 0 means read SCISSOR_MAX_OPEN_FDS (default 32768)
    //   cfg.max_staging_bytes -- 0 means read SCISSOR_MAX_STAGING_MB (unset: unlimited)
    //   cfg.use_gpu         -- currently unused
    // ======================================================
    explicit Loader(IReader& reader, Config cfg = {false, 0, 0});

    // Destructor: closes all open file descriptors.
    ~Loader();

    // ======================================================
    // load: generalized N-file scatter read.
    //
    // 'requests' describes one or more files and the logical
    // chunk offsets to read from each.  'seg_size' is the
    // size of one chunk in bytes.
    //
    // If nthreads > 1, the Regressor is invoked to optimize
    // gap and block_size before coalescing.
    //
    // Returns a LoadHandle. Pass it to next() to retrieve batches.
    //
    // Requirements (std::invalid_argument otherwise): seg_size is
    // a multiple of 4096, offsets are sorted ascending without
    // duplicates, and every request uses the same offsets.
    // ======================================================
    LoadHandle load(const std::vector<FileRequest>& requests,
                    size_t                          seg_size,
                    TransferType                    tier,
                    FileSrc                         src,
                    int                             nthreads   = 1,
                    uint32_t                        gap        = 0,
                    size_t                          block_size = 0);

    // ======================================================
    // load: single-file sequential read.
    //
    // Reads 'total_read_size' bytes from 'file' in batches of
    // approximately 'seg_size' bytes each.
    // ======================================================
    LoadHandle load(const std::string& file,
                    size_t             total_read_size,
                    size_t             seg_size,
                    TransferType       tier);

    // ======================================================
    // load: two-file comparison pattern.
    //
    // Convenience overload for the reproducibility check
    // (state-diff) pattern: two checkpoint files are read at
    // the same set of offsets simultaneously.
    // ======================================================
    LoadHandle load(const std::string&       file0,
                    const std::string&       file1,
                    std::vector<size_t>      offsets,
                    size_t                   seg_size,
                    TransferType             tier,
                    FileSrc                  src,
                    int                      nthreads   = 1,
                    uint32_t                 gap        = 0,
                    size_t                   block_size = 0);

    // ======================================================
    // next: retrieve the next completed batch.
    //
    // Blocks until a batch is ready.  Returns nullptr when
    // all batches for the given handle have been returned.
    // The returned pointer is valid until the next call to
    // next() or close() with the same handle (which frees the
    // previous batch).
    // ======================================================
    common::Batch* next(LoadHandle handle, TransferType tier);

    // ======================================================
    // close: release all internal resources for a handle.
    // May be called at any time: reads still in flight for the
    // handle are awaited and its remaining batches are freed.
    // ======================================================
    void close(LoadHandle handle);

    // I/O efficiency statistics (valid after load() completes)
    size_t wasted_bytes(LoadHandle handle) const;
    size_t iop_count(LoadHandle handle)    const;

    // Gap and batch size used by a scatter load (the Regressor's
    // choice when nthreads > 1).  {0, 0} for sequential loads.
    LoadParams params(LoadHandle handle) const;

private:
    // ======================================================
    // get_or_open_fd: LRU-caching file descriptor manager.
    //
    // On hit: moves the entry to the front of the LRU list.
    // On miss at capacity: closes the least recently used FD,
    // opens the new file with O_RDONLY | O_DIRECT.
    // ======================================================
    int get_or_open_fd(const std::string& path);

    // Sequential batch staging (single-file load)
    void enqueue_reads_sequential(int id, int fd, size_t seg_size,
                                  size_t n_segs_per_read,
                                  size_t total_n_segs,
                                  size_t total_read_size);

    IReader&            reader_;      // I/O backend (not owned)
    Config              cfg_;         // effective configuration
    common::HostCache*  host_cache_;  // staging pipeline (owned)
    Aggregator          aggregator_;  // stateless coalescing engine

    // Per-handle counters
    std::unordered_map<int, size_t> wasted_bytes_;
    std::unordered_map<int, size_t> iop_count_;
    std::unordered_map<int, size_t> ready_count_;    // call count for next()
    std::unordered_map<int, size_t> total_batches_;  // batches staged per handle
    std::unordered_map<int, bool>   holding_;        // a batch returned by next() is not yet freed
    std::unordered_map<int, LoadParams> params_;   // gap/block used per scatter load
    int                             instance_count_ = 0;

    // LRU FD cache
    using LruEntry = std::pair<std::string, int>;
    std::list<LruEntry>                                 lru_list_;
    std::unordered_map<std::string,
                       std::list<LruEntry>::iterator>   lru_map_;
    size_t                                              lru_cap_  = 64;
};

} // namespace scissor

#endif // SCISSOR_LOADER_HPP
