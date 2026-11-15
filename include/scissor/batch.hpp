#ifndef SCISSOR_BATCH_HPP
#define SCISSOR_BATCH_HPP

// ======================================================
// scissor/batch.hpp
//
// Public header for scissor::common::Batch, the internal
// batch descriptor that Loader::next() returns.
//
// This type is part of the public API because Loader::next()
// returns common::Batch*.  It lives in the scissor::common
// namespace to signal that callers should access data through
// the Segment vector (data[i].buffer) rather than treating
// the struct as a value type.
//
// Lifetime: the returned pointer is valid until the next call
// to Loader::next() with the same LoadHandle, which releases
// the previous batch's memory back to the staging pool.
// ======================================================

#include <scissor/io_reader.hpp>
#include <scissor/log.hpp>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace scissor {
namespace common {

struct Batch {
    std::vector<Segment> data;         // Segment descriptors (pre-allocated)
    size_t batch_len;                  // Maximum number of segments in this batch
    size_t proc_offt;                  // Number of requested (logical) offsets in this batch
    size_t count;                      // Number of segments actually pushed
    size_t size;                       // Total bytes (sum of all segment sizes)
    uint8_t* batch_ptr = nullptr;      // Contiguous backing allocation (set by allocate())

    size_t chunk_size_bytes_   = 0;    // bytes per logical chunk
    std::vector<size_t> seg_first_chunk_; // Position (in chunks) of the first chunk in each segment
    std::vector<size_t> seg_num_chunks_;  // Number of chunks in each segment
    size_t total_n_chunks_     = 0;    // Total number of chunks in this batch

    size_t interleave_         = 1;    // 1 for single-file, 2 for {prev,cur} dual-file pattern
    size_t update_counter_     = 0;    // Rotates 0..interleave_-1, chunk mapping updated only on 0

    // ======================================================
    // Constructor.
    //   size_       : maximum number of segments
    //   offt_count  : number of logical offsets (proc_offt)
    //   chunk_size  : bytes per logical chunk (must be 4K-aligned)
    //   interleave  : 2 for dual-file reads, 1 otherwise
    // ======================================================
    Batch(size_t size_, size_t offt_count = 0,
          size_t chunk_size = 0, size_t interleave = 1)
        : data(size_), batch_len(size_), proc_offt(offt_count),
          count(0), size(0), chunk_size_bytes_(chunk_size),
          interleave_(interleave) {
        assert((chunk_size_bytes_ % 4096) == 0 &&
               "chunk_size must be a multiple of 4096 (O_DIRECT requirement)");
    }

    // Non-copyable via assignment. The copy constructor is used only by
    // host_cache to create a per-tier replica.
    Batch& operator=(const Batch&) = delete;

    Batch(const Batch& other)
        : data(other.batch_len), batch_len(other.batch_len),
          proc_offt(other.proc_offt), count(0), size(other.size),
          chunk_size_bytes_(other.chunk_size_bytes_),
          seg_first_chunk_(other.seg_first_chunk_),
          seg_num_chunks_(other.seg_num_chunks_),
          total_n_chunks_(other.total_n_chunks_) {
        for (size_t i = 0; i < other.count; ++i) {
            SCISSOR_DEBUG("copy segment",
                "index", i, "offset", other.data[i].offset,
                "size_kb", other.data[i].size / 1024);
            data[i] = other.data[i];
            // buffer is not shared: the destination tier allocates its own
            data[i].buffer = nullptr;
        }
    }

    explicit Batch(Batch* other) : Batch(*other) {}

    ~Batch() {
        if (batch_ptr) {
            free(batch_ptr);
            batch_ptr = nullptr;
        }
    }

    // ======================================================
    // push: append one Segment to this batch.
    // Updates chunk position tracking only on the first
    // interleave slot (update_counter_ == 0).
    // ======================================================
    void push(Segment item) {
        assert(count < batch_len);
        // O_DIRECT alignment enforcement
        assert((item.offset % 4096) == 0 &&
               "Segment offset must be 4K-aligned");
        assert((item.size   % 4096) == 0 &&
               "Segment size must be a multiple of 4096");
        assert(chunk_size_bytes_ > 0 &&
               "chunk_size_bytes_ must be set before push()");
        assert((item.size % chunk_size_bytes_) == 0 &&
               "Segment size must be a multiple of chunk_size");

        if (update_counter_ == 0) {
            seg_first_chunk_.push_back(total_n_chunks_);
            size_t seg_chunks = item.size / chunk_size_bytes_;
            seg_num_chunks_.push_back(seg_chunks);
            total_n_chunks_ += seg_chunks;
        }

        data[count++] = item;
        size += item.size;
        update_counter_ = (update_counter_ + 1) % interleave_;
    }

    // ======================================================
    // allocate: claim a single contiguous 4K-aligned buffer
    // and distribute sub-ranges to each segment's buffer pointer.
    // Must be called before the batch is handed to IReader::enqueue_reads.
    // ======================================================
    void allocate() {
        SCISSOR_DEBUG("batch allocate",
            "batch_len", batch_len, "total_bytes", size);
        constexpr size_t DIRECT_ALIGN = 4096;
        assert((size % DIRECT_ALIGN) == 0 &&
               "batch total size must be a multiple of 4096 (O_DIRECT)");

        if (batch_ptr) {
            free(batch_ptr);
            batch_ptr = nullptr;
        }

        void* ptr = nullptr;
        int ret = posix_memalign(&ptr, DIRECT_ALIGN, size);
        if (ret != 0) {
            SCISSOR_FATAL("posix_memalign failed",
                "errno_msg", std::strerror(ret),
                "size", size);
        }
        batch_ptr = static_cast<uint8_t*>(ptr);

        size_t cur = 0;
        for (size_t i = 0; i < count; ++i) {
            data[i].buffer = batch_ptr + cur;
            cur += data[i].size;
        }
        assert(cur == size);
    }

    // ======================================================
    // Accessors for sub-views used by HostCache.
    // ======================================================
    std::vector<Segment> to_vec() const {
        return std::vector<Segment>(data.begin(), data.begin() + count);
    }
    std::vector<Segment> left_vec() const {
        // Left half: first file's segments in a dual-file batch
        return std::vector<Segment>(data.begin(), data.begin() + count / 2);
    }
    std::vector<Segment> right_vec() const {
        // Right half: second file's segments in a dual-file batch
        return std::vector<Segment>(data.begin() + count / 2, data.begin() + count);
    }

    size_t n_segments()   const { return count; }
    size_t n_chunks()     const { return total_n_chunks_; }
    const std::vector<size_t>& first_chunk_idx() const { return seg_first_chunk_; }
    const std::vector<size_t>& chunk_counts()    const { return seg_num_chunks_; }
};

} // namespace common
} // namespace scissor

#endif // SCISSOR_BATCH_HPP
