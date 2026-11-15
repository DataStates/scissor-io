#ifndef SCISSOR_AGGREGATOR_HPP
#define SCISSOR_AGGREGATOR_HPP

// ======================================================
// scissor/aggregator.hpp
//
// Aggregator: gap-aware offset coalescing.
//
// Converts a list of logical chunk offsets (which may be
// scattered) into a smaller list of contiguous read
// segments by merging adjacent offsets that are within
// 'gap' chunks of each other.  This is the core mechanism
// that amortizes I/O request overhead on PFS and NVMe by
// reading slightly more data (the gap region) in exchange
// for issuing far fewer IOPs.
//
// Design: Aggregator::coalesce() is a pure function.  It
// has no mutable state and is safe to call concurrently
// from multiple threads without synchronization.  The
// caller owns the returned Result.
// ======================================================

#include <scissor/io_reader.hpp>
#include <cstddef>
#include <vector>

namespace scissor {

// ======================================================
// Aggregator::Result: output of one coalesce() call.
// ======================================================
struct AggregatorResult {
    // Coalesced read descriptors (id and fd are not set here:
    // the caller fills fd from the LRU cache and id from its own counter).
    std::vector<Segment>  segments;

    // Chunk offsets that are actually covered by the coalesced segments.
    // This may include gap chunks that were not in the original offsets list.
    std::vector<size_t>   covered_offsets;

    // Number of coalesced I/O operations (== segments.size()).
    size_t  iop_count    = 0;

    // Bytes read beyond what was strictly needed due to gap merging.
    size_t  wasted_bytes = 0;

    // Batch boundaries.  Batch i holds segments
    // [batch_ends[i-1], batch_ends[i]) (batch 0 starts at 0), and
    // batch_chunks[i] is the number of requested offsets it covers.
    std::vector<size_t>   batch_ends;
    std::vector<size_t>   batch_chunks;
};

// ======================================================
// Aggregator: stateless coalescing engine.
// ======================================================
class Aggregator {
public:
    // Default constructor.  Aggregator holds no state.
    Aggregator() = default;

    // coalesce: merge 'offsets' into contiguous read segments.
    //
    // Parameters:
    //   offsets          -- logical chunk offsets to read (sorted ascending, no duplicates)
    //   seg_size         -- size of one chunk in bytes
    //   gap              -- merge two consecutive offsets when their index
    //                       distance is <= gap.
    //                       gap=0: no merging, one segment per offset.
    //                       gap=1: only adjacent offsets are merged.
    //                       gap=g: up to g-1 unrequested chunks are read in between.
    //   used_chks_per_read -- minimum number of chunks to accumulate before
    //                       emitting a batch.  Controls how many Segments are
    //                       grouped together before stage_in is called.
    //                       Pass 1 to emit one Segment per coalesced group.
    //
    // Returns: AggregatorResult with coalesced Segments and statistics.
    // The returned Segment objects have offset and size set. id and fd are 0/-1
    // and must be filled by the caller before enqueueing.
    //
    // Thread safety: this function is pure and reentrant.
    AggregatorResult coalesce(const std::vector<size_t>& offsets,
                              size_t                     seg_size,
                              uint32_t                   gap,
                              size_t                     used_chks_per_read = 1) const;
};

} // namespace scissor

#endif // SCISSOR_AGGREGATOR_HPP
