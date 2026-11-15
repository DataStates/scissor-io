#include <scissor/aggregator.hpp>
#include <scissor/log.hpp>
#include <cassert>

namespace scissor {

// ======================================================
// Aggregator::coalesce
//
// Ported from data_loader_t::coalesce() in
// state-diff/src/loader/data_loader.cpp.
//
// Key changes from the original:
//   - Stateless: IOP_count is accumulated locally and
//     returned in AggregatorResult instead of updating
//     a per-id member map.
//   - Uses scissor::Segment as the read descriptor.
//   - Calls no stage_in(). The caller decides what to do
//     with each completed group of Segments (e.g., pass to
//     HostCache::stage_in).
//   - Returns covered_offsets so callers know which chunks
//     were actually read (including gap fills).
//   - No fd or buffer assignment -- caller sets those.
// ======================================================
AggregatorResult
Aggregator::coalesce(const std::vector<size_t>& offsets,
                     size_t                     seg_size,
                     uint32_t                   gap,
                     size_t                     used_chks_per_read) const
{
    SCISSOR_DEBUG("coalesce entry",
        "offsets_in", offsets.size(),
        "gap_chunks", static_cast<size_t>(gap),
        "used_chunks_per_read", used_chks_per_read);
    SCISSOR_TIMER_START(coalesce);

    AggregatorResult result;
    if (offsets.empty()) {
        return result;
    }

    assert(seg_size > 0);
    assert(used_chks_per_read >= 1);

    size_t start          = offsets[0];
    size_t last_offset    = start;
    size_t in_group_offt  = 1;    // number of original offsets in the current group
    size_t wait_for_count = 0;    // accumulated original offsets across current flush group
    size_t iop_id         = 0;    // local IOP counter (returned in result)

    // Pending segments not yet flushed into result.segments
    std::vector<Segment> pending;

    // Helper lambda: flush pending segments and all accumulated
    // covered_offsets into result.  Resets pending and wait_for_count.
    auto flush_pending = [&]() {
        if (pending.empty()) return;
        for (auto& s : pending) {
            result.segments.push_back(s);
        }
        result.batch_ends.push_back(result.segments.size());
        result.batch_chunks.push_back(wait_for_count);
        pending.clear();
        wait_for_count = 0;
    };

    // Helper lambda: close the current group, emit one Segment,
    // accumulate covered offsets, and conditionally flush.
    auto close_group = [&]() {
        // Byte offset of the first chunk in this group
        size_t segment_start = start * seg_size;
        // Number of chunks spanned (including gaps)
        size_t n_segs = last_offset - start + 1;
        size_t combined_size = n_segs * seg_size;
        // Only the original chunks count as "needed"
        size_t needed_size = in_group_offt * seg_size;
        result.wasted_bytes += (combined_size - needed_size);

        // Build a Segment descriptor.  id and fd are placeholders:
        // the caller fills them before passing to IReader::enqueue_reads.
        Segment seg;
        seg.id     = iop_id++;
        seg.fd     = -1;
        seg.offset = segment_start;
        seg.size   = combined_size;
        seg.buffer = nullptr;
        pending.push_back(seg);
        result.iop_count++;

        // Record every chunk offset covered by this segment (including gaps)
        for (size_t c = start; c <= last_offset; ++c) {
            result.covered_offsets.push_back(c);
        }

        wait_for_count += in_group_offt;

        // Flush when we have accumulated at least used_chks_per_read chunks
        if (wait_for_count >= used_chks_per_read) {
            flush_pending();
        }
    };

    for (size_t i = 1; i < offsets.size(); ++i) {
        size_t curr = offsets[i];
        if (curr - last_offset <= static_cast<size_t>(gap)) {
            // Within gap: extend current group
            last_offset = curr;
            in_group_offt++;
        } else {
            // Beyond gap: close the current group and start a new one
            close_group();
            start         = curr;
            last_offset   = curr;
            in_group_offt = 1;
        }
    }

    // Close the final group
    close_group();

    // Flush any remaining pending segments
    flush_pending();

    SCISSOR_TIMER_PERF(coalesce, "coalesce complete",
        "segments_out",  result.segments.size(),
        "iop_count",     result.iop_count,
        "wasted_bytes",  result.wasted_bytes);

    return result;
}

} // namespace scissor
