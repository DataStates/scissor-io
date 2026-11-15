#ifndef SCISSOR_DETAIL_HOST_CACHE_HPP
#define SCISSOR_DETAIL_HOST_CACHE_HPP

// ======================================================
// scissor/src/common/host_cache.hpp
//
// Host-side staging cache: manages the pipeline of
// batch allocations, async I/O submissions, and
// completion delivery to the Loader.
//
// Ported from state-diff/src/loader/host_cache.hpp.
// Key changes:
//   - base_io_reader_t replaced with scissor::IReader.
//   - batch_t replaced with scissor::common::Batch.
//   - debug.hpp replaced with scissor/log.hpp.
//
// Threading model: stage_in() enqueues Batch* on fetch_q_.
// set_reader() hands off the first batch immediately to
// IReader::enqueue_reads().  Subsequent batches are
// dequeued lazily in submit_work(), which is called from
// get_completed() before returning the next ready batch.
// A batch's buffer is allocated when it is handed to the
// reader and freed by release() or drop().
// ======================================================

#include "base_cache.hpp"
#include <scissor/io_reader.hpp>

#include <cstdint>
#include <cstring>
#include <variant>
#include <unordered_map>

namespace scissor {
namespace common {

class HostCache : public BaseCache {
    using SingleOrDoubleReader =
        std::variant<IReader*, std::pair<IReader*, IReader*>>;

    std::unordered_map<int, SingleOrDoubleReader> freader_;

    // Wait for the current front batch to complete, stage it out,
    // and enqueue the next batch (if any).
    void submit_work(int id);

    // Allocate a batch and hand it to the handle's reader(s).
    void enqueue(int id, Batch* batch);

    // wait_all() on the handle's reader(s), returns the first error.
    int wait_reader(int id);

public:
    HostCache()  = default;
    ~HostCache();

    // Single-reader path (sequential or single-file load)
    void set_reader(int id, IReader* reader);

    // Dual-reader path (two-file parallel load)
    void set_reader(int id, IReader* reader0, IReader* reader1);

    void activate(int id) override {}  // lifecycle managed by set_reader
    void fetch_(int id)   override {}  // driven by submit_work, not a separate thread

    void stage_in(int id, Batch* batch) override;
    void stage_out(int id, Batch* batch) override;
    bool wait_for_completion() override;
    Batch* get_completed(int id) override;
    bool release(int id) override;

    // Wait for the handle's in-flight reads, then free all of its
    // batches and forget the handle.
    void drop(int id);
    void coalesce_and_copy(Batch* batch, void* dst) override;
};

} // namespace common
} // namespace scissor

#endif // SCISSOR_DETAIL_HOST_CACHE_HPP
