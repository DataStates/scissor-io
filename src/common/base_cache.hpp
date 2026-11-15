#ifndef SCISSOR_DETAIL_BASE_CACHE_HPP
#define SCISSOR_DETAIL_BASE_CACHE_HPP

// ======================================================
// scissor/src/common/base_cache.hpp
//
// Abstract base class for the staging cache tier.
// Ported from state-diff/src/loader/base_cache.hpp.
//
// Key changes:
//   - batch_t replaced with scissor::common::Batch.
//   - Unimplemented commented-out members removed.
// ======================================================

#include "metadata_queue.hpp"

#include <atomic>
#include <shared_mutex>
#include <thread>
#include <unordered_map>

namespace scissor {
namespace common {

class BaseCache {
protected:
    size_t tot_cache_size_ = 0;

    std::unordered_map<int, std::thread>        fetch_thread_;
    std::unordered_map<int, MetadataQueue>      fetch_q_;
    std::unordered_map<int, MetadataQueue>      ready_q_;
    std::atomic<bool>                           is_active_ = true;
    std::unordered_map<int, std::shared_mutex>  fetch_q_mutex_;
    std::unordered_map<int, std::shared_mutex>  ready_q_mutex_;

public:
    BaseCache()           = default;
    virtual ~BaseCache()  = default;

    virtual void activate(int id)                          = 0;
    virtual void stage_in(int id, Batch* batch)            = 0;
    virtual void stage_out(int id, Batch* batch)           = 0;
    virtual void fetch_(int id)                            = 0;
    virtual bool wait_for_completion()                     = 0;
    virtual Batch* get_completed(int id)                   = 0;
    virtual bool release(int id)                           = 0;
    virtual void coalesce_and_copy(Batch* batch, void* dst) = 0;
};

} // namespace common
} // namespace scissor

#endif // SCISSOR_DETAIL_BASE_CACHE_HPP
