#ifndef SCISSOR_DETAIL_METADATA_QUEUE_HPP
#define SCISSOR_DETAIL_METADATA_QUEUE_HPP

// ======================================================
// scissor/src/common/metadata_queue.hpp
//
// Thread-safe FIFO queue of Batch* pointers.
// Used by HostCache to coordinate the fetch pipeline
// (fetch_q_: pending reads) and the ready pipeline
// (ready_q_: completed reads available to the caller).
//
// Ported from state-diff/src/loader/metadata_queue.hpp.
// Changes: batch_t replaced with scissor::common::Batch.
// ======================================================

#include <scissor/batch.hpp>

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace scissor {
namespace common {

class MetadataQueue {
public:
    MetadataQueue()  = default;
    ~MetadataQueue() = default;

    // Push a batch pointer (non-blocking, wakes one waiter)
    void push(Batch* seg) {
        std::unique_lock<std::mutex> lock(mtx_);
        meta_q_.push_back(seg);
        cv_.notify_one();
    }

    // Pop the front item.
    // Waits until an item is available or the queue is deactivated.
    // Returns without popping if the queue is empty after deactivation.
    void pop() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !meta_q_.empty() || !is_active_; });
        if (!meta_q_.empty()) {
            meta_q_.pop_front();
            cv_.notify_one();
        }
    }

    // Return the front item without removing it.
    // Blocks until an item is available or the queue is deactivated.
    Batch* front() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !meta_q_.empty() || !is_active_; });
        if (meta_q_.empty()) return nullptr;
        return meta_q_.front();
    }

    // Atomically swap the internal deque with the caller's deque.
    // Returns false if the queue is empty.
    bool swap_batches(std::deque<Batch*>& batches) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (meta_q_.empty()) return false;
        batches.swap(meta_q_);
        return true;
    }

    // Block until the queue drains to empty or is deactivated.
    bool wait_until_empty() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return meta_q_.empty() || !is_active_; });
        return true;
    }

    // Block until at least one item arrives or the queue is deactivated.
    bool wait_any() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !meta_q_.empty() || !is_active_; });
        return true;
    }

    // Block until at least 'count' items are present or the queue is deactivated.
    bool wait_for(size_t count) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this, count] {
            return meta_q_.size() >= count || !is_active_;
        });
        return true;
    }

    size_t size() {
        std::unique_lock<std::mutex> lock(mtx_);
        return meta_q_.size();
    }

    // Deactivate: unblocks all waiters so threads can exit cleanly.
    void set_inactive() {
        std::unique_lock<std::mutex> lock(mtx_);
        is_active_ = false;
        cv_.notify_all();
    }

private:
    std::deque<Batch*>      meta_q_;
    std::mutex              mtx_;
    std::condition_variable cv_;
    bool                    is_active_ = true;
};

} // namespace common
} // namespace scissor

#endif // SCISSOR_DETAIL_METADATA_QUEUE_HPP
