#include "host_cache.hpp"
#include <cstring>
#include <cassert>
#include <deque>
#include <system_error>

namespace scissor {
namespace common {

namespace {

// Throw if an IReader call reported a failure (negative errno).
void check_io(int ret, const char* what) {
    if (ret < 0) {
        SCISSOR_ERROR("reader call failed", "call", what, "errno", -ret,
                      "errno_msg", std::strerror(-ret));
        throw std::system_error(-ret, std::generic_category(),
                                std::string("scissor: ") + what);
    }
}

// Delete every batch held by a queue.
void free_queue(MetadataQueue& q) {
    std::deque<Batch*> batches;
    if (q.swap_batches(batches)) {
        for (Batch* b : batches) delete b;
    }
}

} // namespace

HostCache::~HostCache() {
    std::vector<int> ids;
    ids.reserve(fetch_q_.size() + ready_q_.size());
    for (auto& fq : fetch_q_) ids.push_back(fq.first);
    for (auto& rq : ready_q_) ids.push_back(rq.first);
    for (int id : ids) drop(id);
    is_active_ = false;
    for (auto& t  : fetch_thread_) {
        if (t.second.joinable()) t.second.join();
    }
    SCISSOR_DEBUG("HostCache destroyed");
}

void HostCache::stage_in(int id, Batch* batch) {
    // Memory is allocated when the batch is handed to the reader
    // (see enqueue()), so only the in-flight batch and the batch
    // held by the consumer occupy staging memory.
    fetch_q_[id].push(batch);
    SCISSOR_DEBUG("batch staged for I/O",
        "id", id, "batch_len", batch->batch_len, "size_bytes", batch->size);
}

void HostCache::stage_out(int id, Batch* batch) {
    ready_q_[id].push(batch);
    SCISSOR_DEBUG("batch staged out (ready)", "id", id);
}

void HostCache::enqueue(int id, Batch* batch) {
    batch->allocate();
    if (auto* r = std::get_if<IReader*>(&freader_[id])) {
        check_io((*r)->enqueue_reads(batch->to_vec()), "enqueue_reads");
    } else if (auto* rp = std::get_if<std::pair<IReader*, IReader*>>(&freader_[id])) {
        check_io(rp->first->enqueue_reads(batch->left_vec()), "enqueue_reads");
        check_io(rp->second->enqueue_reads(batch->right_vec()), "enqueue_reads");
    }
}

int HostCache::wait_reader(int id) {
    if (auto* r = std::get_if<IReader*>(&freader_[id])) {
        return (*r)->wait_all();
    } else if (auto* rp = std::get_if<std::pair<IReader*, IReader*>>(&freader_[id])) {
        int r0 = rp->first->wait_all();
        int r1 = rp->second->wait_all();
        return (r0 < 0) ? r0 : r1;
    }
    return 0;
}

void HostCache::set_reader(int id, IReader* reader) {
    SCISSOR_INFO("set_reader (single)", "id", id);
    assert(reader != nullptr);
    freader_[id] = reader;

    // Enqueue the first read immediately
    if (fetch_q_[id].size() > 0) {
        enqueue(id, fetch_q_[id].front());
    }
}

void HostCache::set_reader(int id, IReader* reader0, IReader* reader1) {
    SCISSOR_DEBUG("set_reader (dual)", "id", id);
    assert(reader0 != nullptr && reader1 != nullptr);
    freader_[id] = std::make_pair(reader0, reader1);

    // Enqueue the first read on both readers
    if (fetch_q_[id].size() > 0) {
        enqueue(id, fetch_q_[id].front());
    }
}

void HostCache::submit_work(int id) {
    if (freader_.find(id) == freader_.end()) {
        SCISSOR_ERROR("submit_work: no valid reader found", "id", id);
        return;
    }

    // Wait for the current front batch to complete
    Batch* prev_batch = fetch_q_[id].front();
    check_io(wait_reader(id), "wait_all");

    // Move the completed batch to the ready queue
    SCISSOR_DEBUG("batch complete, moving to ready_q", "id", id);
    stage_out(id, prev_batch);
    fetch_q_[id].pop();

    // Enqueue the next batch if more work is queued
    if (fetch_q_[id].size() > 0) {
        enqueue(id, fetch_q_[id].front());
    }
}

bool HostCache::wait_for_completion() {
    SCISSOR_DEBUG("waiting for all fetch and ready queues to drain");
    for (auto& fq : fetch_q_) fq.second.wait_until_empty();
    for (auto& rq : ready_q_) rq.second.wait_until_empty();
    return true;
}

Batch* HostCache::get_completed(int id) {
    SCISSOR_DEBUG("get_completed", "id", id);
    submit_work(id);
    return ready_q_[id].front();
}

bool HostCache::release(int id) {
    auto it = ready_q_.find(id);
    if (it == ready_q_.end() || it->second.size() == 0) return false;
    SCISSOR_DEBUG("release previous batch", "id", id);
    Batch* batch = it->second.front();
    it->second.pop();
    delete batch;
    return true;
}

void HostCache::drop(int id) {
    auto fq = fetch_q_.find(id);
    if (fq != fetch_q_.end()) {
        // The front batch of a handle with a reader is in flight:
        // wait for it before its memory is freed.
        if (fq->second.size() > 0 && freader_.find(id) != freader_.end()) {
            int ret = wait_reader(id);
            if (ret < 0) {
                SCISSOR_WARN("read failed while dropping handle", "id", id,
                             "errno_msg", std::strerror(-ret));
            }
        }
        free_queue(fq->second);
        fetch_q_.erase(fq);
    }
    auto rq = ready_q_.find(id);
    if (rq != ready_q_.end()) {
        free_queue(rq->second);
        ready_q_.erase(rq);
    }
    freader_.erase(id);
    SCISSOR_DEBUG("handle dropped", "id", id);
}

void HostCache::coalesce_and_copy(Batch* batch, void* dst) {
    // Copy all segments in the batch to a contiguous destination buffer.
    // Segments are already contiguous in batch_ptr, but this helper
    // handles non-contiguous cases (e.g., after a buffer shuffle).
    uint8_t* destination = static_cast<uint8_t*>(dst);
    for (size_t i = 0; i < batch->count; ++i) {
        SCISSOR_DEBUG("coalesce_and_copy segment",
            "index", i, "total", batch->count);
        Segment& seg = batch->data[i];
        std::memcpy(destination, seg.buffer, seg.size);
        destination += seg.size;
    }
}

} // namespace common
} // namespace scissor
