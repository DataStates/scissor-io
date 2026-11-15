#include <scissor/readers.hpp>
#include <scissor/log.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <system_error>

namespace scissor {

LiburingReader::LiburingReader() : LiburingReader(DEFAULT_RING_SIZE, DEFAULT_BATCH_SIZE) {
}

LiburingReader::LiburingReader(size_t qdepth, size_t cqdepth) {
    // Set ring params
    memset(&ring_params, 0, sizeof(ring_params));
    ring_params.flags |= IORING_SETUP_SQPOLL;
    ring_params.sq_thread_idle = 2000;

    MAX_RING_SIZE = qdepth;
    BATCH_SIZE    = cqdepth;
    // Initialize ring
    int ret = io_uring_queue_init_params(MAX_RING_SIZE, &ring, &ring_params);
    if (ret < 0) {
        throw std::system_error(-ret, std::generic_category(),
                                "scissor: io_uring_queue_init_params failed");
    }
    req_submitted = 0;
    req_completed = 0;
    req_consumed  = 0;
    return;
}

LiburingReader::~LiburingReader() {
    // Destroy queue
    io_uring_queue_exit(&ring);
}

uint32_t LiburingReader::submit_more() {
    uint32_t nsubmitted = 0;
    while (!submissions.empty()) {
        // Get pointer to next submission queue entry
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        if (!sqe)
            break;
        // Get segment from queue
        auto seg = submissions.front();
        // Save ID
        // Instead of the file offset, we are using a unique ID for the segment
        // that users should provide, because multiple files in the same ring with
        // the same ID introduces ambiguity for the kernel to determine which request
        // completed.
        io_uring_sqe_set_data64(sqe, seg.id);
        io_uring_prep_read(sqe, seg.fd, seg.buffer, seg.size, seg.offset);
        // Remove segment from queue
        submissions.pop();
        nsubmitted += 1;
    };
    // Submit queued reads
    int ret = io_uring_submit(&ring);
    if (ret < 0)
        throw std::system_error(-ret, std::generic_category(),
                                "scissor: io_uring_submit failed");

    req_submitted += nsubmitted;
    return nsubmitted;
}

// Process any available completions
uint32_t LiburingReader::request_completion() {
    size_t in_progress = req_submitted - req_completed;

    if (in_progress == 0)
        return 0;

    // Get number of entries to consume
    uint32_t nentries = (in_progress < BATCH_SIZE)
                            ? static_cast<uint32_t>(in_progress)
                            : static_cast<uint32_t>(BATCH_SIZE);

    struct io_uring_cqe* one_entry = nullptr;
    int ret = io_uring_wait_cqe_nr(&ring, &one_entry, nentries);
    if (ret < 0) {
        fprintf(stderr, "io_uring_wait_cqe_nr %u: %s\n", nentries, strerror(-ret));
    }

    // Iterates completion events
    uint32_t nready = 0;
    uint32_t head;
    struct io_uring_cqe* cqe_ptr;
    io_uring_for_each_cqe(&ring, head, cqe_ptr) {
        if (cqe_ptr->res < 0) {
            fprintf(stderr, "read error: %s\n", strerror(-cqe_ptr->res));
            if (first_error == 0) first_error = cqe_ptr->res;
        }
        size_t id = io_uring_cqe_get_data64(cqe_ptr);
        completions.insert(id);
        nready += 1;
    }
    // Update ring and count of total requests completed
    io_uring_cq_advance(&ring, nready);
    req_completed += nready;

    return nready;
}

int LiburingReader::enqueue_reads(const std::vector<Segment>& segments) {
    // Push segments onto queue and add ID to the in progress set
    SCISSOR_TIMER_START(seg_prep);
    for (size_t i = 0; i < segments.size(); i++) {
        submissions.push(segments[i]);
    }
    SCISSOR_TIMER_PERF(seg_prep, "Prepare segments",
                       "count", segments.size());

    SCISSOR_TIMER_START(subm);
    submit_more();
    SCISSOR_TIMER_PERF(subm, "Post enqueue submit",
                       "submitted", req_submitted,
                       "in_flight", req_submitted - req_completed,
                       "sq_depth",  submissions.size());

    return 0;
}

int LiburingReader::wait(size_t id) {
    while (completions.find(id) == completions.end()) {
        // Do not wait if there is nothing inflight
        if (req_completed >= req_submitted && submissions.empty())
            return -1;
        request_completion();
        if (!submissions.empty()) {
            submit_more();
        }
    }
    completions.erase(id);
    req_consumed += 1;
    return 0;
}

std::vector<size_t> LiburingReader::wait_n(size_t n) {
    size_t nwait = std::min(n, req_submitted - req_consumed);
    std::vector<size_t> ids;
    while ((nwait > 0) && completions.size() < nwait) {
        request_completion();
        if (submissions.size() > 0)
            submit_more();
    }
    for (size_t i = 0; i < nwait; i++) {
        auto iter = completions.begin();
        std::advance(iter, i);
        size_t id = *(iter);
        ids.push_back(id);
    }
    auto iter = completions.begin();
    std::advance(iter, nwait);
    completions.erase(completions.begin(), iter);
    req_consumed += nwait;
    return ids;
}

size_t LiburingReader::wait_any() {
    while (completions.empty()) {
        // Do not wait if there is nothing inflight or to be submitted
        if (req_completed >= req_submitted && submissions.empty())
            return static_cast<size_t>(-1);
        request_completion();
        if (!submissions.empty()) {
            submit_more();
        }
    }
    size_t id = *completions.begin();
    completions.erase(id);
    req_consumed += 1;
    return id;
}

int LiburingReader::wait_all() {

    SCISSOR_TIMER_START(wait_loop);
    size_t pending   = req_submitted - req_completed;  // inflight sqes
    size_t remainder = submissions.size() + pending;   // still-to-submit + inflight

    std::vector<io_uring_cqe*> cqes(BATCH_SIZE);

    while (remainder > 0) {
        uint32_t want      = static_cast<uint32_t>(std::min(remainder, BATCH_SIZE));
        uint32_t completed = io_uring_peek_batch_cqe(&ring, cqes.data(), want);

        if (completed > 0) {
            // We only need this loop to ensure the data is properly read
            for (uint32_t i = 0; i < completed; ++i) {
                io_uring_cqe* cqe_ptr = cqes[i];
                if (cqe_ptr->res < 0) {
                    fprintf(stderr, "read error: %s\n", strerror(-cqe_ptr->res));
                    if (first_error == 0) first_error = cqe_ptr->res;
                }
            }
            // Mark completed requests
            io_uring_cq_advance(&ring, completed);
            req_completed += completed;

            pending   -= completed;
            remainder  = submissions.size() + pending;

            // Submit more if there is more in the submission queue
            // and update the remainder value
            if (!submissions.empty()) {
                pending   += submit_more();
                remainder  = submissions.size() + pending;
            }
        } else {
            // There is nothing inflight
            if (pending == 0 && submissions.empty())
                break;
            struct io_uring_cqe* one_entry;
            uint32_t wait_n = static_cast<uint32_t>(std::min(pending, BATCH_SIZE));
            if (wait_n == 0)
                wait_n = 1;  // to ensure that we at least wait for one element
            int ret = io_uring_wait_cqe_nr(&ring, &one_entry, wait_n);
            if (ret < 0)
                fprintf(stderr, "io_uring_wait_cqe_nr: %s\n", strerror(-ret));
        }
    }
    SCISSOR_TIMER_PERF(wait_loop, "Loop submit and complete",
                       "total_submitted", req_submitted);
    completions.clear();
    req_consumed = req_submitted;

    int err = first_error;
    first_error = 0;
    return err;
}

} // namespace scissor
