/*
 * engine_uring.cpp: io_uring engine via liburing
 * Uses SQE/CQE ring buffers for near-zero-syscall async IO.
 * Requires: -luring  (liburing-dev / liburing-devel, kernel >= 5.1)
 *
 * The ring is set up like scissor::LiburingReader: IORING_SETUP_SQPOLL with
 * sq_thread_idle = 2000 ms, and ring size = configured queue depth.
 *
 * Each io_req_t is broken into ceil(length/chunk_size) chunk-sized SQEs so
 * the submission count matches the N reads that scissor coalesces into M < N
 * segments.  A per-sample sample_tracker_t aggregates chunk CQEs, and reap()
 * returns completed-sample counts, not chunk counts.
 */
#include "io_engine.h"
#include <liburing.h>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <ctime>
#include <cstdio>
#include <fcntl.h>
#include <atomic>

/* ── Timing helper ───────────────────────────────────────────────── */
static inline uint64_t uring_mono_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL
         + static_cast<uint64_t>(ts.tv_nsec);
}

/* ── Per-sample completion tracker ──────────────────────────────── */
/*
 * One tracker per io_req_t.  chunks_left counts outstanding chunk SQEs, and
 * when it reaches zero the sample is complete.  error captures the first
 * negative CQE result seen across all chunks.
 */
struct uring_sample_tracker_t {
    io_req_t        *req;
    std::atomic<int> chunks_left;
    std::atomic<int> error;       /* 0 or first negative CQE result */

    uring_sample_tracker_t(io_req_t *r, int n)
        : req(r), chunks_left(n), error(0) {}
};

typedef struct {
    struct io_uring    ring;
    int                queue_depth;
    std::atomic<int>   n_inflight;  /* in-flight samples (not chunks) */
    int                use_sqpoll;
} uring_priv_t;

static int uring_init(io_engine_t *eng, int queue_depth)
{
    uring_priv_t *p = static_cast<uring_priv_t *>(calloc(1, sizeof(*p)));
    p->queue_depth  = queue_depth;
    p->use_sqpoll   = 1;

    struct io_uring_params params;
    memset(&params, 0, sizeof(params));

    if (p->use_sqpoll) {
        params.flags          = IORING_SETUP_SQPOLL;
        params.sq_thread_idle = 2000;
    }

    int ret = io_uring_queue_init_params(queue_depth, &p->ring, &params);
    if (ret < 0) {
        fprintf(stderr, "io_uring_queue_init_params(%d) failed: %s\n",
                queue_depth, strerror(-ret));
        free(p); return ret;
    }

    p->n_inflight.store(0);
    eng->priv        = p;
    eng->queue_depth = queue_depth;
    return 0;
}

static int uring_submit(io_engine_t *eng, io_req_t *reqs, int n)
{
    uring_priv_t *p = static_cast<uring_priv_t *>(eng->priv);
    size_t chunk_size = (eng->chunk_size > 0) ? eng->chunk_size : 4096;
    int samples_submitted = 0;

    uint64_t submit_time = uring_mono_ns();

    for (int i = 0; i < n; i++) {
        size_t n_chunks = (reqs[i].length + chunk_size - 1) / chunk_size;

        reqs[i].submit_ns = submit_time;
        reqs[i].status    = 0;

        auto *tracker = new uring_sample_tracker_t(
                                &reqs[i], static_cast<int>(n_chunks));

        int chunks_queued = 0;
        for (size_t c = 0; c < n_chunks; c++) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&p->ring);
            if (!sqe) {
                /* SQ full: flush pending SQEs to kernel to reclaim slots.
                 * With SQPOLL the kernel thread frees slots asynchronously,
                 * so wait until at least one becomes available. */
                io_uring_submit(&p->ring);
                if (p->use_sqpoll)
                    io_uring_sqring_wait(&p->ring);
                sqe = io_uring_get_sqe(&p->ring);
                if (!sqe) {
                    fprintf(stderr,
                            "io_uring: SQ still full after flush "
                            "(queue_depth=%d may be too small for chunk-level IO)\n",
                            p->queue_depth);
                    break;
                }
            }

            io_uring_prep_read(sqe,
                               reqs[i].fd,
                               static_cast<char *>(reqs[i].buf) + c * chunk_size,
                               chunk_size,
                               reqs[i].offset + c * chunk_size);
            io_uring_sqe_set_data(sqe, tracker);
            chunks_queued++;
        }

        int missed = static_cast<int>(n_chunks) - chunks_queued;
        if (missed > 0) {
            /*
             * Adjust tracker for chunks that were never queued.
             * fetch_sub returns the OLD value. If it equals missed the
             * tracker's count has been fully accounted for without any
             * in-flight SQEs, so free it immediately as an error.
             */
            if (tracker->chunks_left.fetch_sub(missed) == missed) {
                tracker->req->status      = -ENOBUFS;
                tracker->req->complete_ns = uring_mono_ns();
                delete tracker;
                continue;
            }
        }

        samples_submitted++;
    }

    int ret = io_uring_submit(&p->ring);
    if (ret < 0) {
        fprintf(stderr, "io_uring_submit: %s\n", strerror(-ret));
        return ret;
    }

    p->n_inflight.fetch_add(samples_submitted);
    return samples_submitted;
}

static int uring_reap(io_engine_t *eng, int min_completions, int timeout_us)
{
    uring_priv_t *p = static_cast<uring_priv_t *>(eng->priv);

    if (p->n_inflight.load() == 0 && min_completions == 0)
        return 0;

    int reaped_samples = 0;

    uint64_t deadline_ns = 0;
    if (timeout_us > 0)
        deadline_ns = uring_mono_ns()
                    + static_cast<uint64_t>(timeout_us) * 1000ULL;

    /*
     * Loop until at least min_completions samples are reaped or timeout.
     * Each iteration waits for ≥1 CQE (chunk completion), and a sample is
     * counted only when its last chunk's CQE arrives (chunks_left → 0).
     */
    while (reaped_samples < min_completions) {
        struct io_uring_cqe *cqe;
        int wait_ret;

        if (deadline_ns > 0) {
            uint64_t now = uring_mono_ns();
            if (now >= deadline_ns) break;
            uint64_t remaining_us = (deadline_ns - now) / 1000ULL;
            struct __kernel_timespec ts = {
                .tv_sec  = static_cast<long long>(remaining_us / 1000000ULL),
                .tv_nsec = static_cast<long long>(remaining_us % 1000000ULL)
                         * 1000LL,
            };
            wait_ret = io_uring_wait_cqes(&p->ring, &cqe, 1, &ts, nullptr);
        } else {
            wait_ret = io_uring_wait_cqe(&p->ring, &cqe);
        }

        if (wait_ret < 0) break;

        uint64_t now = uring_mono_ns();
        int cqe_count = 0;
        unsigned head;

        io_uring_for_each_cqe(&p->ring, head, cqe) {
            auto *tracker = static_cast<uring_sample_tracker_t *>(
                                io_uring_cqe_get_data(cqe));
            if (tracker) {
                if (cqe->res < 0) {
                    int expected = 0;
                    tracker->error.compare_exchange_strong(expected, cqe->res);
                }
                if (tracker->chunks_left.fetch_sub(1) == 1) {
                    /* Last chunk for this sample */
                    tracker->req->complete_ns = now;
                    tracker->req->status      = (tracker->error.load() == 0)
                                               ? 1 : tracker->error.load();
                    delete tracker;
                    reaped_samples++;
                }
            }
            cqe_count++;
        }
        io_uring_cq_advance(&p->ring, cqe_count);
    }

    p->n_inflight.fetch_sub(reaped_samples);
    return reaped_samples;
}

static void uring_destroy(io_engine_t *eng)
{
    uring_priv_t *p = static_cast<uring_priv_t *>(eng->priv);
    io_uring_queue_exit(&p->ring);
    free(p);
}

static io_engine_ops_t uring_ops = {
    .name    = "io_uring",
    .init    = uring_init,
    .submit  = uring_submit,
    .reap    = uring_reap,
    .destroy = uring_destroy,
};

io_engine_t *engine_uring_create(void)
{
    io_engine_t *e = static_cast<io_engine_t *>(calloc(1, sizeof(*e)));
    e->ops = &uring_ops;
    return e;
}
