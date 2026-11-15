/*
 * engine_libaio.cpp: Linux native asynchronous IO (libaio) engine
 * Uses io_submit() / io_getevents() for async pread operations.
 * Requires: -laio  (libaio-dev / libaio-devel package)
 *
 * Each io_req_t is broken into ceil(length/chunk_size) chunk-sized iocbs so
 * the submission count matches the N reads that scissor coalesces into M < N
 * segments.  A per-sample sample_tracker_t aggregates chunk events, and reap()
 * returns completed-sample counts, not chunk counts.
 */
#include "io_engine.h"
#include <libaio.h>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <ctime>
#include <cstdio>
#include <atomic>
#include <vector>

/* ── Timing helper ───────────────────────────────────────────────── */
static inline uint64_t libaio_mono_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL
         + static_cast<uint64_t>(ts.tv_nsec);
}

/* ── Per-sample completion tracker ──────────────────────────────── */
/*
 * One tracker per io_req_t.  iocbs holds one iocb per chunk, and their
 * lifetime matches the tracker so the kernel always has a valid iocb
 * until the last io_getevents() event arrives.
 */
struct libaio_sample_tracker_t {
    io_req_t            *req;
    std::atomic<int>     chunks_left;
    std::atomic<int>     error;       /* 0 or first negative result   */
    std::vector<struct iocb> iocbs;   /* one iocb per chunk, lifetime = tracker */

    libaio_sample_tracker_t(io_req_t *r, int n)
        : req(r), chunks_left(n), error(0), iocbs(static_cast<size_t>(n)) {}
};

typedef struct {
    io_context_t       ctx;
    struct io_event   *events;
    int                max_events;
    std::atomic<int>   n_inflight;   /* in-flight samples (not chunks) */
} libaio_priv_t;

static int libaio_init(io_engine_t *eng, int queue_depth)
{
    libaio_priv_t *p = static_cast<libaio_priv_t *>(calloc(1, sizeof(*p)));
    p->max_events    = queue_depth;
    p->ctx           = 0;

    /*
     * Size the kernel AIO context to hold all simultaneously in-flight iocbs.
     * Each sample generates ceil(record_size / chunk_size) iocbs.
     * chunk_size is set on eng before engine_init is called (bench.cpp:487).
     * We bound max record size at 1 MB, generous for any training sample
     * (ResNet 160 KB, BERT 8 KB, DLRM 160 B).  With chunk_size == record_size
     * (the intended configuration) this evaluates to 1 chunk per sample and
     * ctx_size == queue_depth.
     */
    size_t cs = (eng->chunk_size > 0) ? eng->chunk_size : 4096;
    const size_t MAX_RECORD_BYTES = 1024UL * 1024;   /* 1 MB upper bound */
    int max_chunks_per_sample = static_cast<int>((MAX_RECORD_BYTES + cs - 1) / cs);
    if (max_chunks_per_sample < 1) max_chunks_per_sample = 1;
    int ctx_size = queue_depth * max_chunks_per_sample;
    int ret = io_setup(ctx_size, &p->ctx);
    if (ret < 0) {
        fprintf(stderr, "io_setup(%d) failed: %s\n", ctx_size, strerror(-ret));
        free(p); return ret;
    }

    p->events = static_cast<struct io_event *>(
                    malloc(static_cast<size_t>(queue_depth)
                           * sizeof(struct io_event)));
    p->n_inflight.store(0);

    eng->priv        = p;
    eng->queue_depth = queue_depth;
    return 0;
}

static int libaio_submit(io_engine_t *eng, io_req_t *reqs, int n)
{
    libaio_priv_t *p = static_cast<libaio_priv_t *>(eng->priv);
    size_t chunk_size = (eng->chunk_size > 0) ? eng->chunk_size : 4096;

    uint64_t submit_time = libaio_mono_ns();

    /*
     * Build the flat iocb pointer list across all samples × chunks.
     * Each tracker owns its iocbs vector so pointers remain valid until
     * the tracker is deleted after the last io_getevents() event.
     */
    std::vector<struct iocb *> iocb_ptrs;
    iocb_ptrs.reserve(static_cast<size_t>(n) * 64);

    for (int i = 0; i < n; i++) {
        size_t n_chunks = (reqs[i].length + chunk_size - 1) / chunk_size;
        reqs[i].submit_ns = submit_time;
        reqs[i].status    = 0;

        auto *tracker = new libaio_sample_tracker_t(
                                &reqs[i], static_cast<int>(n_chunks));

        for (size_t c = 0; c < n_chunks; c++) {
            struct iocb *cb = &tracker->iocbs[c];
            io_prep_pread(cb,
                          reqs[i].fd,
                          static_cast<char *>(reqs[i].buf) + c * chunk_size,
                          chunk_size,
                          static_cast<long long>(reqs[i].offset
                                                 + c * chunk_size));
            cb->data = tracker;
            iocb_ptrs.push_back(cb);
        }
    }

    int ret = io_submit(p->ctx,
                        static_cast<long>(iocb_ptrs.size()),
                        iocb_ptrs.data());
    if (ret < 0) {
        fprintf(stderr, "io_submit failed: %s\n", strerror(-ret));
        return ret;
    }

    p->n_inflight.fetch_add(n);  /* track samples, not chunks */
    return n;
}

static int libaio_reap(io_engine_t *eng, int min_completions, int timeout_us)
{
    libaio_priv_t *p = static_cast<libaio_priv_t *>(eng->priv);

    if (p->n_inflight.load() == 0 && min_completions == 0)
        return 0;

    int reaped_samples = 0;

    uint64_t deadline_ns = 0;
    if (timeout_us > 0)
        deadline_ns = libaio_mono_ns()
                    + static_cast<uint64_t>(timeout_us) * 1000ULL;

    /*
     * Loop until at least min_completions samples complete or timeout.
     * Each io_getevents() call returns chunk-level events, and a sample is
     * counted only when its last chunk's event arrives (chunks_left → 0).
     */
    while (reaped_samples < min_completions) {
        struct timespec ts_timeout;
        struct timespec *pts = nullptr;

        if (deadline_ns > 0) {
            uint64_t now = libaio_mono_ns();
            if (now >= deadline_ns) break;
            uint64_t remaining_ns = deadline_ns - now;
            ts_timeout.tv_sec  = static_cast<time_t>(
                                      remaining_ns / 1000000000ULL);
            ts_timeout.tv_nsec = static_cast<long>(
                                      remaining_ns % 1000000000ULL);
            pts = &ts_timeout;
        }

        int n = io_getevents(p->ctx, 1, p->max_events, p->events, pts);
        if (n < 0) {
            if (n != -EINTR)
                fprintf(stderr, "io_getevents: %s\n", strerror(-n));
            break;
        }
        if (n == 0) break;   /* timeout with no events */

        uint64_t now = libaio_mono_ns();

        for (int i = 0; i < n; i++) {
            auto *tracker = static_cast<libaio_sample_tracker_t *>(
                                p->events[i].data);
            if (!tracker) continue;

            long res = static_cast<long>(p->events[i].res);
            if (res < 0) {
                int expected = 0;
                tracker->error.compare_exchange_strong(
                    expected, static_cast<int>(res));
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
    }

    p->n_inflight.fetch_sub(reaped_samples);
    return reaped_samples;
}

static void libaio_destroy(io_engine_t *eng)
{
    libaio_priv_t *p = static_cast<libaio_priv_t *>(eng->priv);
    io_destroy(p->ctx);
    free(p->events);
    free(p);
}

static io_engine_ops_t libaio_ops = {
    .name    = "libaio",
    .init    = libaio_init,
    .submit  = libaio_submit,
    .reap    = libaio_reap,
    .destroy = libaio_destroy,
};

io_engine_t *engine_libaio_create(void)
{
    io_engine_t *e = static_cast<io_engine_t *>(calloc(1, sizeof(*e)));
    e->ops = &libaio_ops;
    return e;
}
