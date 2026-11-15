/*
 * engine_scissor_sr.cpp: single-ring scissor IO engine for AI training
 *
 * Replaces the per-file-ring + Loader/HostCache design of engine_scissor.cpp
 * with a simpler, higher effective-QD architecture.  Both engines coexist.
 * Select this one with --engine scissor_sr.
 *
 * ── Design ────────────────────────────────────────────────────────────────
 *
 *   One shared ring (scissor::LiburingReader):
 *     All files' segments are submitted to a single io_uring ring, giving
 *     the kernel one unified view of all pending IO.  Effective QD =
 *     queue_depth across all files, vs. N×1 (one segment per file at a time)
 *     in the old per-file-ring design.
 *
 *   Loader / HostCache bypassed:
 *     scissor::Loader / HostCache run a sequential batch pipeline per handle
 *     (enqueue → wait_all → enqueue …).  We drive the ring directly via
 *     scissor::Aggregator (for coalescing) + ring.enqueue_reads() /
 *     ring.wait_n().
 *
 *   Per-file independent aggregation:
 *     Aggregator::coalesce() is a pure function (no shared mutable state).
 *     It is called once per file per step: each file's offset list is
 *     coalesced independently with the same (gap, block) parameters derived
 *     by the Regressor from the representative file.
 *
 *   Admission control at queue_depth:
 *     At most min(queue_depth, SR_BUF_POOL_MAX) segments are in-flight
 *     simultaneously.  Each in-flight segment draws a pre-allocated,
 *     page-aligned buffer from a fixed pool (no per-segment syscall).
 *     As reap() harvests completions, buffers are returned to the pool
 *     and freed slots refilled from the pending-segment list via
 *     submit_available().
 *
 *   No threads:
 *     submit() aggregates all segments and enqueues the first batch.
 *     reap() harvests completions and enqueues more.  All ring access
 *     is from the bench.cpp main thread, so no locking is required.
 *
 *   Per-segment completion granularity:
 *     Each coalesced segment tracks the io_req_t indices it covers.
 *     Completions are reported as soon as each segment finishes, not
 *     when the whole file finishes (finer than the old per-file model).
 *
 * ── Worst-case overhead (N=16, M=4096, gap=0, chunk_size=163840 B) ────────
 *
 *   Segments: N×M = 65 536 (no coalescing).
 *   Rounds:   ceil(65 536 / 64) = 1 024  enqueue_reads() calls.
 *   Old design: 65 536 rounds (1 seg/file × N files, per-file sequential).
 *   Peak buffer: queue_depth × chunk_size = 64 × 163 840 = 10 MB.
 *   Old design:  N × chunk_size = 16 × 163 840 = 2.5 MB.
 *
 *   The extra 7.5 MB is the price of 64× higher effective QD.
 *   On any HPC node this is negligible.
 */

#include "io_engine.h"

#include <scissor/aggregator.hpp>
#include <scissor/readers.hpp>
#include <scissor/regressor.hpp>
#include <scissor/types.hpp>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <unordered_map>
#include <vector>

/* ── Timing helper ─────────────────────────────────────────────────── */

static inline uint64_t sr_mono_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL
         + static_cast<uint64_t>(ts.tv_nsec);
}

/* ── Recycling buffer pool entry ───────────────────────────────────── */
/*
 * The pool stores freed buffers as (ptr, size) pairs.  On next
 * submission we scan for an entry with size >= needed. If found we
 * reuse it, otherwise we posix_memalign a new one.  After the first
 * epoch all allocations are amortised: posix_memalign is only called
 * when a new segment size is encountered, which is rare in practice
 * because access patterns repeat across steps.
 *
 * This avoids two bugs in the previous fixed-size pool design:
 *  1. buf_slot_size = chunk_size×17 was too small for segments that
 *     span many coalesced data chunks (could be >> 17 chunks), causing
 *     a heap overflow and SIGABRT.
 *  2. Pre-allocating 512 × 2.7 MB = 1.4 GB at init was wasteful and
 *     risked hitting mlock / virtual-memory limits on HPC systems.
 */
struct sr_poolbuf_t {
    uint8_t *ptr;
    size_t   size;
};

/* ── Per-segment in-flight state ───────────────────────────────────── */

struct sr_inflight_t {
    uint8_t         *buf;
    size_t           buf_size;  /* needed to return correctly-sized entry   */
    std::vector<int>  req_idxs; /* pending_reqs[] indices covered by seg   */
};

/* ── Pre-aggregated segment, waiting for a free admission slot ─────── */

struct sr_pending_seg_t {
    scissor::Segment    seg;       /* fd, offset, size set. id/buf = 0/null */
    std::vector<int>    req_idxs;  /* same as sr_inflight_t::req_idxs       */
};

/* ── Private engine state ──────────────────────────────────────────── */

struct scissor_sr_priv_t {
    /* ── Persistent across steps ──────────────────────────────────── */
    scissor::LiburingReader               ring;
    scissor::Aggregator                   aggregator;   /* stateless */
    scissor::Regressor                    regressor;
    std::unordered_map<int, std::string>  fd_to_path;
    int                                   queue_depth;

    /* ── Per-step state (reset at the top of each submit()) ────────── */
    io_req_t *pending_reqs;
    int       pending_n;

    std::vector<sr_pending_seg_t>          all_segs;
    size_t                                 submit_head;  /* next to enqueue */
    int                                    n_inflight;

    /* in-flight: segment_id → {buf, req_idxs} */
    std::unordered_map<size_t, sr_inflight_t> inflight;

    /* ── Persistent across steps (monotonically increasing) ────────── */
    size_t seg_id_counter;   /* unique across ring's lifetime, never resets */

    /* ── Recycling buffer pool ─────────────────────────────────────── */
    std::vector<sr_poolbuf_t> free_bufs;  /* idle buffers ready for reuse  */

    scissor_sr_priv_t()
        : regressor(scissor::FileSrc::SSD)
        , queue_depth(64)
        , pending_reqs(nullptr)
        , pending_n(0)
        , submit_head(0)
        , n_inflight(0)
        , seg_id_counter(0)
    {}
};

/* ── Enqueue up to (queue_depth − n_inflight) pending segments ─────── */
/*
 * Called from both submit() (initial batch) and reap() (refill after
 * completions free slots).  One ring.enqueue_reads() call per invocation
 * → one io_uring_submit() syscall for however many segments fit.
 */
static void sr_submit_available(scissor_sr_priv_t *priv)
{
    int free_slots = priv->queue_depth - priv->n_inflight;
    if (free_slots <= 0 || priv->submit_head >= priv->all_segs.size())
        return;

    std::vector<scissor::Segment> to_submit;
    to_submit.reserve(static_cast<size_t>(free_slots));

    while (free_slots > 0 && priv->submit_head < priv->all_segs.size()) {
        sr_pending_seg_t &ps = priv->all_segs[priv->submit_head];
        size_t need = ps.seg.size;

        /*
         * Find a recycled buffer of sufficient size (swap-remove for O(1)
         * removal), or posix_memalign a fresh one.  After the first epoch
         * all segment sizes are warm in the pool and new allocations are rare.
         */
        uint8_t *buf  = nullptr;
        for (size_t k = 0; k < priv->free_bufs.size(); k++) {
            if (priv->free_bufs[k].size >= need) {
                buf = priv->free_bufs[k].ptr;
                priv->free_bufs[k] = priv->free_bufs.back();
                priv->free_bufs.pop_back();
                break;
            }
        }
        if (!buf) {
            void *p = nullptr;
            if (posix_memalign(&p, 4096, need) != 0) {
                fprintf(stderr, "[engine_scissor_sr] posix_memalign(%zu): %s\n",
                        need, strerror(errno));
                break;   /* OOM: retry on next reap() */
            }
            buf = static_cast<uint8_t *>(p);
        }

        ps.seg.id     = priv->seg_id_counter++;
        ps.seg.buffer = buf;

        priv->inflight.emplace(ps.seg.id,
            sr_inflight_t{buf, need, std::move(ps.req_idxs)});

        to_submit.push_back(ps.seg);
        ++priv->submit_head;
        ++priv->n_inflight;
        --free_slots;
    }

    if (!to_submit.empty())
        priv->ring.enqueue_reads(to_submit);
}

/* ── Engine operations ─────────────────────────────────────────────── */

static int scissor_sr_init(io_engine_t *eng, int queue_depth)
{
    auto *priv        = new scissor_sr_priv_t();
    priv->queue_depth = queue_depth;
    /* free_bufs starts empty. Buffers are allocated lazily on first use
     * and recycled across steps, so no pre-allocation is needed. */
    eng->priv        = priv;
    eng->queue_depth = queue_depth;
    return 0;
}

static int scissor_sr_submit(io_engine_t *eng, io_req_t *reqs, int n)
{
    auto *priv = static_cast<scissor_sr_priv_t *>(eng->priv);

    if (priv->fd_to_path.empty()) {
        fprintf(stderr, "[engine_scissor_sr] submit called before set_paths\n");
        return -1;
    }

    if (n == 0)
        return 0;

    /* ── Safety drain: previous step must be fully reaped ─────────── */
    if (priv->n_inflight != 0) {
        fprintf(stderr,
            "[engine_scissor_sr] WARNING: %d segments still in flight "
            "at start of new step, draining\n", priv->n_inflight);
        priv->ring.wait_all();
        for (auto &[id, info] : priv->inflight)
            priv->free_bufs.push_back({info.buf, info.buf_size});
        priv->inflight.clear();
        priv->n_inflight = 0;
    }

    /* ── Reset per-step state ──────────────────────────────────────── */
    priv->all_segs.clear();
    priv->submit_head = 0;

    size_t chunk_size = (eng->chunk_size > 0) ? eng->chunk_size : 4096;

    uint64_t submit_time = sr_mono_ns();
    for (int i = 0; i < n; i++) {
        reqs[i].submit_ns = submit_time;
        reqs[i].status    = 0;
    }

    /* ── Build per-file chunk-index lists and chunk→req mapping ────── */
    /*
     * fd_to_offsets[fd]       : sorted, deduped chunk indices for this file.
     * fd_chunk_to_reqs[fd][c] : req indices whose read covers chunk c.
     *
     * With chunk_size = record_size, each req spans exactly one chunk.
     * Two reqs at the same chunk (duplicate sample in trace) are handled
     * by the vector<int>: both are marked done on that segment's completion.
     */
    std::unordered_map<int, std::vector<size_t>>
        fd_to_offsets;
    std::unordered_map<int,
        std::unordered_map<size_t, std::vector<int>>>
        fd_chunk_to_reqs;

    for (int i = 0; i < n; i++) {
        int      fd     = reqs[i].fd;
        uint64_t offset = reqs[i].offset;
        uint32_t length = reqs[i].length;

        if (priv->fd_to_path.find(fd) == priv->fd_to_path.end()) {
            fprintf(stderr, "[engine_scissor_sr] unknown fd %d\n", fd);
            return -1;
        }

        /* All chunk indices spanned by [offset, offset+length) */
        size_t first_chunk = static_cast<size_t>(offset / chunk_size);
        size_t last_chunk  = static_cast<size_t>(
                                 (offset + length - 1) / chunk_size + 1);

        for (size_t c = first_chunk; c < last_chunk; c++) {
            fd_to_offsets[fd].push_back(c);
            fd_chunk_to_reqs[fd][c].push_back(i);
        }
    }

    /* Sort and deduplicate chunk indices per file */
    for (auto &[fd, offsets] : fd_to_offsets) {
        std::sort(offsets.begin(), offsets.end());
        offsets.erase(std::unique(offsets.begin(), offsets.end()),
                      offsets.end());
    }

    /* ── Regressor: one call on the representative file's offsets ─── */
    int n_cores = std::max(2, eng->queue_depth);
    int opt_gap;

    if (eng->force_gap >= 0) {
        opt_gap = eng->force_gap;
    } else {
        const auto &rep = fd_to_offsets.begin()->second;
        opt_gap = priv->regressor.optimize(rep, chunk_size, n_cores).first;
    }

    /*
     * used_chks_per_read controls how many chunks are grouped into one
     * coalesced read submission.
     *   1 (default): one submission per contiguous run, for the finest completion
     *     granularity. This is correct because we do not pipeline through HostCache.
     *   >1 (force_batch_bytes set): group multiple chunks per submission to
     *     study batch-size sensitivity without Regressor influence.
     */
    const uint32_t used_chks = (eng->force_batch_bytes > 0)
        ? static_cast<uint32_t>(eng->force_batch_bytes / chunk_size)
        : 1u;

    /* ── Per-file independent aggregation ─────────────────────────── */
    /*
     * Aggregator::coalesce() is pure and reentrant.  Each file gets
     * its own coalescing pass with the same (gap, block) parameters.
     * The resulting segments carry byte offsets (chunk_idx × chunk_size)
     * and byte sizes (n_chunks × chunk_size).
     */
    for (auto &[fd, offsets] : fd_to_offsets) {
        scissor::AggregatorResult agg =
            priv->aggregator.coalesce(offsets, chunk_size,
                                      static_cast<uint32_t>(opt_gap),
                                      used_chks);

        const auto &cmap = fd_chunk_to_reqs.at(fd);

        for (const scissor::Segment &s : agg.segments) {
            sr_pending_seg_t ps;
            ps.seg.id     = 0;         /* assigned at submission time */
            ps.seg.fd     = fd;
            ps.seg.offset = s.offset;  /* byte offset in file         */
            ps.seg.size   = s.size;    /* byte count                  */
            ps.seg.buffer = nullptr;   /* assigned at submission time */

            /*
             * Map segment byte range → req indices.
             *
             * The segment covers chunk indices [c_start, c_start + n_chunks).
             * For each chunk in that range that appears in cmap, collect the
             * corresponding req indices.  Gap-fill chunks (not in cmap) are
             * skipped: their bytes are read and discarded.
             *
             * Because Aggregator only emits segments that contain at least
             * one originally requested chunk, ps.req_idxs is never empty.
             */
            size_t c_start  = s.offset / chunk_size;
            size_t n_chunks = s.size   / chunk_size;
            for (size_t c = c_start; c < c_start + n_chunks; c++) {
                auto it = cmap.find(c);
                if (it != cmap.end()) {
                    for (int idx : it->second)
                        ps.req_idxs.push_back(idx);
                }
            }
            /*
             * Deduplicate req_idxs.  A req whose length > chunk_size spans
             * multiple consecutive chunks, and all of those chunks land in the
             * same coalesced segment, so the req_idx is pushed once per
             * chunk.  Without dedup, reap() would mark it complete multiple
             * times and decrement pending_n past zero.
             * With chunk_size == record_size (the intended configuration)
             * each req spans exactly one chunk and no duplicates arise, but
             * the dedup is cheap insurance.
             */
            std::sort(ps.req_idxs.begin(), ps.req_idxs.end());
            ps.req_idxs.erase(
                std::unique(ps.req_idxs.begin(), ps.req_idxs.end()),
                ps.req_idxs.end());

            priv->all_segs.push_back(std::move(ps));
        }
    }

    priv->pending_reqs = reqs;
    priv->pending_n    = n;

    /* Enqueue the first batch (up to queue_depth segments) */
    sr_submit_available(priv);

    return n;
}

static int scissor_sr_reap(io_engine_t *eng, int min_completions,
                            int timeout_us)
{
    (void)min_completions; /* we drain queue_depth segments per call   */
    (void)timeout_us;      /* ring.wait_n() has no timeout, bench retries */
    auto *priv = static_cast<scissor_sr_priv_t *>(eng->priv);

    if (priv->pending_n == 0)
        return 0;

    /*
     * Wait for one full batch of completions.
     * nwait = min(n_inflight, queue_depth) so we never ask for more
     * IDs than what the ring currently has in flight.
     * Choosing queue_depth (not min_completions) as the ceiling
     * amortises the wait_n() call overhead: we drain a full batch
     * of queue_depth segments per call rather than one at a time.
     */
    int nwait_int = std::min(priv->n_inflight, priv->queue_depth);
    if (nwait_int <= 0)
        return 0;
    size_t nwait = static_cast<size_t>(nwait_int);

    std::vector<size_t> done_ids = priv->ring.wait_n(nwait);

    uint64_t now        = sr_mono_ns();
    int      newly_done = 0;

    for (size_t seg_id : done_ids) {
        auto it = priv->inflight.find(seg_id);
        if (it == priv->inflight.end()) continue;

        sr_inflight_t &info = it->second;

        for (int idx : info.req_idxs) {
            priv->pending_reqs[idx].complete_ns = now;
            priv->pending_reqs[idx].status      = 1;
            ++newly_done;
        }

        priv->free_bufs.push_back({info.buf, info.buf_size});
        priv->inflight.erase(it);
        --priv->n_inflight;
    }

    /* Refill freed slots */
    sr_submit_available(priv);

    priv->pending_n -= newly_done;
    if (priv->pending_n <= 0) {
        priv->pending_reqs = nullptr;
        priv->pending_n    = 0;   /* floor at zero. Dedup prevents underflow
                                     but guard anyway for robustness         */
    }

    return newly_done;
}

static void scissor_sr_destroy(io_engine_t *eng)
{
    auto *priv = static_cast<scissor_sr_priv_t *>(eng->priv);
    if (priv->n_inflight > 0) {
        priv->ring.wait_all();
        for (auto &[id, info] : priv->inflight)
            priv->free_bufs.push_back({info.buf, info.buf_size});
        priv->inflight.clear();
    }
    for (auto &b : priv->free_bufs)
        free(b.ptr);
    delete priv;
    eng->priv = nullptr;
}

static io_engine_ops_t scissor_sr_ops = {
    .name    = "scissor_sr",
    .init    = scissor_sr_init,
    .submit  = scissor_sr_submit,
    .reap    = scissor_sr_reap,
    .destroy = scissor_sr_destroy,
};

/* ── Factory and path-registration ────────────────────────────────── */

io_engine_t *engine_scissor_sr_create(void)
{
    io_engine_t *e = static_cast<io_engine_t *>(calloc(1, sizeof(*e)));
    e->ops = &scissor_sr_ops;
    return e;
}

void engine_scissor_sr_set_paths(io_engine_t *eng,
                                  const char **paths,
                                  const int   *fds,
                                  int          n)
{
    auto *priv = static_cast<scissor_sr_priv_t *>(eng->priv);
    priv->fd_to_path.clear();
    for (int i = 0; i < n; i++)
        priv->fd_to_path[fds[i]] = paths[i];
}
