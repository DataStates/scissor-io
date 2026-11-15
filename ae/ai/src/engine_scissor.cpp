/*
 * engine_scissor.cpp: scissor::Loader IO engine for AI training workloads
 *
 * Wraps scissor::Loader (backed by scissor::LiburingReader) as a
 * bench.cpp-compatible io_engine_t.
 *
 * Design:
 *
 *   Per-file independent loaders:
 *     Each file gets its own (reader, loader) pair and
 *     therefore its own io_uring ring.  This eliminates the shared
 *     last_retrieving_id_ bottleneck in Loader and allows all files' IO
 *     pipelines to run in parallel, matching how pread/libaio/io_uring
 *     submit all requests across all files simultaneously.
 *
 *   Per-file producer threads:
 *     submit() spawns one producer thread per file.  Each thread drains its
 *     file's loader handle (loader.next() loop) and posts a file_completion_t
 *     to the shared queue when done.  All threads run concurrently so IO
 *     across files overlaps, not serialises.
 *
 *   Sorted offsets (required by Algorithm 1 and Regressor::optimize()):
 *     Chunk indices are sorted ascending before being passed to the Loader.
 *
 *   Regressor engaged (Algorithm 2):
 *     A single shared Regressor is constructed once at engine-init time.
 *     optimize() is called once per step (pure CPU, read-only) before any
 *     producer thread is launched, so no locking is required.
 *
 *   Consumer interleave:
 *     reap() returns one file's completions per call.  bench.cpp interleaves
 *     compute between calls, matching the state-diff / Kokkos usage where
 *     scissor is a pure producer that does not control the consumer schedule.
 *
 *   Data copy not needed:
 *     bench.cpp reads only submit_ns / complete_ns / length / buf_idx
 *     after reap(). It never dereferences req->buf.
 */

#include "io_engine.h"

#include <scissor/loader.hpp>
#include <scissor/readers.hpp>
#include <scissor/types.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

/* ── Per-file IO stack, split into persistent and per-step parts ─── */

/*
 * file_reader_t: created once per file in set_paths(), lives for the
 * engine lifetime.  Holds only the io_uring ring, which is expensive
 * to set up but safe to keep open across steps once the Loader that
 * used it has been closed and destroyed.
 */
struct file_reader_t {
    /* Ring size and completion batch size: LiburingReader defaults. */
    scissor::LiburingReader reader{scissor::LiburingReader::DEFAULT_RING_SIZE,
                                   scissor::LiburingReader::DEFAULT_BATCH_SIZE};

    file_reader_t() = default;
    file_reader_t(const file_reader_t &) = delete;
    file_reader_t &operator=(const file_reader_t &) = delete;
};

/*
 * file_step_t: created fresh at the start of every submit() call and
 * destroyed (via fd_to_step.clear()) at the start of the next submit()
 * after the previous producer thread has been joined.
 *
 * scissor::Loader accumulates per-handle state that is not fully reset
 * by close(). Rebuilding the Loader each step avoids stale state that
 * causes loader.next() to block indefinitely on step 2+.
 *
 * The loader holds a reference to the reader so file_step_t must
 * not be moved after construction. Use unique_ptr.
 */
struct file_step_t {
    scissor::Loader loader;

    explicit file_step_t(scissor::LiburingReader &reader)
        : loader(reader, scissor::Config{false, 0, 64})
    {}

    file_step_t(const file_step_t &) = delete;
    file_step_t &operator=(const file_step_t &) = delete;
};

/* ── One entry pushed to the completion queue per file ───────────── */

struct file_completion_t {
    int      fd;
    uint64_t complete_ns;
    int      n_reqs;
};

/* ── Private engine state ────────────────────────────────────────── */

struct scissor_priv_t {
    /*
     * Regressor is constructed once so models.conf is read only once.
     * optimize() is called from the main thread before any producer
     * thread is launched, so there is no concurrent access.
     */
    scissor::Regressor regressor;

    /* fd → path (populated by engine_scissor_set_paths) */
    std::unordered_map<int, std::string> fd_to_path;

    /* fd → persistent io_uring ring, created once in set_paths() */
    std::unordered_map<int, std::unique_ptr<file_reader_t>> fd_to_reader;

    /* fd → per-step loader stack, rebuilt at the top of every submit() */
    std::unordered_map<int, std::unique_ptr<file_step_t>> fd_to_step;

    /* Per-step tracking */
    std::vector<scissor::LoadHandle> active_handles;   /* parallel to handle_fds */
    std::vector<int>                 handle_fds;

    std::unordered_map<int, std::vector<int>> fd_to_req_indices;

    io_req_t *pending_reqs = nullptr;
    int       pending_n    = 0;

    /* Outer producer thread (joins per-file threads, signals done) */
    std::thread             producer_thread;
    std::mutex              completion_mutex;
    std::condition_variable completion_cv;
    std::deque<file_completion_t> completion_queue;
    bool                    producer_done = true;

    scissor_priv_t()
        : regressor(scissor::FileSrc::SSD)   /* reads models.conf once */
    {}
};

/* ── Timing helper ───────────────────────────────────────────────── */

static inline uint64_t mono_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL
         + static_cast<uint64_t>(ts.tv_nsec);
}

/* ── Engine operations ───────────────────────────────────────────── */

static int scissor_init(io_engine_t *eng, int queue_depth)
{
    eng->priv        = new scissor_priv_t();
    eng->queue_depth = queue_depth;
    return 0;
}

static int scissor_submit(io_engine_t *eng, io_req_t *reqs, int n)
{
    scissor_priv_t *priv = static_cast<scissor_priv_t *>(eng->priv);

    if (priv->fd_to_path.empty()) {
        fprintf(stderr, "[engine_scissor] submit called before set_paths\n");
        return -1;
    }

    /* Join previous producer thread before starting a new step */
    if (priv->producer_thread.joinable())
        priv->producer_thread.join();

    /*
     * Rebuild per-step loader stacks, reusing the persistent io_uring
     * rings.  scissor::Loader accumulates handle state that is not fully
     * reset by close(). Recreating it each step avoids the stale state
     * that causes loader.next() to block forever on step 2+.
     * The reader (io_uring ring) is expensive to create and is
     * safe to keep open: only the Loader is rebuilt.
     */
    priv->fd_to_step.clear();
    for (auto &[fd, reader] : priv->fd_to_reader)
        priv->fd_to_step[fd] = std::make_unique<file_step_t>(reader->reader);

    /* Record submission timestamp for every request */
    uint64_t submit_time = mono_ns();
    for (int i = 0; i < n; i++) {
        reqs[i].submit_ns = submit_time;
        reqs[i].status    = 0;
    }

    /*
     * Build one FileRequest per unique fd and record which request
     * indices belong to each fd (for completion marking in reap()).
     * Each sample's byte range [offset, offset+length) is covered by
     * the chunk indices [first_chunk, last_chunk).
     */
    size_t chunk_size = (eng->chunk_size > 0) ? eng->chunk_size : 4096;
    std::unordered_map<int, scissor::FileRequest> fd_to_req;
    priv->fd_to_req_indices.clear();

    for (int i = 0; i < n; i++) {
        int      fd     = reqs[i].fd;
        uint64_t offset = reqs[i].offset;
        uint32_t length = reqs[i].length;

        auto path_it = priv->fd_to_path.find(fd);
        if (path_it == priv->fd_to_path.end()) {
            fprintf(stderr, "[engine_scissor] unknown fd %d\n", fd);
            return -1;
        }

        scissor::FileRequest &fr = fd_to_req[fd];
        if (fr.path.empty())
            fr.path = path_it->second;

        size_t first_chunk = static_cast<size_t>(offset / chunk_size);
        size_t last_chunk  = static_cast<size_t>(
                                 (offset + length - 1 + chunk_size - 1) / chunk_size);
        for (size_t c = first_chunk; c < last_chunk; c++)
            fr.offsets.push_back(c);

        priv->fd_to_req_indices[fd].push_back(i);
    }

    /*
     * Sort then deduplicate chunk indices.
     * Two samples whose byte ranges overlap on the same chunk would otherwise
     * push that chunk index multiple times, causing the Aggregator's
     * wasted_bytes computation (segment_bytes - useful_chunk_bytes) to go
     * negative and wrap as uint64_t.
     */
    for (auto &[fd, fr] : fd_to_req) {
        std::sort(fr.offsets.begin(), fr.offsets.end());
        fr.offsets.erase(std::unique(fr.offsets.begin(), fr.offsets.end()),
                         fr.offsets.end());
    }

    /*
     * Run the Regressor once on the first file's offsets.  All files
     * share the same workload parameters so one optimize() call gives
     * (opt_gap, opt_block) that applies to every file.
     */
    int n_cores = std::max(2, eng->queue_depth);
    const std::vector<size_t> &representative_offsets =
        fd_to_req.begin()->second.offsets;
    /*
     * Gap / block-size selection:
     *
     *   force_gap < 0, force_batch_bytes == 0 (auto): run the Regressor to
     *     jointly find (gap*, block*) that maximises predicted throughput.
     *
     *   force_gap >= 0 (gap sweep): fix the gap. block_size comes from
     *     force_batch_bytes if set, otherwise falls back to seg_size (used_chks=1).
     *
     *   force_batch_bytes > 0 (batch sweep): override block_size regardless of
     *     gap selection, allowing independent batch-size sensitivity analysis.
     */
    int    opt_gap;
    size_t opt_block;
    if (eng->force_gap >= 0) {
        opt_gap   = eng->force_gap;
        opt_block = eng->force_batch_bytes;  /* 0 falls back to seg_size in Loader */
    } else {
        auto [g, b] = priv->regressor.optimize(
                          representative_offsets, chunk_size, n_cores);
        opt_gap   = g;
        opt_block = (eng->force_batch_bytes > 0) ? eng->force_batch_bytes : b;
    }

    /*
     * Issue one load() per file using that file's own loader instance.
     * Each loader has its own io_uring ring, so all files' first batches
     * are submitted to the kernel simultaneously here.
     */
    priv->active_handles.clear();
    priv->handle_fds.clear();
    for (auto &[fd, fr] : fd_to_req) {
        file_step_t &step = *priv->fd_to_step.at(fd);
        scissor::LoadHandle load_handle = step.loader.load(
            std::vector<scissor::FileRequest>{fr},
            chunk_size,
            scissor::TransferType::FileToHost,
            scissor::FileSrc::SSD,
            1,                                   /* nthreads=1: skip internal Regressor */
            static_cast<uint32_t>(opt_gap),
            opt_block);
        priv->active_handles.push_back(load_handle);
        priv->handle_fds.push_back(fd);
    }

    priv->pending_reqs = reqs;
    priv->pending_n    = n;

    /*
     * Launch one producer thread per file.  All threads drain their
     * respective loader handles concurrently: each loader has its own
     * io_uring ring so there is no shared kernel state.  The outer
     * producer thread joins all per-file threads then signals done.
     */
    {
        std::lock_guard<std::mutex> lock(priv->completion_mutex);
        priv->completion_queue.clear();
        priv->producer_done = false;
    }

    priv->producer_thread = std::thread([priv]() {
        std::vector<std::thread> file_threads;
        file_threads.reserve(priv->active_handles.size());

        for (size_t i = 0; i < priv->active_handles.size(); ++i) {
            file_threads.emplace_back([priv, i]() {
                int                  fd          = priv->handle_fds[i];
                scissor::LoadHandle  load_handle = priv->active_handles[i];
                scissor::Loader     &loader      = priv->fd_to_step.at(fd)->loader;

                while (loader.next(load_handle,
                                   scissor::TransferType::FileToHost) != nullptr)
                    ;
                loader.close(load_handle);

                file_completion_t comp;
                comp.fd          = fd;
                comp.complete_ns = mono_ns();
                comp.n_reqs      = static_cast<int>(
                                       priv->fd_to_req_indices.at(fd).size());

                {
                    std::lock_guard<std::mutex> lock(priv->completion_mutex);
                    priv->completion_queue.push_back(comp);
                }
                priv->completion_cv.notify_one();
            });
        }

        for (auto &t : file_threads)
            t.join();

        priv->active_handles.clear();

        {
            std::lock_guard<std::mutex> lock(priv->completion_mutex);
            priv->producer_done = true;
        }
        priv->completion_cv.notify_all();
    });

    return n;
}

static int scissor_reap(io_engine_t *eng, int min_completions, int timeout_us)
{
    scissor_priv_t *priv = static_cast<scissor_priv_t *>(eng->priv);
    (void)min_completions;  /* one file completion returned per call */

    if (priv->pending_n == 0)
        return 0;

    /*
     * Wait for any per-file thread to post a completion.
     * Because all file threads run in parallel the first completion
     * typically arrives after max(file_io_times) / n_files rather than
     * their sum.
     */
    std::unique_lock<std::mutex> lock(priv->completion_mutex);

    if (timeout_us > 0) {
        auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::microseconds(timeout_us);
        priv->completion_cv.wait_until(lock, deadline, [priv]() {
            return !priv->completion_queue.empty() || priv->producer_done;
        });
    } else {
        priv->completion_cv.wait(lock, [priv]() {
            return !priv->completion_queue.empty() || priv->producer_done;
        });
    }

    if (priv->completion_queue.empty())
        return 0;

    file_completion_t comp = priv->completion_queue.front();
    priv->completion_queue.pop_front();
    lock.unlock();

    /* Mark all requests for this file complete */
    auto it = priv->fd_to_req_indices.find(comp.fd);
    if (it != priv->fd_to_req_indices.end()) {
        for (int idx : it->second) {
            priv->pending_reqs[idx].complete_ns = comp.complete_ns;
            priv->pending_reqs[idx].status      = 1;
        }
    }

    priv->pending_n -= comp.n_reqs;
    if (priv->pending_n <= 0) {
        priv->pending_reqs = nullptr;
        priv->pending_n    = 0;
    }

    return comp.n_reqs;
}

static void scissor_destroy(io_engine_t *eng)
{
    scissor_priv_t *priv = static_cast<scissor_priv_t *>(eng->priv);
    if (priv->producer_thread.joinable())
        priv->producer_thread.join();
    delete priv;
    eng->priv = nullptr;
}

static io_engine_ops_t scissor_ops = {
    .name    = "scissor",
    .init    = scissor_init,
    .submit  = scissor_submit,
    .reap    = scissor_reap,
    .destroy = scissor_destroy,
};

/* ── Factory and path-registration (called from bench.cpp) ──────── */

io_engine_t *engine_scissor_create(void)
{
    io_engine_t *engine = static_cast<io_engine_t *>(calloc(1, sizeof(*engine)));
    engine->ops = &scissor_ops;
    return engine;
}

void engine_scissor_set_paths(io_engine_t *eng,
                              const char **paths,
                              const int   *fds,
                              int          n)
{
    scissor_priv_t *priv = static_cast<scissor_priv_t *>(eng->priv);
    priv->fd_to_path.clear();
    priv->fd_to_reader.clear();
    priv->fd_to_step.clear();
    for (int i = 0; i < n; i++) {
        priv->fd_to_path[fds[i]]   = paths[i];
        priv->fd_to_reader[fds[i]] = std::make_unique<file_reader_t>();
    }
}
