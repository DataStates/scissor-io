/*
 * engine_mmap.cpp: mmap engine with POSIX thread pool
 *
 * Each file is mapped into virtual address space once (MAP_SHARED, whole file)
 * on first access and kept open for the engine lifetime.  Worker threads
 * serve requests via memcpy from the mapped region, without any per-request
 * syscall beyond the page-fault IO that the kernel issues on demand.
 * Each mapping is advised with madvise(MADV_RANDOM), disabling kernel
 * readahead around faulting pages.
 *
 * O_DIRECT is not used (mmap is incompatible with O_DIRECT), so the kernel page
 * cache is the IO path.  Drop page caches between runs with:
 *   sync; echo 3 > /proc/sys/vm/drop_caches
 * to ensure cold measurements.
 */
#include "io_engine.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <atomic>

#define MMAP_MAX_THREADS 128
#define MMAP_MAX_QUEUE   4096
#define MMAP_MAX_FDS     4096   /* generous upper bound for open fd values */

typedef struct {
    io_req_t        *req;
    std::atomic<int> done;
} mmap_work_item_t;

typedef struct {
    /* Thread pool */
    pthread_t        threads[MMAP_MAX_THREADS];
    int              n_threads;

    /* Work queue (ring buffer) */
    mmap_work_item_t       *queue[MMAP_MAX_QUEUE];
    std::atomic<int>        q_head;
    std::atomic<int>        q_tail;
    sem_t                   work_sem;
    sem_t                   free_sem;
    std::atomic<int>        shutdown;
    std::atomic<int>        n_inflight;

    /*
     * Monotonically-increasing completion counters for the current batch.
     *
     * n_completed:  incremented by each worker on finishing a request,
     *                never decremented, so it remains accurate even when
     *                workers finish before reap() is called.
     * n_reaped_out: total count already returned by reap() this batch,
     *                advanced inside reap() so repeated calls return the
     *                correct delta rather than the cumulative total.
     *
     * Both are reset to 0 in mmap_submit() at the start of each batch.
     */
    std::atomic<int>        n_completed;
    std::atomic<int>        n_reaped_out;

    /* Per-request item array for the current batch */
    mmap_work_item_t       *pending_items;

    /* Per-fd whole-file mmap cache (protected by mmap_mutex) */
    void            *fd_base[MMAP_MAX_FDS];
    size_t           fd_size[MMAP_MAX_FDS];
    bool             fd_mapped[MMAP_MAX_FDS];
    pthread_mutex_t  mmap_mutex;
} mmap_priv_t;

/* ── Per-fd mmap helper (called from worker, mutex held externally) ── */
static int ensure_mapped(mmap_priv_t *p, int fd)
{
    if (fd < 0 || fd >= MMAP_MAX_FDS) return -EBADF;
    if (p->fd_mapped[fd]) return 0;

    struct stat st;
    if (fstat(fd, &st) < 0) return -errno;

    void *base = mmap(nullptr, (size_t)st.st_size,
                      PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) return -errno;

    if (madvise(base, (size_t)st.st_size, MADV_RANDOM) != 0) {
        int err = errno;
        munmap(base, (size_t)st.st_size);
        return -err;
    }

    p->fd_base[fd]   = base;
    p->fd_size[fd]   = (size_t)st.st_size;
    p->fd_mapped[fd] = true;
    return 0;
}

/* ── Worker thread ───────────────────────────────────────────────── */
static void *mmap_worker_thread(void *arg)
{
    mmap_priv_t *p = static_cast<mmap_priv_t *>(arg);

    while (1) {
        sem_wait(&p->work_sem);
        if (p->shutdown.load()) break;

        int tail = p->q_tail.fetch_add(1) % MMAP_MAX_QUEUE;
        mmap_work_item_t *item = p->queue[tail];
        sem_post(&p->free_sem);

        if (!item || !item->req) continue;

        io_req_t *req = item->req;

        /* Ensure the file is mapped (lazy, mutex-protected init) */
        pthread_mutex_lock(&p->mmap_mutex);
        int rc = ensure_mapped(p, req->fd);
        void  *base    = (rc == 0) ? p->fd_base[req->fd] : nullptr;
        size_t mapsize = (rc == 0) ? p->fd_size[req->fd] : 0;
        pthread_mutex_unlock(&p->mmap_mutex);

        if (base == nullptr || req->offset + req->length > mapsize) {
            req->status = (rc != 0) ? rc : -EIO;
        } else {
            memcpy(req->buf,
                   static_cast<const char *>(base) + req->offset,
                   req->length);
            req->status = 1;
        }

        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        req->complete_ns = static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL
                         + static_cast<uint64_t>(ts.tv_nsec);

        item->done.store(1);
        p->n_inflight.fetch_sub(1);
        p->n_completed.fetch_add(1);
    }
    return nullptr;
}

/* ── Engine operations ───────────────────────────────────────────── */
static int mmap_init(io_engine_t *eng, int queue_depth)
{
    mmap_priv_t *p = static_cast<mmap_priv_t *>(calloc(1, sizeof(*p)));

    p->n_threads = (queue_depth < MMAP_MAX_THREADS)
                 ? queue_depth : MMAP_MAX_THREADS;

    sem_init(&p->work_sem, 0, 0);
    sem_init(&p->free_sem, 0, MMAP_MAX_QUEUE);
    pthread_mutex_init(&p->mmap_mutex, nullptr);
    p->shutdown.store(0);
    p->n_inflight.store(0);
    p->n_completed.store(0);
    p->n_reaped_out.store(0);
    p->q_head.store(0);
    p->q_tail.store(0);

    memset(p->fd_mapped, 0, sizeof(p->fd_mapped));

    for (int i = 0; i < p->n_threads; i++)
        pthread_create(&p->threads[i], nullptr, mmap_worker_thread, p);

    eng->priv        = p;
    eng->queue_depth = p->n_threads;
    return 0;
}

static int mmap_submit(io_engine_t *eng, io_req_t *reqs, int n)
{
    mmap_priv_t *p = static_cast<mmap_priv_t *>(eng->priv);

    mmap_work_item_t *items = static_cast<mmap_work_item_t *>(
        malloc(n * sizeof(mmap_work_item_t)));
    if (!items) return -ENOMEM;

    /*
     * Reset completion counters and record inflight count BEFORE enqueuing
     * any work.  Workers start executing as soon as sem_post(&work_sem) is
     * called inside the loop below. Resetting AFTER the loop loses any
     * completions that occurred during submission, causing mmap_reap to spin
     * for 30 s waiting for items that already finished.
     */
    p->pending_items = items;
    p->n_completed.store(0);
    p->n_reaped_out.store(0);
    p->n_inflight.store(n);

    for (int i = 0; i < n; i++) {
        items[i].req = &reqs[i];
        items[i].done.store(0);

        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        reqs[i].submit_ns = static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL
                          + static_cast<uint64_t>(ts.tv_nsec);
        reqs[i].status    = 0;

        sem_wait(&p->free_sem);
        int head = p->q_head.fetch_add(1) % MMAP_MAX_QUEUE;
        p->queue[head] = &items[i];
        sem_post(&p->work_sem);
    }

    return n;
}

static int mmap_reap(io_engine_t *eng, int min_completions, int timeout_us)
{
    mmap_priv_t *p = static_cast<mmap_priv_t *>(eng->priv);

    /*
     * Use the monotonically-increasing n_completed counter rather than
     * snapshotting n_inflight at entry.  The old snapshot approach returned
     * 0 whenever workers finished before reap() was called (n_inflight
     * already 0 at entry → start=0 → completed = 0−0 = 0), which caused
     * bench.cpp to break out of its reap loop immediately and skip the
     * entire compute-simulation phase.
     *
     * Here we wait until at least min_completions *new* items have finished
     * beyond what previous reap() calls already reported (n_reaped_out).
     * If workers are already done at entry, n_completed >= target instantly
     * and we return without spinning.
     */
    int already_out = p->n_reaped_out.load();
    int target      = already_out + min_completions;

    uint64_t deadline_ns = 0;
    if (timeout_us > 0) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        deadline_ns = static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL
                    + static_cast<uint64_t>(ts.tv_nsec)
                    + static_cast<uint64_t>(timeout_us) * 1000;
    }

    while (p->n_completed.load() < target) {
        if (timeout_us > 0) {
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            uint64_t now = static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL
                         + static_cast<uint64_t>(ts.tv_nsec);
            if (now >= deadline_ns) break;
        }
        struct timespec ns1 = {0, 100};
        nanosleep(&ns1, nullptr);
    }

    int now_completed    = p->n_completed.load();
    int new_completions  = now_completed - already_out;
    if (new_completions <= 0) return 0;   /* timed out with nothing new */

    p->n_reaped_out.fetch_add(new_completions);

    if (p->n_inflight.load() == 0 && p->pending_items) {
        free(p->pending_items);
        p->pending_items = nullptr;
    }
    return new_completions;
}

static void mmap_destroy(io_engine_t *eng)
{
    mmap_priv_t *p = static_cast<mmap_priv_t *>(eng->priv);

    p->shutdown.store(1);
    for (int i = 0; i < p->n_threads; i++)
        sem_post(&p->work_sem);
    for (int i = 0; i < p->n_threads; i++)
        pthread_join(p->threads[i], nullptr);

    /* Unmap all cached file mappings */
    for (int fd = 0; fd < MMAP_MAX_FDS; fd++) {
        if (p->fd_mapped[fd])
            munmap(p->fd_base[fd], p->fd_size[fd]);
    }

    sem_destroy(&p->work_sem);
    sem_destroy(&p->free_sem);
    pthread_mutex_destroy(&p->mmap_mutex);
    free(p->pending_items);   /* nullptr-safe */
    free(p);
}

static io_engine_ops_t mmap_ops = {
    .name    = "mmap",
    .init    = mmap_init,
    .submit  = mmap_submit,
    .reap    = mmap_reap,
    .destroy = mmap_destroy,
};

io_engine_t *engine_mmap_create(void)
{
    io_engine_t *e = static_cast<io_engine_t *>(calloc(1, sizeof(*e)));
    e->ops = &mmap_ops;
    return e;
}
