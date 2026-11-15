/*
 * engine_pread.cpp: pread(2) engine with POSIX thread pool
 * Each IO request is dispatched to a worker thread that calls pread().
 * Simulates parallel random reads via N_THREADS concurrent syscalls.
 */
#include "io_engine.h"
#include <pthread.h>
#include <semaphore.h>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <cerrno>
#include <atomic>
#include <ctime>

#define PREAD_MAX_THREADS 128
#define PREAD_MAX_QUEUE   4096

typedef struct {
    io_req_t          *req;
    std::atomic<int>   done;     /* 1 when complete */
} work_item_t;

typedef struct {
    /* Thread pool */
    pthread_t   threads[PREAD_MAX_THREADS];
    int         n_threads;
    size_t      chunk_size;

    /* Work queue (ring buffer) */
    work_item_t        *queue[PREAD_MAX_QUEUE];
    std::atomic<int>    q_head;
    std::atomic<int>    q_tail;
    sem_t               work_sem;     /* signals available work      */
    sem_t               free_sem;     /* signals available queue slots */
    std::atomic<int>    shutdown;

    /* In-flight tracking */
    std::atomic<int>    n_inflight;

    /* Heap allocation for the current submit batch, freed in reap/destroy */
    work_item_t        *pending_items;
} pread_priv_t;

static void *pread_worker_thread(void *arg)
{
    pread_priv_t *p = static_cast<pread_priv_t *>(arg);
    while (1) {
        sem_wait(&p->work_sem);
        if (p->shutdown.load()) break;

        /* Dequeue one item */
        int tail = p->q_tail.fetch_add(1) % PREAD_MAX_QUEUE;
        work_item_t *item = p->queue[tail];
        sem_post(&p->free_sem);

        if (!item || !item->req) continue;

        io_req_t *req = item->req;

        /*
         * Issue one pread() per chunk to match the N chunk-sized reads that
         * io_uring/libaio enqueue, against which scissor's M < N coalesced
         * segments are compared.
         */
        size_t chunk_size = p->chunk_size;
        size_t n_chunks   = (req->length + chunk_size - 1) / chunk_size;

        req->status = 1;
        for (size_t c = 0; c < n_chunks && req->status == 1; c++) {
            void  *dst = static_cast<char *>(req->buf) + c * chunk_size;
            off_t  off = static_cast<off_t>(req->offset + c * chunk_size);
            ssize_t n  = pread(req->fd, dst, chunk_size, off);
            if (n < 0) {
                req->status = -errno;
            } else if (static_cast<size_t>(n) < chunk_size) {
                /* Partial: retry remainder (O_DIRECT may short-read) */
                size_t done = static_cast<size_t>(n);
                while (done < chunk_size) {
                    n = pread(req->fd,
                              static_cast<char *>(dst) + done,
                              chunk_size - done,
                              off + static_cast<off_t>(done));
                    if (n <= 0) break;
                    done += static_cast<size_t>(n);
                }
                /* Accept a short last chunk at end-of-file */
                if (req->offset + c * chunk_size + done < req->offset + req->length)
                    req->status = -EIO;
            }
        }

        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        req->complete_ns = static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;

        item->done.store(1);
        p->n_inflight.fetch_sub(1);
    }
    return nullptr;
}

static int pread_init(io_engine_t *eng, int queue_depth)
{
    pread_priv_t *p = static_cast<pread_priv_t *>(calloc(1, sizeof(*p)));
    p->n_threads  = queue_depth < PREAD_MAX_THREADS
                  ? queue_depth : PREAD_MAX_THREADS;
    p->chunk_size = (eng->chunk_size > 0) ? eng->chunk_size : 4096;

    sem_init(&p->work_sem, 0, 0);
    sem_init(&p->free_sem, 0, PREAD_MAX_QUEUE);
    p->shutdown.store(0);
    p->n_inflight.store(0);
    p->q_head.store(0);
    p->q_tail.store(0);

    for (int i = 0; i < p->n_threads; i++)
        pthread_create(&p->threads[i], nullptr, pread_worker_thread, p);

    eng->priv        = p;
    eng->queue_depth = p->n_threads;
    return 0;
}

static int pread_submit(io_engine_t *eng, io_req_t *reqs, int n)
{
    pread_priv_t *p = static_cast<pread_priv_t *>(eng->priv);

    work_item_t *items = static_cast<work_item_t *>(malloc(n * sizeof(work_item_t)));
    if (!items) return -ENOMEM;

    for (int i = 0; i < n; i++) {
        items[i].req = &reqs[i];
        items[i].done.store(0);

        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        reqs[i].submit_ns = static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
        reqs[i].status    = 0;

        sem_wait(&p->free_sem);
        int head = p->q_head.fetch_add(1) % PREAD_MAX_QUEUE;
        p->queue[head] = &items[i];
        sem_post(&p->work_sem);
    }

    p->pending_items = items;
    p->n_inflight.fetch_add(n);
    return n;
}

static int pread_reap(io_engine_t *eng, int min_completions, int timeout_us)
{
    pread_priv_t *p = static_cast<pread_priv_t *>(eng->priv);

    int start           = p->n_inflight.load();
    int target_inflight = start - min_completions;
    if (target_inflight < 0) target_inflight = 0;

    uint64_t deadline_ns = 0;
    if (timeout_us > 0) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        deadline_ns = static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL
                    + ts.tv_nsec + static_cast<uint64_t>(timeout_us) * 1000;
    }

    /* Spin-poll n_inflight with 100 ns sleeps. This avoids busy-wait CPU burn
     * while remaining responsive enough for short IO latencies. */
    while (p->n_inflight.load() > target_inflight) {
        if (timeout_us > 0) {
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            uint64_t now = static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
            if (now >= deadline_ns) break;
        }
        struct timespec ns1 = {0, 100};
        nanosleep(&ns1, nullptr);
    }

    int completed = start - p->n_inflight.load();

    /* Free the current batch once all items have retired */
    if (p->n_inflight.load() == 0 && p->pending_items) {
        free(p->pending_items);
        p->pending_items = nullptr;
    }
    return completed;
}

static void pread_destroy(io_engine_t *eng)
{
    pread_priv_t *p = static_cast<pread_priv_t *>(eng->priv);
    p->shutdown.store(1);
    for (int i = 0; i < p->n_threads; i++)
        sem_post(&p->work_sem);
    for (int i = 0; i < p->n_threads; i++)
        pthread_join(p->threads[i], nullptr);
    sem_destroy(&p->work_sem);
    sem_destroy(&p->free_sem);
    free(p->pending_items);  /* nullptr-safe */
    free(p);
}

static io_engine_ops_t pread_ops = {
    .name    = "pread",
    .init    = pread_init,
    .submit  = pread_submit,
    .reap    = pread_reap,
    .destroy = pread_destroy,
};

io_engine_t *engine_pread_create(void)
{
    io_engine_t *e = static_cast<io_engine_t *>(calloc(1, sizeof(*e)));
    e->ops = &pread_ops;
    return e;
}
