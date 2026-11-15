/*
 * io_engine.h: Pluggable IO engine vtable interface
 * All engines implement this interface, and bench.cpp uses it polymorphically.
 */
#ifndef IO_ENGINE_H
#define IO_ENGINE_H

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <atomic>

/* ── Per-request descriptor ──────────────────────────────────────── */
typedef struct {
    int      fd;          /* file descriptor                          */
    uint64_t offset;      /* byte offset in file                      */
    uint32_t length;      /* bytes to read                            */
    void    *buf;         /* destination host buffer (pinned)         */
    int      buf_idx;     /* buffer pool slot index                   */
    uint32_t sample_id;   /* logical sample index (for ordering)      */
    uint64_t submit_ns;   /* timestamp when submitted                 */
    uint64_t complete_ns; /* timestamp when completed                 */
    int      status;      /* 0=pending, 1=complete, <0=error          */
} io_req_t;

/* ── Engine vtable ───────────────────────────────────────────────── */
typedef struct io_engine io_engine_t;

typedef struct {
    const char *name;
    int  (*init)    (io_engine_t *eng, int queue_depth);
    int  (*submit)  (io_engine_t *eng, io_req_t *reqs, int n);
    int  (*reap)    (io_engine_t *eng, int min_completions,
                     int timeout_us);
    void (*destroy) (io_engine_t *eng);
} io_engine_ops_t;

struct io_engine {
    io_engine_ops_t       *ops;
    void                  *priv;          /* engine-private state    */
    int                    queue_depth;
    size_t                 chunk_size;    /* IO granularity, multiple of 4096 */
    int                    force_gap;     /* scissor: override Regressor gap (-1 = auto) */
    size_t                 force_batch_bytes; /* scissor: override batch size (0 = auto)  */
    std::atomic<int>       n_inflight;
};

/* ── Engine factories ────────────────────────────────────────────── */
io_engine_t *engine_pread_create(void);
io_engine_t *engine_mmap_create(void);
io_engine_t *engine_libaio_create(void);
io_engine_t *engine_uring_create(void);

#ifdef HAS_SCISSOR
io_engine_t *engine_scissor_create(void);
void engine_scissor_set_paths(io_engine_t *eng,
                              const char **paths,
                              const int   *fds,
                              int          n);

/* Single-ring variant (engine_scissor_sr.cpp) */
io_engine_t *engine_scissor_sr_create(void);
void engine_scissor_sr_set_paths(io_engine_t *eng,
                                  const char **paths,
                                  const int   *fds,
                                  int          n);
#endif

static inline void engine_free(io_engine_t *eng)
{
    if (eng) {
        if (eng->ops && eng->ops->destroy) eng->ops->destroy(eng);
        free(eng);
    }
}

/* ── Convenience wrappers ────────────────────────────────────────── */
static inline int engine_init(io_engine_t *e, int qd)
{ return e->ops->init(e, qd); }

static inline int engine_submit(io_engine_t *e, io_req_t *r, int n)
{ return e->ops->submit(e, r, n); }

static inline int engine_reap(io_engine_t *e, int min, int tus)
{ return e->ops->reap(e, min, tus); }

#endif /* IO_ENGINE_H */
