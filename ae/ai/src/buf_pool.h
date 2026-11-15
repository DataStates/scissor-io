/*
 * buf_pool.h: Lock-free pinned buffer pool
 * Buffers are page-aligned (4 KB) and mlock'd for O_DIRECT DMA.
 */
#ifndef BUF_POOL_H
#define BUF_POOL_H

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <atomic>
#include <sys/mman.h>
#include <cstdio>

#define BUF_ALIGN 4096

typedef struct {
    void                **slots;        /* array of buffer pointers  */
    int                   capacity;
    size_t                slot_size;    /* bytes per slot             */
    std::atomic<int>      free_top;     /* top of free-list stack    */
    int                  *free_stack;   /* lock-free free list       */
} buf_pool_t;

static inline buf_pool_t *buf_pool_create(int capacity, size_t slot_size)
{
    buf_pool_t *p = static_cast<buf_pool_t *>(calloc(1, sizeof(*p)));
    p->capacity   = capacity;
    p->slot_size  = slot_size;
    p->slots      = static_cast<void **>(malloc(capacity * sizeof(void *)));
    p->free_stack = static_cast<int *>(malloc(capacity * sizeof(int)));

    for (int i = 0; i < capacity; i++) {
        if (posix_memalign(&p->slots[i], BUF_ALIGN, slot_size) != 0) {
            perror("posix_memalign"); exit(1);
        }
        /* Pin in RAM: required for O_DIRECT DMA */
        if (mlock(p->slots[i], slot_size) != 0) {
            /* Non-fatal: may lack CAP_IPC_LOCK, so warn and continue */
            static int warned = 0;
            if (!warned) {
                fprintf(stderr, "[buf_pool] mlock failed (need CAP_IPC_LOCK "
                        "or raise RLIMIT_MEMLOCK), continuing without pin\n");
                warned = 1;
            }
        }
        p->free_stack[i] = i;
    }
    p->free_top.store(capacity - 1);
    return p;
}

/* Acquire a buffer slot (returns index, or -1 if pool empty). */
static inline int buf_pool_acquire(buf_pool_t *p)
{
    int top = p->free_top.load();
    while (top >= 0) {
        if (p->free_top.compare_exchange_weak(top, top - 1))
            return p->free_stack[top];
        /* CAS failed: retry with updated top */
    }
    return -1;   /* pool exhausted */
}

/* Release a buffer slot back to the pool. */
static inline void buf_pool_release(buf_pool_t *p, int idx)
{
    /* Previous code used fetch_add to claim the new stack slot before writing
     * the slot value, creating a window where a concurrent acquire() could read
     * free_stack[top] before idx was stored there, returning a garbage fd index.
     * Fix: write the slot first, then make it visible via CAS so no acquirer
     * can observe the incremented free_top before the write is complete. */
    int top = p->free_top.load(std::memory_order_relaxed);
    while (true) {
        p->free_stack[top + 1] = idx;
        if (p->free_top.compare_exchange_weak(top, top + 1,
                                              std::memory_order_release,
                                              std::memory_order_relaxed))
            break;
        /* CAS failed: another release changed free_top. Reload and retry. */
    }
}

static inline void *buf_pool_ptr(buf_pool_t *p, int idx)
{
    return p->slots[idx];
}

static inline void buf_pool_destroy(buf_pool_t *p)
{
    if (!p) return;
    for (int i = 0; i < p->capacity; i++) {
        munlock(p->slots[i], p->slot_size);
        free(p->slots[i]);
    }
    free(p->slots);
    free(p->free_stack);
    free(p);
}

#endif /* BUF_POOL_H */
