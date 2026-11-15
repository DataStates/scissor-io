/*
 * bench.cpp: Main benchmark driver
 *
 * Reads config from environment / CLI, loads trace, opens dataset files,
 * replays the trace through each IO engine with overlapped busy_wait
 * computation, and reports Throughput, IOPS, and IO wait P50/P95/P99.
 *
 * Usage:
 *   ./bench --trace traces/resnet50_trace.bin \
 *           --manifest /data/bench/manifest.txt \
 *           --engine pread|libaio|io_uring|scissor \
 *           --queue-depth 512 \
 *           --compute-mean-ms 28.5 \
 *           --compute-std-ms 4.2 \
 *           --compute-dist normal|gamma|uniform \
 *           --o-direct \
 *           --report-json results/out.json
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <getopt.h>
#include <sys/stat.h>

#include "trace.h"
#include "io_engine.h"
#include "buf_pool.h"
#include "metrics.h"

/* ── Configuration ───────────────────────────────────────────────── */
typedef struct {
    char   trace_bin[512];
    char   manifest[512];
    char   engine_name[32];
    char   report_json[512];
    char   compute_dist[16];   /* normal | gamma | uniform            */
    int    queue_depth;
    int    o_direct;           /* 1 = O_DIRECT | O_RDONLY             */
    double compute_mean_ms;
    double compute_std_ms;
    size_t chunk_size;         /* IO granularity in bytes, multiple of 4096 */
    int    force_gap;          /* scissor: override Regressor gap (-1 = auto) */
    size_t force_batch_bytes;  /* scissor: override batch size in bytes (0 = auto) */
    int    verbose;
} bench_config_t;

/* ── Timing helpers ──────────────────────────────────────────────── */
static inline uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* ── Compute simulator: busy-wait for `ms` milliseconds ─────────── */
static void busy_wait_ms(double ms)
{
    if (ms <= 0.0) return;
    uint64_t end = now_ns() + (uint64_t)(ms * 1e6);
    while (now_ns() < end)
        __asm__ volatile("pause" ::: "memory");   /* CPU hint          */
}

/* ── Sample compute time from distribution ───────────────────────── */
/* Simple Box-Muller for normal distribution (no external dep) */
static double randn(double mu, double sigma)
{
    static int    have_spare = 0;
    static double spare;
    if (have_spare) {
        have_spare = 0;
        return mu + sigma * spare;
    }
    double u, v, s;
    do {
        u = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        v = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    s = sqrt(-2.0 * log(s) / s);
    spare      = v * s;
    have_spare = 1;
    return fmax(0.0, mu + sigma * u * s);
}

static double rand_gamma(double shape, double scale)
{
    /* Marsaglia & Tsang method */
    if (shape < 1.0) {
        return rand_gamma(1.0 + shape, scale)
               * pow((double)rand() / RAND_MAX, 1.0 / shape);
    }
    double d = shape - 1.0 / 3.0;
    double c = 1.0 / sqrt(9.0 * d);
    double x, v;
    while (1) {
        do { x = randn(0.0, 1.0); v = 1.0 + c * x; } while (v <= 0.0);
        v = v * v * v;
        double u = (double)rand() / RAND_MAX;
        if (u < 1.0 - 0.0331 * (x * x) * (x * x)) return d * v * scale;
        if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v))) return d*v*scale;
    }
}

static double sample_compute_ms(const bench_config_t *cfg)
{
    double mu    = cfg->compute_mean_ms;
    double sigma = cfg->compute_std_ms;

    if (strcmp(cfg->compute_dist, "gamma") == 0) {
        double k     = (mu / sigma) * (mu / sigma);
        double theta = (sigma * sigma) / mu;
        return rand_gamma(k, theta);
    } else if (strcmp(cfg->compute_dist, "uniform") == 0) {
        double lo = fmax(0.0, mu - 3.0 * sigma);
        double hi = mu + 3.0 * sigma;
        return lo + ((double)rand() / RAND_MAX) * (hi - lo);
    } else {   /* normal */
        return fmax(0.0, randn(mu, sigma));
    }
}

/* ── Open dataset files from manifest ───────────────────────────── */
/*
 * Opens all files listed in the manifest.  If paths_out is non-NULL,
 * *paths_out receives a malloc'd array of strdup'd path strings that
 * the caller must free (free each element, then free the array).
 */
static int open_files(const char *manifest_path, int o_direct,
                      int **fds_out, int *n_files_out,
                      char ***paths_out)
{
    FILE *fp = fopen(manifest_path, "r");
    if (!fp) { perror(manifest_path); return -1; }

    char lines[256][512];
    int  n = 0;
    while (n < 256 && fgets(lines[n], sizeof(lines[n]), fp))
        lines[n][strcspn(lines[n], "\n")] = 0, n++;
    fclose(fp);

    int flags = O_RDONLY;
    if (o_direct) flags |= O_DIRECT;

    int   *fds   = static_cast<int *>(malloc(n * sizeof(int)));
    char **paths = paths_out ? static_cast<char **>(malloc(n * sizeof(char *))) : nullptr;

    for (int i = 0; i < n; i++) {
        fds[i] = open(lines[i], flags);
        if (fds[i] < 0) {
            fprintf(stderr, "open(%s): %s\n", lines[i], strerror(errno));
            /* Retry without O_DIRECT (filesystem may not support it) */
            if (o_direct) {
                fprintf(stderr, "  Retrying without O_DIRECT\n");
                fds[i] = open(lines[i], O_RDONLY);
            }
            if (fds[i] < 0) {
                if (paths) {
                    for (int j = 0; j < i; j++) free(paths[j]);
                    free(paths);
                }
                free(fds);
                return -1;
            }
        }
        if (paths) paths[i] = strdup(lines[i]);
    }

    *fds_out     = fds;
    *n_files_out = n;
    if (paths_out) *paths_out = paths;
    return 0;
}

/* ── Max records per step (for pool sizing) ──────────────────────── */
/*
 * Compute the largest step batch in the trace.  Called before the
 * buffer pool is created so we can size the pool to the actual working
 * set rather than to queue_depth (which can now be much larger than
 * the batch size).
 */
static int trace_max_batch(const trace_t *trace)
{
    int max_batch = 0;
    int cur_step  = -1;
    for (size_t i = 0; i < trace->n_records; i++) {
        int s = (int)trace->records[i].step;
        if (s != cur_step) {
            size_t end = i;
            while (end < trace->n_records &&
                   (int)trace->records[end].step == s)
                end++;
            int cnt = (int)(end - i);
            if (cnt > max_batch) max_batch = cnt;
            cur_step = s;
        }
    }
    return (max_batch > 0) ? max_batch : 1;
}

/* ── Main benchmark loop ─────────────────────────────────────────── */
static int run_benchmark(io_engine_t  *engine,
                         trace_t      *trace,
                         int          *fds,
                         buf_pool_t   *pool,
                         bench_config_t *cfg,
                         metrics_t    *m)
{
    /* Group trace records by step */
    int      cur_step   = -1;
    size_t   step_start = 0;
    uint64_t step_begin_ns = now_ns();
    uint64_t total_bytes   = 0;

    /* Allocate request array for max batch */
    int max_batch = 0;
    for (size_t i = 0; i < trace->n_records; i++) {
        int s = (int)trace->records[i].step;
        if (s != cur_step) {
            /* count records in this step */
            size_t end = i;
            while (end < trace->n_records &&
                   (int)trace->records[end].step == s) end++;
            int cnt = (int)(end - i);
            if (cnt > max_batch) max_batch = cnt;
            cur_step = s;
        }
    }
    cur_step = -1;

    io_req_t *reqs = static_cast<io_req_t *>(calloc(max_batch, sizeof(io_req_t)));

    for (size_t i = 0; i <= trace->n_records; i++) {
        int new_step = (i < trace->n_records)
                     ? (int)trace->records[i].step
                     : cur_step + 1;

        if (new_step != cur_step) {
            /* ── Process completed step ────────────────────────── */
            if (cur_step >= 0) {
                int batch_n = (int)(i - step_start);

                /*
                 * Sample the compute budget for this step up front so the
                 * proportional slice per reap chunk is stable across the loop.
                 */
                double compute_ms = sample_compute_ms(cfg);

                /*
                 * Interleaved IO + compute loop.
                 *
                 * Each engine returns at least one "chunk" of completions per
                 * reap() call (one file for scissor, typically all requests for
                 * pread/libaio/io_uring which complete in bulk).  After each
                 * chunk, a proportional slice of compute runs as a busy-wait,
                 * mirroring the producer-consumer overlap in the original
                 * state-diff / Kokkos usage where scissor is a pure producer
                 * and the compute thread consumes independently.
                 *
                 * io_wait_ns is measured from the first reap call to the last
                 * IO completion timestamp, excluding compute time.  step_wall_ns
                 * reflects actual elapsed time including overlap.
                 */
                uint64_t t_io_start  = now_ns();
                uint64_t t_last_io   = t_io_start;
                int      total_reaped = 0;

                while (total_reaped < batch_n) {
                    int n = engine->ops->reap(engine, 1, 30000000); /* 30s */
                    if (n <= 0) break;
                    total_reaped += n;
                    t_last_io     = now_ns();

                    /* Proportional compute slice for this chunk */
                    double chunk_ms = compute_ms * (double)n / batch_n;
                    busy_wait_ms(chunk_ms);
                }

                uint64_t t_step_done = now_ns();

                /* Record per-request latencies and release buffer slots */
                uint64_t step_bytes = 0;
                for (int j = 0; j < batch_n; j++) {
                    metrics_record_request(m,
                        reqs[j].submit_ns,
                        reqs[j].complete_ns);
                    step_bytes  += reqs[j].length;
                    total_bytes += reqs[j].length;
                    buf_pool_release(pool, reqs[j].buf_idx);
                }

                metrics_record_step(m,
                    cur_step,
                    t_io_start   - step_begin_ns,  /* io_submit_ns  */
                    t_last_io    - t_io_start,      /* io_wait_ns    */
                    (uint64_t)(compute_ms * 1e6),   /* compute_ns    */
                    t_step_done  - step_begin_ns,   /* step_wall_ns  */
                    step_bytes,
                    batch_n);

                if (cfg->verbose)
                    printf("  step %4d: io_wait=%.1f ms  "
                           "compute=%.1f ms  batch=%d  %.1f MB\n",
                           cur_step,
                           (t_last_io - t_io_start) / 1e6,
                           compute_ms,
                           batch_n,
                           step_bytes / 1e6);

                step_begin_ns = t_step_done;
            }

            cur_step   = new_step;
            step_start = i;
            if (i >= trace->n_records) break;

            /* ── Build and submit batch for new step ─────────── */
            size_t batch_end = i;
            while (batch_end < trace->n_records &&
                   (int)trace->records[batch_end].step == new_step)
                batch_end++;
            int batch_n = (int)(batch_end - i);

            for (int j = 0; j < batch_n; j++) {
                trace_record_t *rec = &trace->records[i + j];

                /* Acquire buffer slot */
                int slot = buf_pool_acquire(pool);
                while (slot < 0) {
                    /* Pool exhausted: reap some completions */
                    engine->ops->reap(engine, 1, 1000);
                    slot = buf_pool_acquire(pool);
                }

                reqs[j].fd        = fds[rec->file_id];
                reqs[j].offset    = rec->offset;
                reqs[j].length    = rec->length;
                reqs[j].buf       = buf_pool_ptr(pool, slot);
                reqs[j].buf_idx   = slot;
                reqs[j].sample_id = (uint32_t)(i + j);
                reqs[j].status    = 0;
                reqs[j].submit_ns = 0;
                reqs[j].complete_ns = 0;
            }

            /* Submit full batch in one call */
            engine->ops->submit(engine, reqs, batch_n);
        }
    }

    free(reqs);
    (void)total_bytes;
    return 0;
}

/* ── CLI parsing ─────────────────────────────────────────────────── */
static void parse_args(int argc, char **argv, bench_config_t *cfg)
{
    static struct option opts[] = {
        {"trace",           required_argument, 0, 't'},
        {"manifest",        required_argument, 0, 'm'},
        {"engine",          required_argument, 0, 'e'},
        {"queue-depth",     required_argument, 0, 'q'},
        {"compute-mean-ms", required_argument, 0, 'c'},
        {"compute-std-ms",  required_argument, 0, 's'},
        {"compute-dist",    required_argument, 0, 'd'},
        {"chunk-size",      required_argument, 0, 'k'},
        {"force-gap",       required_argument, 0, 'g'},
        {"force-batch-mb",  required_argument, 0, 'b'},
        {"o-direct",        no_argument,       0, 'D'},
        {"report-json",     required_argument, 0, 'r'},
        {"verbose",         no_argument,       0, 'v'},
        {0, 0, 0, 0}
    };

    /* Defaults */
    strncpy(cfg->engine_name,   "pread",  sizeof(cfg->engine_name)-1);
    strncpy(cfg->compute_dist,  "normal", sizeof(cfg->compute_dist)-1);
    cfg->queue_depth      = 256;
    cfg->o_direct         = 0;
    cfg->compute_mean_ms  = 25.0;
    cfg->compute_std_ms   = 4.0;
    cfg->chunk_size       = 4096;
    cfg->force_gap        = -1;
    cfg->force_batch_bytes = 0;
    cfg->verbose          = 0;
    strcpy(cfg->report_json, "results/bench_result.json");

    int c, idx;
    while ((c = getopt_long(argc, argv, "t:m:e:q:c:s:d:k:g:Dr:v",
                            opts, &idx)) != -1) {
        switch (c) {
        case 't': strncpy(cfg->trace_bin,  optarg, 511); break;
        case 'm': strncpy(cfg->manifest,   optarg, 511); break;
        case 'e': strncpy(cfg->engine_name,optarg, 31);  break;
        case 'q': cfg->queue_depth     = atoi(optarg);   break;
        case 'c': cfg->compute_mean_ms = atof(optarg);   break;
        case 's': cfg->compute_std_ms  = atof(optarg);   break;
        case 'd': strncpy(cfg->compute_dist, optarg, 15); break;
        case 'k': {
            size_t ks = (size_t)atol(optarg);
            if (ks == 0 || ks % 4096 != 0) {
                fprintf(stderr, "--chunk-size must be a non-zero multiple of 4096\n");
                exit(1);
            }
            cfg->chunk_size = ks;
            break;
        }
        case 'g': cfg->force_gap        = atoi(optarg);               break;
        case 'b': cfg->force_batch_bytes = (size_t)atol(optarg)
                                           * 1024UL * 1024UL;         break;
        case 'D': cfg->o_direct        = 1;               break;
        case 'r': strncpy(cfg->report_json, optarg, 511); break;
        case 'v': cfg->verbose         = 1;               break;
        default:
            fprintf(stderr,
                "Usage: bench --trace FILE --manifest FILE "
                "--engine pread|mmap|libaio|io_uring|scissor|scissor_sr [options]\n");
            exit(1);
        }
    }

    if (!cfg->trace_bin[0] || !cfg->manifest[0]) {
        fprintf(stderr, "Error: --trace and --manifest are required\n");
        exit(1);
    }
}

/* ── Entry point ─────────────────────────────────────────────────── */
int main(int argc, char **argv)
{
    bench_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    parse_args(argc, argv, &cfg);

    printf("\n[bench] Engine:       %s\n",  cfg.engine_name);
    printf("[bench] Trace:        %s\n",  cfg.trace_bin);
    printf("[bench] Manifest:     %s\n",  cfg.manifest);
    printf("[bench] Queue depth:  %d\n",  cfg.queue_depth);
    printf("[bench] Chunk size:   %zu KB\n", cfg.chunk_size / 1024);
    printf("[bench] O_DIRECT:     %s\n",  cfg.o_direct ? "yes" : "no");
    printf("[bench] Compute:      %.1f ± %.1f ms (%s)\n",
           cfg.compute_mean_ms, cfg.compute_std_ms, cfg.compute_dist);

    /* Load trace */
    trace_t *trace = trace_load(cfg.trace_bin);
    if (!trace) { fprintf(stderr, "Failed to load trace\n"); return 1; }
    printf("[bench] Trace:        %zu records, %d steps, %d files\n",
           trace->n_records, trace->n_steps, trace->n_files);

    /* Open dataset files */
    int    *fds     = nullptr;
    char  **paths   = nullptr;
    int     n_files = 0;

    if (open_files(cfg.manifest, cfg.o_direct, &fds, &n_files, &paths) < 0) {
        trace_free(trace); return 1;
    }
    printf("[bench] Opened:       %d dataset files\n", n_files);

    /* Derive buffer slot size from the actual maximum record length in the
     * trace rather than using a hardcoded 2 MB ceiling.  For 8 KB bert
     * records the old 2 MB default wasted 250× memory per slot and caused
     * mlock pressure (512 MB pinned for queue_depth=128). */
    size_t max_sample = 0;
    for (size_t i = 0; i < trace->n_records; i++) {
        if (trace->records[i].length > max_sample)
            max_sample = trace->records[i].length;
    }
    /* Round up to 4 KB O_DIRECT alignment boundary */
    max_sample = (max_sample + 4095) & ~(size_t)4095;
    if (max_sample == 0) max_sample = 65536;   /* fallback: 64 KB */

    /*
     * Pool capacity is derived from the actual largest step batch in the
     * trace, not from queue_depth.  queue_depth can now be 32 K (matching
     * the io_uring ring size and the storage controller's concurrency), but
     * each step only submits max_batch requests simultaneously.  Allocating
     * queue_depth×2 slots at QD=32 K and 160 KB/slot would pin 10 GB of
     * memory, which is both wasteful and likely to fail mlock.
     */
    int        max_batch = trace_max_batch(trace);
    buf_pool_t *pool     = buf_pool_create(max_batch * 2, max_sample);
    printf("[bench] Buffer pool:  %d slots × %zu KB = %zu MB pinned\n",
           max_batch * 2,
           max_sample / 1024,
           (size_t)max_batch * 2 * max_sample / (1024 * 1024));

    /* Create IO engine */
    io_engine_t *engine = nullptr;
    if (strcmp(cfg.engine_name, "pread") == 0)
        engine = engine_pread_create();
    else if (strcmp(cfg.engine_name, "mmap") == 0)
        engine = engine_mmap_create();
#ifdef HAS_LIBAIO
    else if (strcmp(cfg.engine_name, "libaio") == 0)
        engine = engine_libaio_create();
#endif
#ifdef HAS_URING
    else if (strcmp(cfg.engine_name, "io_uring") == 0)
        engine = engine_uring_create();
#endif
#ifdef HAS_SCISSOR
    else if (strcmp(cfg.engine_name, "scissor") == 0)
        engine = engine_scissor_create();
    else if (strcmp(cfg.engine_name, "scissor_sr") == 0)
        engine = engine_scissor_sr_create();
#endif
    else {
        fprintf(stderr, "Unknown engine: %s\n", cfg.engine_name);
#ifndef HAS_URING
        if (strcmp(cfg.engine_name, "io_uring") == 0)
            fprintf(stderr, "  (io_uring engine was not compiled in: liburing not found)\n");
#endif
#ifndef HAS_SCISSOR
        if (strcmp(cfg.engine_name, "scissor") == 0)
            fprintf(stderr, "  (scissor engine was not compiled in)\n");
#endif
        return 1;
    }

    engine->chunk_size         = cfg.chunk_size;
    engine->force_gap          = cfg.force_gap;
    engine->force_batch_bytes  = cfg.force_batch_bytes;
    if (engine_init(engine, cfg.queue_depth) < 0) {
        fprintf(stderr, "Engine init failed\n"); return 1;
    }

    /* Register fd→path mapping for scissor engines */
#ifdef HAS_SCISSOR
    if (strcmp(cfg.engine_name, "scissor") == 0) {
        engine_scissor_set_paths(engine,
                                 (const char **)paths,
                                 fds,
                                 n_files);
    } else if (strcmp(cfg.engine_name, "scissor_sr") == 0) {
        engine_scissor_sr_set_paths(engine,
                                    (const char **)paths,
                                    fds,
                                    n_files);
    }
#endif

    /*
     * Annotate the scissor engine name with the gap mode so the JSON
     * "engine" field distinguishes gap-sweep runs from each other and
     * from the Regressor-auto run.
     */
    if (strcmp(cfg.engine_name, "scissor") == 0) {
        if (cfg.force_gap >= 0)
            snprintf(cfg.engine_name, sizeof(cfg.engine_name),
                     "scissor_g%d", cfg.force_gap);
        else
            strncpy(cfg.engine_name, "scissor_gauto",
                    sizeof(cfg.engine_name) - 1);
    } else if (strcmp(cfg.engine_name, "scissor_sr") == 0) {
        if (cfg.force_gap >= 0)
            snprintf(cfg.engine_name, sizeof(cfg.engine_name),
                     "scissor_sr_g%d", cfg.force_gap);
        else
            strncpy(cfg.engine_name, "scissor_sr_gauto",
                    sizeof(cfg.engine_name) - 1);
    }

    /* Create metrics collector */
    int max_reqs = trace->n_records + trace->n_steps;
    metrics_t *m = metrics_create(cfg.engine_name,
                                  trace->n_steps + 1,
                                  max_reqs);

    /* ── RUN ─────────────────────────────────────────────────────── */
    printf("[bench] Starting benchmark ...\n\n");
    srand(42);   /* reproducible compute time sampling */

    uint64_t t_start = now_ns();
    run_benchmark(engine, trace, fds, pool, &cfg, m);
    uint64_t t_end   = now_ns();

    printf("[bench] Completed in %.2f s\n", (t_end - t_start) / 1e9);

    /* ── REPORT ──────────────────────────────────────────────────── */
    metrics_print_report(m);

    /* Ensure output directory exists */
    {
        char dir[512];
        strncpy(dir, cfg.report_json, sizeof(dir)-1);
        char *slash = strrchr(dir, '/');
        if (slash) { *slash = 0; mkdir(dir, 0755); }
    }
    metrics_write_json(m, cfg.report_json);

    /* Cleanup */
    metrics_free(m);
    engine_free(engine);
    buf_pool_destroy(pool);
    for (int i = 0; i < n_files; i++) close(fds[i]);
    free(fds);
    if (paths) {
        for (int i = 0; i < n_files; i++) free(paths[i]);
        free(paths);
    }
    trace_free(trace);

    return 0;
}
