/*
 * metrics.h: Per-step and per-request metrics collection.
 * Computes throughput, IOPS, and IO wait latency percentiles.
 */
#ifndef METRICS_H
#define METRICS_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

typedef struct {
    int      step;
    uint64_t io_submit_ns;     /* time spent in submit() call         */
    uint64_t io_wait_ns;       /* time blocked waiting for completions*/
    uint64_t compute_ns;       /* actual busy_wait duration           */
    uint64_t step_wall_ns;     /* total wall clock for step           */
    uint64_t bytes_read;       /* total bytes read this step          */
    int      n_requests;       /* number of IO requests this step     */
} step_metrics_t;

typedef struct {
    /* per-request IO latency samples (submit→complete) in nanoseconds */
    uint64_t *lat_samples;
    size_t    lat_count;
    size_t    lat_cap;

    step_metrics_t *steps;
    int             n_steps;
    int             step_cap;

    /* engine name for reporting */
    char engine_name[32];
} metrics_t;

static inline metrics_t *metrics_create(const char *engine_name,
                                        int max_steps,
                                        int max_reqs)
{
    metrics_t *m  = static_cast<metrics_t *>(calloc(1, sizeof(*m)));
    m->steps      = static_cast<step_metrics_t *>(calloc(max_steps, sizeof(step_metrics_t)));
    m->step_cap   = max_steps;
    m->lat_samples = static_cast<uint64_t *>(calloc(max_reqs, sizeof(uint64_t)));
    m->lat_cap    = max_reqs;
    strncpy(m->engine_name, engine_name, sizeof(m->engine_name)-1);
    return m;
}

static inline void metrics_record_request(metrics_t *m,
                                          uint64_t submit_ns,
                                          uint64_t complete_ns)
{
    if (m->lat_count < m->lat_cap)
        m->lat_samples[m->lat_count++] = complete_ns - submit_ns;
}

static inline void metrics_record_step(metrics_t *m,
                                       int step,
                                       uint64_t io_submit_ns,
                                       uint64_t io_wait_ns,
                                       uint64_t compute_ns,
                                       uint64_t step_wall_ns,
                                       uint64_t bytes_read,
                                       int n_requests)
{
    if (m->n_steps >= m->step_cap) return;
    step_metrics_t *s = &m->steps[m->n_steps++];
    s->step          = step;
    s->io_submit_ns  = io_submit_ns;
    s->io_wait_ns    = io_wait_ns;
    s->compute_ns    = compute_ns;
    s->step_wall_ns  = step_wall_ns;
    s->bytes_read    = bytes_read;
    s->n_requests    = n_requests;
}

/* Compare function for qsort */
static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(uint64_t *)a;
    uint64_t y = *(uint64_t *)b;
    return (x > y) - (x < y);
}

static inline uint64_t percentile(uint64_t *sorted, size_t n, double pct)
{
    if (n == 0) return 0;
    size_t idx = (size_t)(pct / 100.0 * (n - 1) + 0.5);
    if (idx >= n) idx = n - 1;
    return sorted[idx];
}

static inline void metrics_print_report(metrics_t *m)
{
    if (m->n_steps == 0 || m->lat_count == 0) {
        printf("[%s] No metrics recorded.\n", m->engine_name);
        return;
    }

    /* Aggregate step metrics */
    uint64_t total_bytes    = 0;
    uint64_t total_wall_ns  = 0;
    uint64_t total_io_ns    = 0;
    int      total_reqs     = 0;

    for (int i = 0; i < m->n_steps; i++) {
        total_bytes   += m->steps[i].bytes_read;
        total_wall_ns += m->steps[i].step_wall_ns;
        total_io_ns   += m->steps[i].io_wait_ns;
        total_reqs    += m->steps[i].n_requests;
    }

    double elapsed_s    = total_wall_ns / 1e9;
    double throughput   = (double)total_bytes / (1 << 30) / elapsed_s;
    double iops         = (double)total_reqs  / elapsed_s;

    /* Sort latency samples for percentiles */
    qsort(m->lat_samples, m->lat_count, sizeof(uint64_t), cmp_u64);

    double p50 = percentile(m->lat_samples, m->lat_count, 50.0) / 1e6;
    double p95 = percentile(m->lat_samples, m->lat_count, 95.0) / 1e6;
    double p99 = percentile(m->lat_samples, m->lat_count, 99.0) / 1e6;

    double gpu_stall = (double)total_io_ns /
                       (double)(total_wall_ns) * 100.0;

    printf("\n");
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  Engine: %-44s║\n", m->engine_name);
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  Steps:          %-35d║\n", m->n_steps);
    printf("║  Total requests: %-35d║\n", total_reqs);
    printf("║  Total data:     %-30.2f GB║\n",
           (double)total_bytes / (1 << 30));
    printf("║  Elapsed:        %-30.2f s ║\n", elapsed_s);
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  Throughput:     %-30.2f GB/s║\n", throughput);
    printf("║  IOPS:           %-34.0f║\n", iops);
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  IO Wait P50:    %-30.2f ms ║\n", p50);
    printf("║  IO Wait P95:    %-30.2f ms ║\n", p95);
    printf("║  IO Wait P99:    %-30.2f ms ║\n", p99);
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  Compute stall:  %-29.1f %% ║\n", gpu_stall);
    printf("╚══════════════════════════════════════════════════════╝\n");
}

static inline void metrics_write_json(metrics_t *m, const char *path)
{
    /* Sort latency samples */
    qsort(m->lat_samples, m->lat_count, sizeof(uint64_t), cmp_u64);
    double p50 = percentile(m->lat_samples, m->lat_count, 50.0) / 1e6;
    double p95 = percentile(m->lat_samples, m->lat_count, 95.0) / 1e6;
    double p99 = percentile(m->lat_samples, m->lat_count, 99.0) / 1e6;

    uint64_t total_bytes = 0, total_wall_ns = 0, total_io_ns = 0;
    int total_reqs = 0;
    for (int i = 0; i < m->n_steps; i++) {
        total_bytes   += m->steps[i].bytes_read;
        total_wall_ns += m->steps[i].step_wall_ns;
        total_io_ns   += m->steps[i].io_wait_ns;
        total_reqs    += m->steps[i].n_requests;
    }
    double elapsed_s  = total_wall_ns / 1e9;
    double throughput = (double)total_bytes / (1 << 30) / elapsed_s;
    double iops       = (double)total_reqs / elapsed_s;

    FILE *fp = fopen(path, "w");
    if (!fp) { perror(path); return; }

    fprintf(fp,
        "{\n"
        "  \"engine\": \"%s\",\n"
        "  \"n_steps\": %d,\n"
        "  \"total_requests\": %d,\n"
        "  \"total_bytes_gb\": %.4f,\n"
        "  \"elapsed_s\": %.4f,\n"
        "  \"throughput_gbs\": %.4f,\n"
        "  \"iops\": %.1f,\n"
        "  \"io_wait_p50_ms\": %.3f,\n"
        "  \"io_wait_p95_ms\": %.3f,\n"
        "  \"io_wait_p99_ms\": %.3f,\n"
        "  \"compute_stall_pct\": %.2f,\n"
        "  \"steps\": [\n",
        m->engine_name,
        m->n_steps,
        total_reqs,
        (double)total_bytes / (1 << 30),
        elapsed_s,
        throughput,
        iops,
        p50, p95, p99,
        (double)total_io_ns / (double)total_wall_ns * 100.0);

    for (int i = 0; i < m->n_steps; i++) {
        step_metrics_t *s = &m->steps[i];
        fprintf(fp,
            "    {\"step\":%d,\"io_wait_ms\":%.3f,"
            "\"compute_ms\":%.3f,\"bytes\":%lu,\"reqs\":%d}%s\n",
            s->step,
            s->io_wait_ns  / 1e6,
            s->compute_ns  / 1e6,
            (unsigned long)s->bytes_read,
            s->n_requests,
            i < m->n_steps - 1 ? "," : "");
    }
    fprintf(fp, "  ]\n}\n");
    fclose(fp);
    printf("[metrics] JSON report → %s\n", path);
}

static inline void metrics_free(metrics_t *m)
{
    if (m) {
        free(m->steps);
        free(m->lat_samples);
        free(m);
    }
}

#endif /* METRICS_H */
