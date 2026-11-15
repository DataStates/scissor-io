/*
 * trace.h: Trace record definitions and binary loader
 * Shared between bench.c and all engine implementations.
 */
#ifndef TRACE_H
#define TRACE_H

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ── Binary trace file format ─────────────────────────────────────── */
#define TRACE_MAGIC   "AIOT"
#define TRACE_VERSION 1

#pragma pack(push, 1)
typedef struct {
    uint32_t step;           /* training step index                   */
    uint16_t file_id;        /* index into file descriptor table      */
    uint16_t _pad;
    uint64_t offset;         /* byte offset in file (4 KB aligned)    */
    uint32_t length;         /* bytes to read (4 KB aligned)          */
    float    ideal_start_s;  /* ideal wall-clock issue time (seconds) */
} trace_record_t;
#pragma pack(pop)

#define TRACE_RECORD_SIZE sizeof(trace_record_t)   /* should be 24 */

typedef struct {
    uint8_t  magic[4];
    uint32_t version;
    uint32_t n_records;
} trace_header_t;

/* ── Loaded trace ─────────────────────────────────────────────────── */
typedef struct {
    trace_record_t *records;
    size_t          n_records;
    int             n_steps;
    int             n_files;
} trace_t;

/* Load binary trace file.  Returns NULL on error. */
static inline trace_t *trace_load(const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return NULL; }

    trace_header_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1) {
        fprintf(stderr, "trace_load: short header\n");
        fclose(fp); return NULL;
    }
    if (memcmp(hdr.magic, TRACE_MAGIC, 4) != 0) {
        fprintf(stderr, "trace_load: bad magic\n");
        fclose(fp); return NULL;
    }

    trace_t *t = static_cast<trace_t *>(calloc(1, sizeof(*t)));
    t->n_records = hdr.n_records;
    t->records   = static_cast<trace_record_t *>(malloc(t->n_records * sizeof(trace_record_t)));
    if (!t->records) { perror("malloc"); fclose(fp); free(t); return NULL; }

    size_t got = fread(t->records, sizeof(trace_record_t),
                       t->n_records, fp);
    fclose(fp);
    if (got != t->n_records) {
        fprintf(stderr, "trace_load: expected %zu records, got %zu\n",
                t->n_records, got);
        free(t->records); free(t); return NULL;
    }

    /* Compute derived fields */
    int max_step = -1, max_file = -1;
    for (size_t i = 0; i < t->n_records; i++) {
        if ((int)t->records[i].step    > max_step) max_step = t->records[i].step;
        if ((int)t->records[i].file_id > max_file) max_file = t->records[i].file_id;
    }
    t->n_steps = max_step + 1;
    t->n_files = max_file + 1;

    return t;
}

static inline void trace_free(trace_t *t)
{
    if (t) { free(t->records); free(t); }
}

#endif /* TRACE_H */
