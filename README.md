# SCISSOR: Scalable I/O for Small Scattered Objects Runtime

SCISSOR is an I/O runtime for workloads that perform batched reads of many
small, scattered chunks from large files. It has four components:

1. **Aggregator**: coalesces sparse chunk offsets into contiguous I/O
   segments (gap-aware coalescing) to reduce the number of I/O operations.
2. **Regressor**: selects the batch size and coalescing gap that maximize
   predicted throughput using calibrated models of I/O and compute time.
3. **Loader**: drives a producer/consumer pipeline that overlaps I/O with
   computation, with an LRU file-descriptor cache.
4. **Reader**: issues the reads. SCISSOR ships an `io_uring` backend
   (`LiburingReader`) plus POSIX `pread` and `mmap` backends. Any backend
   implementing `scissor::IReader` can be used.

## Requirements

- Linux with `io_uring` support (kernel >= 5.11 recommended for unprivileged SQPOLL)
- GCC >= 10 (C++17), CMake >= 3.14
- liburing >= 2.6. If pkg-config does not find a compatible installation,
  CMake builds liburing 2.6 from source (requires network access at build time)
  and installs the static library alongside SCISSOR.

## Build, test, install

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/path/to/install
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build
```

The default build type is `Release`.

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `SCISSOR_ENABLE_TESTS` | `ON` | Build test executables |
| `SCISSOR_ENABLE_TOOLS` | `ON` | Build the `scissor-profile` CLI tool |
| `SCISSOR_USE_SYSTEM_LIBURING` | `ON` | Look for liburing >= 2.6 with pkg-config before building it from source |
| `SCISSOR_MIN_LOG_LEVEL` | `2` | Minimum compiled-in log level (0=DEBUG, 1=PERF, 2=INFO, 3=WARN, 4=ERROR, 5=FATAL), exported to consumers |
| `SCISSOR_DEFAULT_CONFIG_DIR` | `${prefix}/etc/scissor` | Directory searched for `models.conf` at runtime |

### Tests

| CTest name | Executable | Description |
|------------|------------|-------------|
| `AggregatorTests` | `test_aggregator` | Unit tests for `Aggregator::coalesce()` |
| `RegressorTests` | `test_regressor` | Unit tests for `Regressor` config loading and `optimize()` |
| `LoaderTests` | `test_loader` | Loader pipeline test with a `pread`-based mock reader |

## Using SCISSOR from CMake

```cmake
find_package(scissor REQUIRED)
target_link_libraries(myapp PRIVATE scissor::scissor)
```

## API overview

```cpp
#include <scissor/scissor.hpp>

scissor::LiburingReader reader;                 // one io_uring ring, shared across files
scissor::Loader loader(reader, scissor::Config{false, 0, 64});

// Read the same chunk offsets from two files (chunk size must be a multiple of 4096).
std::vector<size_t> offsets = {1, 2, 3, 10, 11, 40};
scissor::LoadHandle h = loader.load("run0.dat", "run1.dat", offsets, 4096,
                                    scissor::TransferType::FileToHost,
                                    scissor::FileSrc::PFS,
                                    /*nthreads=*/1, /*gap=*/2, /*block_size=*/0);

while (scissor::common::Batch* batch = loader.next(h, scissor::TransferType::FileToHost)) {
    for (size_t i = 0; i < batch->n_segments(); ++i) {
        const scissor::Segment& seg = batch->data[i];   // seg.buffer holds seg.size bytes read at seg.offset
    }
}
loader.close(h);
```

`Loader::load` has three forms: a general N-file scatter read
(`std::vector<FileRequest>`, all requests with the same offsets), a
single-file sequential read, and the two-file comparison pattern shown above.
Offsets are chunk indices, sorted ascending without duplicates.

- **Gap.** Two consecutive offsets are merged into one read when their index
  distance is at most `gap`: `gap=0` disables coalescing (one read per
  offset), `gap=1` merges adjacent chunks only, and `gap=g` reads up to `g-1`
  unrequested chunks between two requested ones.
- **Batch size.** Coalesced reads are grouped into batches covering at least
  `block_size` bytes of requested chunks, and `next()` returns one batch at a time.
  When `nthreads > 1`, the Regressor chooses the gap and batch size.
- **Memory.** A batch's buffer is allocated when its reads are submitted and
  freed by the next `next()` or `close()` on the same handle, so each handle
  holds at most two batches (one being read, one returned to the caller).
- **Statistics.** `wasted_bytes(h)` and `iop_count(h)` report coalescing results, and
  `params(h)` returns the gap and batch size the load used (the Regressor's
  choice when `nthreads > 1`).

Files are opened with `O_DIRECT`, so chunk sizes must be multiples of 4096.

**Errors.** `load()` throws `std::invalid_argument` for invalid arguments and
`std::system_error` when a file cannot be opened. `next()` throws
`std::system_error` when a read fails (close the handle afterwards). Reader
constructors throw `std::system_error` on setup failure.

**Threads.** A `Loader` is not thread-safe: call it from one thread or
serialize the calls. Several handles can be open and read in turn. For
parallel loading, use one `Loader` and one reader per thread.

### Readers

| Reader | Constructor | Notes |
|--------|-------------|-------|
| `LiburingReader` | `LiburingReader()` or `LiburingReader(qdepth, cqdepth)` | `io_uring` with `IORING_SETUP_SQPOLL`. `qdepth` = ring entries (default 32768), `cqdepth` = completions consumed per poll (default 1024). |
| `PosixReader` | `PosixReader(nthreads = 1)` or `PosixReader(path, nthreads = 1)` | `pread` thread pool |
| `MmapReader` | `MmapReader(nthreads = 1)` or `MmapReader(path, nthreads = 1)` | `mmap` (`MADV_RANDOM`, one mapping per file) + `memcpy` thread pool |

All readers read from `Segment::fd`, so one reader serves many files, and
identify segments by `Segment::id`. `PosixReader` and `MmapReader` constructed
with a path also serve segments whose `fd` is -1 from that file. Call
`wait_all()` before enqueueing the next group of segments on them.

### Configuration

`scissor::Config{use_gpu, max_staging_bytes, max_open_fds}`:

| Field | Environment fallback (used when the field is 0) | Behavior |
|-------|--------------------------------------|----------|
| `max_open_fds` | `SCISSOR_MAX_OPEN_FDS` (default 32768) | Capacity of the LRU file-descriptor cache |
| `max_staging_bytes` | `SCISSOR_MAX_STAGING_MB` (unset = no limit) | Upper bound on the batch size used by `load` |

`use_gpu` is currently unused.

## Regressor configuration (models.conf)

The Regressor uses a five-parameter I/O model per storage tier (`pfs`, `ssd`,
`hybrid`) and a log-linear compute model. Parameters are loaded from INI files;
each layer overrides the previous one:

1. Built-in defaults (Polaris calibration, compiled in).
2. `${SCISSOR_DEFAULT_CONFIG_DIR}/models.conf` (installed default).
3. `${HOME}/.scissor/models.conf`.
4. The file named by `SCISSOR_MODEL_CONFIG`.
5. A `RegressorConfig` passed to the `Regressor` constructor.

```ini
[compute]
slope     = 0.748
intercept = -17.0366

[pfs]
# throughput = C * N^alpha * exp(-beta*N) * S^gamma * exp(-delta*S)
# io_params = C, alpha, beta, gamma, delta
io_params = 4.29e+00, -3.33e-02, 7.26e-06, 4.54e-01, 1.37e-01
gaps      = 0, 1, 2, 4, 8, 16
```

`gaps` lists the candidate coalescing gaps, in chunks, explored by the Regressor.

`scissor-profile` measures I/O throughput on a target file and writes a
`models.conf`, and `scissor_fit.py` fits the model coefficients from a profile CSV.
Run either with `--help` for options.

## Logging

SCISSOR writes structured JSONL log entries (`ts`, `level`, `rank`, `comp`,
`src`, `msg`, plus key/value fields) to stderr by default.

| Variable | Description |
|----------|-------------|
| `SCISSOR_LOG_LEVEL` | Runtime log level: `DEBUG`, `PERF`, `INFO`, `WARN`, `ERROR`, `FATAL` (default `INFO`) |
| `SCISSOR_LOG_FILE` | Log output file, where `%r` is replaced by the MPI rank |
| `SCISSOR_LOG_RANK0_ONLY` | If set, ranks > 0 only log `WARN` and above |

The rank is detected from `PMI_RANK`, `PMIX_RANK`, `SLURM_PROCID`,
`OMPI_COMM_WORLD_RANK` or `LSB_RANKID`.

## License

MIT. See [LICENSE](LICENSE).
