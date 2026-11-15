#include <scissor/scissor.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <unistd.h>
#include <vector>
#include <sys/stat.h>

// ======================================================
// scissor-profile
//
// CLI tool that benchmarks I/O throughput.
//
// Quick mode (default):
//   Sweeps chunk sizes with a fixed IOP count, finds the best
//   chunk size, and writes built-in defaults to models.conf.
//
// Calibrate mode (--calibrate):
//   Sweeps a 2D grid of (chunk_size x n_iops), measures
//   throughput at each point using direct pread I/O,
//   writes a CSV for downstream fitting by scissor_fit.py.
//
// Usage:
//   scissor-profile --storage {pfs|ssd|hybrid}
//                   --file <path>
//                   [--size-gb <N>]
//                   [--output <path>]
//                   [--calibrate]
//                   [--csv-output <path>]
//                   [--reps <N>]
// ======================================================

// ======================================================
// Argument parsing
// ======================================================
struct Args {
    std::string storage     = "pfs";
    std::string file;
    double      size_gb     = 1.0;
    std::string output      = "models.conf";
    bool        calibrate   = false;
    std::string csv_output  = "profile.csv";
    int         n_reps      = 3;
};

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << "\n"
        << "  --storage    {pfs|ssd|hybrid}  Storage tier to profile (default: pfs)\n"
        << "  --file       <path>            File to benchmark reads against\n"
        << "  --size-gb    <N>               GB of data to read per run (default: 1.0)\n"
        << "  --output     <path>            Output models.conf path (default: models.conf)\n"
        << "  --calibrate                    Enable calibrate mode (2D grid sweep + CSV)\n"
        << "  --csv-output <path>            CSV output path in calibrate mode (default: profile.csv)\n"
        << "  --reps       <N>               Trials per grid point in calibrate mode (default: 3)\n";
}

static Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--storage" || arg == "-s") && i + 1 < argc) {
            args.storage = argv[++i];
        } else if ((arg == "--file" || arg == "-f") && i + 1 < argc) {
            args.file = argv[++i];
        } else if ((arg == "--size-gb" || arg == "-n") && i + 1 < argc) {
            args.size_gb = std::stod(argv[++i]);
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            args.output = argv[++i];
        } else if (arg == "--calibrate") {
            args.calibrate = true;
        } else if (arg == "--csv-output" && i + 1 < argc) {
            args.csv_output = argv[++i];
        } else if (arg == "--reps" && i + 1 < argc) {
            args.n_reps = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            exit(1);
        }
    }
    if (args.file.empty()) {
        std::cerr << "Error: --file is required.\n";
        print_usage(argv[0]);
        exit(1);
    }
    return args;
}

// ======================================================
// get_file_size_bytes: stat file and return size.
// ======================================================
static size_t get_file_size_bytes(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) < 0) {
        std::cerr << "Cannot stat file: " << path << "\n";
        exit(1);
    }
    return static_cast<size_t>(st.st_size);
}

// ======================================================
// Quick-mode: measure throughput using the scissor Loader.
// ======================================================
static double measure_throughput_quick(const std::string& file_path,
                                       size_t chunk_size,
                                       size_t n_chunks) {
    scissor::PosixReader posix_reader(file_path);
    scissor::Loader loader(posix_reader, {false, 0, 4});

    size_t total = chunk_size * n_chunks;
    auto   t0    = std::chrono::high_resolution_clock::now();

    scissor::LoadHandle load_handle = loader.load(file_path, total, chunk_size,
                                        scissor::TransferType::FileToHost);
    scissor::common::Batch* batch = nullptr;
    while ((batch = loader.next(load_handle, scissor::TransferType::FileToHost)) != nullptr) {
        // Drain the pipeline, no computation.
    }
    loader.close(load_handle);

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    double gib       = static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0);
    return gib / elapsed_s;
}

// ======================================================
// Calibrate-mode: one trial of (chunk_size, n_iops) using
// direct pread with random offsets. Returns throughput in GiB/s
// and elapsed_s via output parameter.
// ======================================================
static double measure_throughput_calibrate(const std::string& file_path,
                                           size_t chunk_size,
                                           size_t n_iops,
                                           size_t file_size,
                                           int seed,
                                           double& elapsed_s_out) {
    // Open with O_DIRECT, and fall back without it if that fails.
    int fd = open(file_path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        fd = open(file_path.c_str(), O_RDONLY);
        if (fd < 0) {
            std::cerr << "Cannot open file: " << file_path << "\n";
            exit(1);
        }
    }

    // Allocate page-aligned buffer.
    void* buf = nullptr;
    if (posix_memalign(&buf, 4096, chunk_size) != 0) {
        std::cerr << "posix_memalign failed for chunk_size=" << chunk_size << "\n";
        close(fd);
        exit(1);
    }

    // Generate n_iops random offsets aligned to chunk_size in [0, file_size - chunk_size].
    size_t max_offset = file_size - chunk_size;
    // Number of aligned positions.
    size_t n_positions = max_offset / chunk_size + 1;

    std::vector<off_t> offsets(n_iops);
    // Simple LCG seeded by the given seed.
    uint64_t rng_state = static_cast<uint64_t>(seed);
    for (size_t i = 0; i < n_iops; ++i) {
        // LCG parameters from Numerical Recipes.
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        size_t idx = static_cast<size_t>(rng_state >> 33) % n_positions;
        offsets[i] = static_cast<off_t>(idx * chunk_size);
    }

    // Time the full batch of pread calls.
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    for (size_t i = 0; i < n_iops; ++i) {
        ssize_t nread = pread(fd, buf, chunk_size, offsets[i]);
        (void)nread; // Ignore partial reads for timing purposes.
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    double elapsed_s = (ts_end.tv_sec - ts_start.tv_sec)
                     + (ts_end.tv_nsec - ts_start.tv_nsec) * 1e-9;
    elapsed_s_out = elapsed_s;

    free(buf);
    close(fd);

    double bytes_read = static_cast<double>(n_iops) * static_cast<double>(chunk_size);
    double gib_read   = bytes_read / (1024.0 * 1024.0 * 1024.0);
    return (elapsed_s > 0.0) ? gib_read / elapsed_s : 0.0;
}

// ======================================================
// Median of a vector of doubles (modifies a copy).
// ======================================================
static double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    if (n % 2 == 0) {
        return (values[n / 2 - 1] + values[n / 2]) / 2.0;
    }
    return values[n / 2];
}

// ======================================================
// Quick mode: sweep chunk sizes, find best, write models.conf.
// ======================================================
static int run_quick_mode(const Args& args) {
    std::cout << "scissor-profile (quick mode)\n"
              << "  storage : " << args.storage << "\n"
              << "  file    : " << args.file    << "\n"
              << "  size-gb : " << args.size_gb << "\n"
              << "  output  : " << args.output  << "\n\n";

    size_t file_size = get_file_size_bytes(args.file);
    size_t req_size  = static_cast<size_t>(args.size_gb * 1024.0 * 1024.0 * 1024.0);

    if (req_size > file_size) {
        std::cerr << "Warning: requested size (" << args.size_gb
                  << " GB) exceeds file size ("
                  << file_size / (1024.0 * 1024.0 * 1024.0) << " GB), clamping.\n";
        req_size = file_size;
    }

    // Sweep chunk sizes: 4 KiB to 16 MiB (powers of 4).
    std::vector<size_t> chunk_sizes;
    for (size_t s = 4096; s <= 16 * 1024 * 1024; s *= 4) {
        chunk_sizes.push_back(s);
    }

    double best_throughput = -1.0;
    size_t best_chunk      = chunk_sizes[0];

    std::cout << "Chunk Size (KiB) | Throughput (GiB/s)\n";
    std::cout << "-----------------+-------------------\n";

    for (size_t chunk_size : chunk_sizes) {
        size_t n_chunks = std::min(req_size / chunk_size, size_t{4096});
        if (n_chunks == 0) continue;

        double tput = measure_throughput_quick(args.file, chunk_size, n_chunks);
        std::cout << "  " << std::setw(13) << (chunk_size / 1024)
                  << "  |  " << tput << "\n";
        if (tput > best_throughput) {
            best_throughput = tput;
            best_chunk      = chunk_size;
        }
    }

    std::cout << "\nBest chunk size: " << best_chunk / 1024 << " KiB"
              << " at " << best_throughput << " GiB/s\n";

    // Write output models.conf with best chunk size and built-in default coefficients.
    std::ofstream out_file(args.output);
    if (!out_file.is_open()) {
        std::cerr << "Cannot open output file: " << args.output << "\n";
        return 1;
    }

    out_file << "# scissor/models.conf\n"
             << "# Generated by scissor-profile on " << args.file << "\n"
             << "# Best observed chunk size: " << best_chunk / 1024 << " KiB\n"
             << "# Best observed throughput: " << best_throughput << " GiB/s\n"
             << "# NOTE: io_params coefficients below are the built-in defaults.\n"
             << "# Re-run scissor-profile with a regression fitting tool to obtain\n"
             << "# site-calibrated coefficients.\n"
             << "\n"
             << "[compute]\n"
             << "slope     = 0.748\n"
             << "intercept = -17.0366\n"
             << "\n";

    auto emit_section = [&](const std::string& sec,
                            const std::string& io_params) {
        out_file << "[" << sec << "]\n"
                 << "# Measured best chunk size: " << best_chunk / 1024 << " KiB\n"
                 << "io_params = " << io_params << "\n"
                 << "gaps      = 0, 1, 2, 4, 8, 16\n"
                 << "\n";
    };

    emit_section("pfs",    "4.29e+00, -3.33e-02, 7.26e-06, 4.54e-01, 1.37e-01");
    emit_section("ssd",    "1.13e+01, -5.58e-03, 1.82e-05, 1.64e-01, 7.48e-02");
    emit_section("hybrid", "4.26e+00, -3.47e-02, 4.35e-06, 3.86e-01, -2.72e-02");

    out_file.close();
    std::cout << "Written: " << args.output << "\n";
    return 0;
}

// ======================================================
// Calibrate mode: 2D grid sweep, write CSV.
// ======================================================
static int run_calibrate_mode(const Args& args) {
    std::cout << "scissor-profile (calibrate mode)\n"
              << "  storage    : " << args.storage    << "\n"
              << "  file       : " << args.file       << "\n"
              << "  size-gb    : " << args.size_gb    << "\n"
              << "  reps       : " << args.n_reps     << "\n"
              << "  csv-output : " << args.csv_output << "\n\n";

    size_t file_size = get_file_size_bytes(args.file);
    size_t size_limit = static_cast<size_t>(args.size_gb * 1024.0 * 1024.0 * 1024.0);
    size_t effective_limit = std::min(size_limit, file_size);

    // Grid: chunk_sizes and iop_counts.
    const std::vector<size_t> chunk_sizes = {
        4096, 16384, 65536, 262144, 1048576, 4194304
    };

    std::vector<size_t> iop_counts;
    for (size_t n = 1; n <= 65536; n *= 2) {
        iop_counts.push_back(n);
    }

    // Open CSV.
    std::ofstream csv_file(args.csv_output);
    if (!csv_file.is_open()) {
        std::cerr << "Cannot open CSV output: " << args.csv_output << "\n";
        return 1;
    }
    csv_file << "IOP,Chunk size (B),Total size (B),Load time (s),Throughput (GiB/s)\n";

    // Table header.
    std::cout << std::left
              << std::setw(10) << "IOPs"
              << std::setw(16) << "Chunk (B)"
              << std::setw(18) << "Total size (B)"
              << std::setw(16) << "Load time (s)"
              << "Throughput (GiB/s)\n";
    std::cout << std::string(76, '-') << "\n";

    for (size_t chunk_size : chunk_sizes) {
        for (size_t n_iops : iop_counts) {
            // Skip if total data exceeds limit or file size.
            size_t total_bytes = n_iops * chunk_size;
            if (total_bytes > effective_limit) continue;
            // Also need at least chunk_size bytes in the file.
            if (file_size < chunk_size) continue;

            // Run n_reps trials and collect throughputs and elapsed times.
            std::vector<double> throughputs;
            std::vector<double> elapsed_times;
            throughputs.reserve(args.n_reps);
            elapsed_times.reserve(args.n_reps);

            for (int rep = 0; rep < args.n_reps; ++rep) {
                double elapsed_s = 0.0;
                double tput = measure_throughput_calibrate(
                    args.file, chunk_size, n_iops, file_size,
                    42 + rep, elapsed_s);
                throughputs.push_back(tput);
                elapsed_times.push_back(elapsed_s);
            }

            double median_tput    = median(throughputs);
            double median_elapsed = median(elapsed_times);

            // Write CSV row.
            csv_file << n_iops << ","
                     << chunk_size << ","
                     << total_bytes << ","
                     << std::fixed << std::setprecision(6) << median_elapsed << ","
                     << std::fixed << std::setprecision(6) << median_tput << "\n";

            // Print table row.
            std::cout << std::left << std::defaultfloat
                      << std::setw(10) << n_iops
                      << std::setw(16) << chunk_size
                      << std::setw(18) << total_bytes
                      << std::fixed << std::setprecision(4)
                      << std::setw(16) << median_elapsed
                      << std::setprecision(4) << median_tput << "\n";
        }
    }

    csv_file.close();
    std::cout << "\nCalibration data written to: " << args.csv_output << "\n";
    std::cout << "Run scissor-fit.py to fit models and generate models.conf:\n";
    std::cout << "  python3 scissor_fit.py --profile " << args.csv_output
              << " --storage " << args.storage
              << " --output models.conf\n";
    return 0;
}

// ======================================================
// main
// ======================================================
int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    if (args.calibrate) {
        return run_calibrate_mode(args);
    } else {
        return run_quick_mode(args);
    }
}
