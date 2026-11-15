#include <scissor/regressor.hpp>
#include <scissor/log.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace scissor {

// ======================================================
// Minimal INI parser (internal, no extra dependencies).
//
// Handles:
//   [section]           -- section header
//   key = value         -- key/value pair
//   # comment           -- ignored
//   Blank lines         -- ignored
//
// Returns: section_name -> (key -> raw_value_string)
// ======================================================
static std::map<std::string, std::map<std::string, std::string>>
parse_ini(std::istream& in) {
    std::map<std::string, std::map<std::string, std::string>> result;
    std::string section;
    std::string line;
    while (std::getline(in, line)) {
        // Strip leading whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue; // blank line
        line = line.substr(start);

        if (line.empty() || line[0] == '#') continue; // comment

        if (line[0] == '[') {
            // Section header: [name]
            size_t end = line.find(']');
            if (end != std::string::npos) {
                section = line.substr(1, end - 1);
                // Trim trailing whitespace from section name
                size_t s_end = section.find_last_not_of(" \t");
                if (s_end != std::string::npos) section = section.substr(0, s_end + 1);
            }
        } else {
            // key = value
            size_t eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = line.substr(0, eq);
            std::string val = line.substr(eq + 1);
            // Trim key
            size_t k_end = key.find_last_not_of(" \t");
            if (k_end != std::string::npos) key = key.substr(0, k_end + 1);
            // Trim value leading whitespace
            size_t v_start = val.find_first_not_of(" \t");
            if (v_start != std::string::npos) val = val.substr(v_start);
            // Strip inline comment from value
            size_t hash = val.find('#');
            if (hash != std::string::npos) val = val.substr(0, hash);
            // Trim value trailing whitespace
            size_t v_end = val.find_last_not_of(" \t\r\n");
            if (v_end != std::string::npos) val = val.substr(0, v_end + 1);

            if (!section.empty() && !key.empty()) {
                result[section][key] = val;
            }
        }
    }
    return result;
}

// ======================================================
// Parse a comma-separated list of doubles.
// ======================================================
static std::vector<double> parse_double_array(const std::string& s) {
    std::vector<double> out;
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // Trim whitespace
        size_t start = token.find_first_not_of(" \t");
        size_t end   = token.find_last_not_of(" \t");
        if (start == std::string::npos) continue;
        out.push_back(std::stod(token.substr(start, end - start + 1)));
    }
    return out;
}

// ======================================================
// Parse a comma-separated list of ints.
// ======================================================
static std::vector<int> parse_int_array(const std::string& s) {
    std::vector<int> out;
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        size_t start = token.find_first_not_of(" \t");
        size_t end   = token.find_last_not_of(" \t");
        if (start == std::string::npos) continue;
        out.push_back(std::stoi(token.substr(start, end - start + 1)));
    }
    return out;
}

// ======================================================
// src_section: map FileSrc to INI section name.
// ======================================================
static const char* src_section(FileSrc src) {
    switch (src) {
        case FileSrc::PFS:    return "pfs";
        case FileSrc::SSD:    return "ssd";
        case FileSrc::Hybrid: return "hybrid";
        default:              return "pfs";
    }
}

// ======================================================
// Regressor constructor: layered config loading.
// ======================================================
Regressor::Regressor(FileSrc src, RegressorConfig cfg) {
    // Priority 1: built-in defaults (Polaris calibration).
    switch (src) {
        case FileSrc::PFS:
            io_params_ = {4.29e+00, -3.33e-02, 7.26e-06, 4.54e-01, 1.37e-01};
            break;
        case FileSrc::SSD:
            io_params_ = {1.13e+01, -5.58e-03, 1.82e-05, 1.64e-01, 7.48e-02};
            break;
        case FileSrc::Hybrid:
            io_params_ = {4.26e+00, -3.47e-02, 4.35e-06, 3.86e-01, -2.72e-02};
            break;
        default:
            io_params_ = {4.29e+00, -3.33e-02, 7.26e-06, 4.54e-01, 1.37e-01};
    }
    gaps_               = {0, 1, 2, 4, 8, 16};
    compute_slope_      = 0.748;
    compute_intercept_  = -17.0366;

    // Priority 2: system-wide config at installed prefix.
    bool found_any = false;
#ifdef SCISSOR_DEFAULT_CONFIG_DIR
    {
        std::string sys_path = std::string(SCISSOR_DEFAULT_CONFIG_DIR) + "/models.conf";
        if (load_config(sys_path, src)) {
            SCISSOR_INFO("loaded system models.conf", "path", sys_path.c_str());
            found_any = true;
        }
    }
#endif

    // Priority 3: per-user config at ${HOME}/.scissor/models.conf
    {
        const char* home = std::getenv("HOME");
        if (home && *home) {
            std::string user_path = std::string(home) + "/.scissor/models.conf";
            if (load_config(user_path, src)) {
                SCISSOR_INFO("loaded user models.conf", "path", user_path);
                found_any = true;
            }
        }
    }

    // Priority 4: env var SCISSOR_MODEL_CONFIG
    {
        const char* env_path = std::getenv("SCISSOR_MODEL_CONFIG");
        if (env_path && *env_path) {
            if (load_config(env_path, src)) {
                SCISSOR_INFO("loaded SCISSOR_MODEL_CONFIG", "path", env_path);
                found_any = true;
            } else {
                SCISSOR_WARN("SCISSOR_MODEL_CONFIG set but file not found",
                    "path", env_path);
            }
        }
    }

    if (!found_any) {
        SCISSOR_WARN("no models.conf found at any config path; using built-in defaults");
    }

    // Priority 5: programmatic RegressorConfig overrides.
    if (cfg.io_params.size() == 5) {
        io_params_ = cfg.io_params;
    }
    if (!cfg.gaps.empty()) {
        gaps_ = cfg.gaps;
    }
    if (cfg.compute_slope != 0.0) {
        compute_slope_ = cfg.compute_slope;
    }
    if (cfg.compute_intercept != 0.0) {
        compute_intercept_ = cfg.compute_intercept;
    }

    SCISSOR_INFO("Regressor initialized",
        "src",               src_section(src),
        "compute_slope",     compute_slope_,
        "compute_intercept", compute_intercept_,
        "io_params_C",       io_params_[0],
        "gap_count",         gaps_.size());
}

// ======================================================
// load_config: parse one models.conf and merge relevant
// section.  Returns true if the file was opened.
// ======================================================
bool Regressor::load_config(const std::string& path, FileSrc src) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    auto ini = parse_ini(f);
    const std::string sec = src_section(src);

    // Merge [compute] section if present
    auto it_compute = ini.find("compute");
    if (it_compute != ini.end()) {
        auto& kv = it_compute->second;
        auto k_slope = kv.find("slope");
        if (k_slope != kv.end()) compute_slope_ = std::stod(k_slope->second);
        auto k_int = kv.find("intercept");
        if (k_int != kv.end()) compute_intercept_ = std::stod(k_int->second);
    }

    // Merge storage-tier section if present
    auto it_sec = ini.find(sec);
    if (it_sec != ini.end()) {
        auto& kv = it_sec->second;
        auto k_io = kv.find("io_params");
        if (k_io != kv.end()) {
            auto v = parse_double_array(k_io->second);
            if (v.size() == 5) io_params_ = v;
        }
        auto k_gaps = kv.find("gaps");
        if (k_gaps != kv.end()) {
            auto v = parse_int_array(k_gaps->second);
            if (!v.empty()) gaps_ = v;
        }
    }

    return true;
}

// ======================================================
// Private model helpers (ported from ThroughputOptimizer)
// ======================================================
double Regressor::compute_model(double batch_size) const {
    assert(batch_size > 0.0);
    double log_x = std::log(batch_size);
    return std::exp(compute_intercept_ + compute_slope_ * log_x);
}

double Regressor::io_model(double num_ops, double op_size) const {
    double term_iop_pow = std::pow(num_ops,  io_params_[1]);
    double term_iop_exp = std::exp(-io_params_[2] * num_ops);
    double term_sz_pow  = std::pow(op_size,  io_params_[3]);
    double term_sz_exp  = std::exp(-io_params_[4] * op_size);
    return io_params_[0] * term_iop_pow * term_iop_exp * term_sz_pow * term_sz_exp;
}

std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
Regressor::analyze_contiguity(const std::vector<size_t>& offsets, int gap,
                               size_t chunk_size, size_t batch_size) const {
    if (offsets.empty()) return {{}, {}, {}};

    size_t start             = offsets[0];
    size_t last_offset       = start;
    size_t in_group_offt     = 1;
    size_t wait_for_count    = 0;
    size_t used_chks_per_read = static_cast<size_t>(
        std::ceil(static_cast<double>(batch_size) / chunk_size));

    std::vector<size_t> segments_sizes;
    std::vector<size_t> iop_count_out;
    std::vector<size_t> iop_sizes_out;
    std::vector<size_t> effective_reads_out;
    size_t effective_reads_in_batch = 0;
    size_t diff = static_cast<size_t>(gap);

    for (size_t i = 1; i < offsets.size(); ++i) {
        size_t curr = offsets[i];
        if (curr - last_offset <= diff) {
            last_offset = curr;
            ++in_group_offt;
        } else {
            size_t n_segs = last_offset - start + 1;
            segments_sizes.push_back(n_segs * chunk_size);
            wait_for_count += in_group_offt;
            effective_reads_in_batch += in_group_offt;

            if (wait_for_count >= used_chks_per_read) {
                size_t total =
                    std::accumulate(segments_sizes.begin(), segments_sizes.end(), size_t{0});
                iop_sizes_out.push_back(total);
                iop_count_out.push_back(segments_sizes.size());
                effective_reads_out.push_back(effective_reads_in_batch);
                segments_sizes.clear();
                wait_for_count = 0;
                effective_reads_in_batch = 0;
            }
            start = curr;
            last_offset = curr;
            in_group_offt = 1;
        }
    }
    // Final group
    segments_sizes.push_back((last_offset - start + 1) * chunk_size);
    wait_for_count += in_group_offt;
    effective_reads_in_batch += in_group_offt;
    if (!segments_sizes.empty()) {
        size_t total =
            std::accumulate(segments_sizes.begin(), segments_sizes.end(), size_t{0});
        iop_sizes_out.push_back(total);
        iop_count_out.push_back(segments_sizes.size());
        effective_reads_out.push_back(effective_reads_in_batch);
    }

    return {iop_count_out, iop_sizes_out, effective_reads_out};
}

double Regressor::score(size_t batch_size, const std::vector<size_t>& offsets,
                        int gap, size_t chunk_size, int n_cores) const {
    auto [iop_counts, iop_sizes, eff_reads] =
        analyze_contiguity(offsets, gap, chunk_size, batch_size);

    double total_time = 0.0;
    double total_data_gb =
        std::accumulate(iop_sizes.begin(), iop_sizes.end(), 0.0) / GB_;

    for (size_t i = 0; i < iop_counts.size(); ++i) {
        double work_bytes    = static_cast<double>(eff_reads[i]) * chunk_size;
        double per_core_work = work_bytes / static_cast<double>(n_cores);
        double comp_t        = compute_model(per_core_work);

        double io_size_gb = static_cast<double>(iop_sizes[i]) / GB_;
        double io_thrpt   = io_model(static_cast<double>(iop_counts[i]), io_size_gb);
        double io_t       = io_size_gb / io_thrpt;

        total_time += std::max(comp_t, io_t);
    }
    if (total_time == 0.0) return 0.0;
    return total_data_gb / total_time;
}

// ======================================================
// optimize: grid search over gaps x batch_sizes.
// ======================================================
std::pair<int, size_t>
Regressor::optimize(const std::vector<size_t>& offsets,
                    size_t                     chunk_size,
                    int                        n_cores) const {
    SCISSOR_TIMER_START(opt);

    // Candidate batch sizes: 1 MiB to 1 GiB (powers of 2)
    std::vector<size_t> batch_sizes;
    for (size_t mb = 1; mb <= 1024; mb *= 2) {
        batch_sizes.push_back(mb * 1024 * 1024);
    }

    double best_throughput = -std::numeric_limits<double>::infinity();
    int    best_gap        = -1;
    size_t best_batch      = 0;

    for (int gap : gaps_) {
        for (size_t bs : batch_sizes) {
            double tput = score(bs, offsets, gap, chunk_size, n_cores);
            if (tput > best_throughput) {
                best_throughput = tput;
                best_gap        = gap;
                best_batch      = bs;
            }
        }
    }

    SCISSOR_TIMER_PERF(opt, "optimize complete",
        "best_gap",        best_gap,
        "best_batch_mb",   static_cast<size_t>(best_batch >> 20),
        "best_throughput", best_throughput,
        "n_cores",         n_cores,
        "chunk_size",      chunk_size,
        "offsets",         offsets.size());

    return {best_gap, best_batch};
}

} // namespace scissor
