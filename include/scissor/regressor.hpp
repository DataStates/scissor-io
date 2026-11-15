#ifndef SCISSOR_REGRESSOR_HPP
#define SCISSOR_REGRESSOR_HPP

// ======================================================
// scissor/regressor.hpp
//
// Regressor: throughput model for I/O + compute overlap.
//
// Ported from ThroughputOptimizer in state-diff.  Adds:
//   - Layered config file loading (INI) so coefficients
//     can be tuned per site without recompiling.
//   - Programmatic override via RegressorConfig fields.
//   - SCISSOR_* logging at construction and optimize() exit.
//
// Config loading priority (lowest to highest):
//   1. Built-in defaults (Polaris/Lustre coefficients)
//   2. ${SCISSOR_DEFAULT_CONFIG_DIR}/models.conf
//   3. ${HOME}/.scissor/models.conf
//   4. Path at env var SCISSOR_MODEL_CONFIG
//   5. RegressorConfig fields (if non-empty)
// ======================================================

#include <scissor/types.hpp>
#include <array>
#include <vector>
#include <string>
#include <cstddef>

namespace scissor {

// ======================================================
// RegressorConfig: programmatic override for model params.
// Any field left at its zero/empty default is not applied.
// ======================================================
struct RegressorConfig {
    // io_params: [C, alpha, beta, gamma, delta].
    // Length must be exactly 5 or 0 (meaning "use default/file").
    std::vector<double> io_params;

    // gaps: candidate gap values to explore in optimize().
    // Empty means "use default/file".
    std::vector<int> gaps;

    // Compute model: log-linear coefficients.
    // 0.0 means "use default/file".
    double compute_slope     = 0.0;
    double compute_intercept = 0.0;
};

// ======================================================
// Regressor: throughput model for gap + batch_size search.
// ======================================================
class Regressor {
public:
    // Construct a Regressor for the given storage tier.
    // Reads config files in the layered order described above.
    // Applies 'cfg' fields as the highest-priority override.
    explicit Regressor(FileSrc src, RegressorConfig cfg = {});

    // optimize: search (gap, batch_size) space for best throughput.
    //
    // Parameters:
    //   offsets    -- logical chunk offsets (sorted ascending)
    //   chunk_size -- size of one chunk in bytes
    //   n_cores    -- number of compute threads (for compute model)
    //
    // Returns {best_gap, best_batch_size_bytes}.
    std::pair<int, size_t> optimize(const std::vector<size_t>& offsets,
                                    size_t                     chunk_size,
                                    int                        n_cores) const;

private:
    // IO model: throughput = C * N^alpha * exp(-beta*N) * S^gamma * exp(-delta*S)
    // where N = number of IOPs and S = op size in GB.
    double io_model(double num_ops, double op_size) const;

    // Compute model: time = exp(intercept + slope * log(bytes_per_core))
    double compute_model(double batch_size) const;

    // Score a (batch_size, gap) combination against the offset distribution.
    double score(size_t batch_size, const std::vector<size_t>& offsets,
                 int gap, size_t chunk_size, int n_cores) const;

    // Analyze how offsets coalesce under a given gap and batch_size.
    // Returns (IOP_counts, iop_sizes, effective_reads) per batch.
    std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
    analyze_contiguity(const std::vector<size_t>& offsets, int gap,
                       size_t chunk_size, size_t batch_size) const;

    // Load a models.conf INI file and merge the relevant section into
    // current params.  Returns true if the file was found and parsed.
    bool load_config(const std::string& path, FileSrc src);

    // Conversion constant: bytes to GiB
    static constexpr double GB_ = 1024.0 * 1024.0 * 1024.0;

    double compute_slope_;
    double compute_intercept_;
    std::vector<double> io_params_;   // [C, alpha, beta, gamma, delta]
    std::vector<int>    gaps_;        // candidate gap values
};

} // namespace scissor

#endif // SCISSOR_REGRESSOR_HPP
