#include <scissor/regressor.hpp>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

// ======================================================
// test_regressor.cpp
//
// Unit tests for scissor::Regressor.
// ======================================================

using namespace scissor;

// ======================================================
// Test 1: no config file present -> built-in defaults loaded.
// (HOME/.scissor/models.conf absent and SCISSOR_MODEL_CONFIG
//  unset by the test environment. An installed models.conf
//  holds the same values.)
// The Regressor must construct without error and return the
// pick of the built-in coefficients.
// ======================================================
static void test_builtin_defaults() {
    // Unset env vars that might pick up a real config
    unsetenv("SCISSOR_MODEL_CONFIG");

    Regressor reg(FileSrc::PFS);

    // optimize() must return valid (non-negative gap, positive batch)
    std::vector<size_t> offsets(64);
    std::iota(offsets.begin(), offsets.end(), 0);
    auto [gap, batch] = reg.optimize(offsets, 4096, 4);

    // Contiguous offsets: the built-in PFS model picks gap 1, 1 MiB batches
    assert(gap == 1 && batch == (size_t(1) << 20) && "built-in PFS pick changed");

    std::cout << "[PASS] test_builtin_defaults  (gap=" << gap
              << ", batch_mb=" << (batch >> 20) << ")\n";
}

// ======================================================
// Test 2: write a temp INI file, set SCISSOR_MODEL_CONFIG,
// verify that custom coefficients are loaded.
// ======================================================
static void test_config_file_loading() {
    // Write a temporary INI file with custom pfs io_params
    const char* tmpdir = std::getenv("TMPDIR");
    if (!tmpdir || !*tmpdir) tmpdir = "/tmp";
    std::string tmppath = std::string(tmpdir) + "/scissor_test_models.conf";

    {
        std::ofstream f(tmppath);
        assert(f.is_open() && "Could not open temp file for write");
        f << "# Test config\n";
        f << "[compute]\n";
        f << "slope     = 1.0\n";
        f << "intercept = -10.0\n";
        f << "[pfs]\n";
        // Distinct values so we can verify they were applied
        f << "io_params = 9.99e+00, -1.00e-02, 1.00e-06, 3.00e-01, 1.00e-01\n";
        f << "gaps      = 0, 2, 4\n";
    }

    setenv("SCISSOR_MODEL_CONFIG", tmppath.c_str(), 1);

    Regressor reg(FileSrc::PFS);

    // optimize() should work with the custom coefficients
    std::vector<size_t> offsets(32);
    std::iota(offsets.begin(), offsets.end(), 0);
    auto [gap, batch] = reg.optimize(offsets, 4096, 4);

    // The custom coefficients and gap set {0, 2, 4} give gap 0, 1 MiB batches
    assert(gap == 0 && batch == (size_t(1) << 20) && "custom config pick changed");

    unsetenv("SCISSOR_MODEL_CONFIG");
    remove(tmppath.c_str());

    std::cout << "[PASS] test_config_file_loading  (gap=" << gap
              << ", batch_mb=" << (batch >> 20) << ")\n";
}

// ======================================================
// Test 3: programmatic RegressorConfig overrides file values.
// ======================================================
static void test_programmatic_override() {
    unsetenv("SCISSOR_MODEL_CONFIG");

    RegressorConfig cfg;
    cfg.io_params        = {5.0, -0.01, 0.0, 0.5, 0.1};
    cfg.gaps             = {0, 4, 8};
    cfg.compute_slope    = 0.9;
    cfg.compute_intercept = -15.0;

    Regressor reg(FileSrc::SSD, cfg);

    std::vector<size_t> offsets(16);
    std::iota(offsets.begin(), offsets.end(), 0);
    auto [gap, batch] = reg.optimize(offsets, 4096, 2);

    // The gaps come from the programmatic set {0, 4, 8}
    assert(gap == 0 && batch == (size_t(1) << 20) &&
           "programmatic gaps {0, 4, 8} must give gap 0, 1 MiB batches");

    std::cout << "[PASS] test_programmatic_override  (gap=" << gap
              << ", batch_mb=" << (batch >> 20) << ")\n";
}

// ======================================================
// Test 4: optimize() returns gap >= 0 and batch > 0 for
// a known scattered offset distribution.
// ======================================================
static void test_optimize_scattered() {
    unsetenv("SCISSOR_MODEL_CONFIG");

    Regressor reg(FileSrc::Hybrid);

    // Scattered offsets: every 8th chunk
    std::vector<size_t> offsets;
    for (size_t i = 0; i < 256; i += 8) offsets.push_back(i);

    auto [gap, batch] = reg.optimize(offsets, 4096, 8);

    // Every 8th chunk: gap 8 merges all offsets into contiguous reads
    assert(gap == 8 && batch == (size_t(1) << 20) && "built-in hybrid pick changed");

    std::cout << "[PASS] test_optimize_scattered  (gap=" << gap
              << ", batch_mb=" << (batch >> 20) << ")\n";
}

int main() {
    std::cout << "=== Regressor Tests ===\n";
    test_builtin_defaults();
    test_config_file_loading();
    test_programmatic_override();
    test_optimize_scattered();
    std::cout << "All Regressor tests passed.\n";
    return 0;
}
