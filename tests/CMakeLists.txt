cmake_minimum_required(VERSION 3.14)

# ======================================================
# scissor tests
#
# The tests check their results with assert(), so NDEBUG
# is undefined for them regardless of the build type.
# ======================================================

# test_aggregator: unit test for Aggregator::coalesce().
add_executable(test_aggregator test_aggregator.cpp)
target_link_libraries(test_aggregator PRIVATE scissor)
target_compile_options(test_aggregator PRIVATE -UNDEBUG)
set_target_properties(test_aggregator PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS OFF)
add_test(NAME AggregatorTests COMMAND test_aggregator)

# test_regressor: unit test for Regressor config loading and optimize().
add_executable(test_regressor test_regressor.cpp)
target_link_libraries(test_regressor PRIVATE scissor)
target_compile_options(test_regressor PRIVATE -UNDEBUG)
set_target_properties(test_regressor PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS OFF)
add_test(NAME RegressorTests COMMAND test_regressor)
# Only the built-in defaults and the installed models.conf (same values) are
# read: no per-user config, no SCISSOR_MODEL_CONFIG from the environment.
set_tests_properties(RegressorTests PROPERTIES ENVIRONMENT
    "HOME=${CMAKE_CURRENT_BINARY_DIR}/no-home;SCISSOR_MODEL_CONFIG=")

# test_loader: integration test of the Loader pipeline using a
# pread-based MockReader (no io_uring required).
add_executable(test_loader test_loader.cpp)
target_link_libraries(test_loader PRIVATE scissor)
target_compile_options(test_loader PRIVATE -UNDEBUG)
set_target_properties(test_loader PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS OFF)
add_test(NAME LoaderTests COMMAND test_loader)

# test_readers: end-of-file reads, read errors and truncated files
# through the shipped readers (io_uring cases skip without io_uring).
add_executable(test_readers test_readers.cpp)
target_link_libraries(test_readers PRIVATE scissor)
target_compile_options(test_readers PRIVATE -UNDEBUG)
set_target_properties(test_readers PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS OFF)
add_test(NAME ReaderTests COMMAND test_readers)

# A hang (e.g. in close() or the destructor) fails the test instead of blocking CI.
set_tests_properties(AggregatorTests RegressorTests LoaderTests ReaderTests PROPERTIES TIMEOUT 120)
