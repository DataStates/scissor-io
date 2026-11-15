#include <scissor/aggregator.hpp>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

// ======================================================
// test_aggregator.cpp
//
// Unit tests for scissor::Aggregator::coalesce().
// No I/O is performed, as the Aggregator is pure.
// ======================================================

using namespace scissor;

// Helper: construct a sorted offset list of n sequential chunks
static std::vector<size_t> seq_offsets(size_t n, size_t start = 0) {
    std::vector<size_t> v(n);
    std::iota(v.begin(), v.end(), start);
    return v;
}

// ======================================================
// Test 1: gap=0 disables coalescing: contiguous offsets
// stay one segment per offset.
// ======================================================
static void test_contiguous_no_gap() {
    Aggregator agg;
    size_t seg_size = 4096;
    auto offsets = seq_offsets(16);  // 0..15

    auto result = agg.coalesce(offsets, seg_size, /*gap=*/0);

    assert(result.segments.size() == 16 &&
           "gap=0 must produce one segment per offset");
    assert(result.iop_count == 16 &&
           "iop_count must equal number of segments");
    assert(result.wasted_bytes == 0 &&
           "No gaps: wasted_bytes must be 0");
    for (size_t i = 0; i < 16; ++i) {
        assert(result.segments[i].offset == i * seg_size &&
               result.segments[i].size == seg_size &&
               "Each segment must cover exactly its own chunk");
    }
    assert(result.covered_offsets.size() == 16 &&
           "All 16 offsets must be covered");

    std::cout << "[PASS] test_contiguous_no_gap\n";
}

// ======================================================
// Test 1b: gap=1 merges adjacent offsets only.
// ======================================================
static void test_contiguous_gap_one() {
    Aggregator agg;
    size_t seg_size = 4096;
    // 0..15 contiguous, then 17 (one unrequested chunk in between)
    auto offsets = seq_offsets(16);
    offsets.push_back(17);

    auto result = agg.coalesce(offsets, seg_size, /*gap=*/1);

    assert(result.segments.size() == 2 &&
           "gap=1 must merge 0..15 and keep 17 separate");
    assert(result.segments[0].offset == 0 &&
           result.segments[0].size == 16 * seg_size &&
           "First segment must span chunks 0..15");
    assert(result.segments[1].offset == 17 * seg_size &&
           result.segments[1].size == seg_size &&
           "Second segment must cover chunk 17 only");
    assert(result.wasted_bytes == 0 &&
           "gap=1 never reads unrequested chunks");

    std::cout << "[PASS] test_contiguous_gap_one\n";
}

// ======================================================
// Test 1c: segments are grouped into batches of at least
// used_chks_per_read requested chunks.
// ======================================================
static void test_batch_grouping() {
    Aggregator agg;
    size_t seg_size = 4096;
    // 10 isolated offsets, no coalescing, batches of >= 4 chunks
    std::vector<size_t> offsets = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};

    auto result = agg.coalesce(offsets, seg_size, /*gap=*/0, /*used_chks_per_read=*/4);

    assert(result.segments.size() == 10);
    assert((result.batch_ends   == std::vector<size_t>{4, 8, 10}) &&
           "Batches must close after 4, 8 and 10 segments");
    assert((result.batch_chunks == std::vector<size_t>{4, 4, 2}) &&
           "Batch chunk counts must be 4, 4 and the remaining 2");

    // Coalesced groups count by requested chunks: {0,1,2} {10,11} {20,21,22}
    std::vector<size_t> grouped = {0, 1, 2, 10, 11, 20, 21, 22};
    auto r2 = agg.coalesce(grouped, seg_size, /*gap=*/1, /*used_chks_per_read=*/4);
    assert(r2.segments.size() == 3);
    assert((r2.batch_ends   == std::vector<size_t>{2, 3}) &&
           "First batch closes once 5 >= 4 chunks are covered");
    assert((r2.batch_chunks == std::vector<size_t>{5, 3}));

    std::cout << "[PASS] test_batch_grouping\n";
}

// ======================================================
// Test 2: two groups separated by a gap > gap threshold
// produce 2 segments.
// ======================================================
static void test_two_groups_with_gap() {
    Aggregator agg;
    size_t seg_size = 4096;
    // Offsets 0..7 then 20..27 (gap of 12 chunks with gap threshold=2)
    std::vector<size_t> offsets;
    for (size_t i = 0; i < 8; ++i)  offsets.push_back(i);
    for (size_t i = 20; i < 28; ++i) offsets.push_back(i);

    auto result = agg.coalesce(offsets, seg_size, /*gap=*/2);

    assert(result.segments.size() == 2 &&
           "Gap > threshold must split into 2 segments");
    assert(result.iop_count == 2 &&
           "iop_count must equal segment count");

    // First segment: chunks 0..7
    assert(result.segments[0].offset == 0);
    assert(result.segments[0].size == 8 * seg_size);

    // Second segment: chunks 20..27
    assert(result.segments[1].offset == 20 * seg_size);
    assert(result.segments[1].size == 8 * seg_size);

    std::cout << "[PASS] test_two_groups_with_gap\n";
}

// ======================================================
// Test 3: wasted_bytes == (gap_fill_chunks * seg_size) per group.
// ======================================================
static void test_wasted_bytes() {
    Aggregator agg;
    size_t seg_size = 4096;
    // Offsets 0 and 2 (gap of 2 chunks, threshold gap=2 merges them)
    // The read will cover [0,1,2] => 3 chunks but only 2 are requested
    std::vector<size_t> offsets = {0, 2};

    auto result = agg.coalesce(offsets, seg_size, /*gap=*/2);

    assert(result.segments.size() == 1 &&
           "Gap <= threshold must merge into 1 segment");
    // Wasted: 1 chunk (chunk 1 is in the gap)
    assert(result.wasted_bytes == 1 * seg_size &&
           "Wasted bytes must equal gap fill chunks * seg_size");

    std::cout << "[PASS] test_wasted_bytes\n";
}

// ======================================================
// Test 4: iop_count matches number of coalesced segments.
// ======================================================
static void test_iop_count_matches_segments() {
    Aggregator agg;
    size_t seg_size = 4096;
    // Create 4 isolated chunks with gaps of 10 between each (threshold=3)
    std::vector<size_t> offsets = {0, 10, 20, 30};

    auto result = agg.coalesce(offsets, seg_size, /*gap=*/3);

    assert(result.iop_count == result.segments.size() &&
           "iop_count must always equal segments.size()");
    assert(result.segments.size() == 4 &&
           "4 isolated chunks with gap>threshold must produce 4 segments");

    std::cout << "[PASS] test_iop_count_matches_segments\n";
}

// ======================================================
// Test 5: empty offset list produces empty result.
// ======================================================
static void test_empty_offsets() {
    Aggregator agg;
    auto result = agg.coalesce({}, 4096, 0);

    assert(result.segments.empty()       && "Empty input: no segments");
    assert(result.covered_offsets.empty() && "Empty input: no covered offsets");
    assert(result.iop_count == 0         && "Empty input: iop_count=0");
    assert(result.wasted_bytes == 0      && "Empty input: wasted_bytes=0");

    std::cout << "[PASS] test_empty_offsets\n";
}

int main() {
    std::cout << "=== Aggregator Tests ===\n";
    test_contiguous_no_gap();
    test_contiguous_gap_one();
    test_batch_grouping();
    test_two_groups_with_gap();
    test_wasted_bytes();
    test_iop_count_matches_segments();
    test_empty_offsets();
    std::cout << "All Aggregator tests passed.\n";
    return 0;
}
