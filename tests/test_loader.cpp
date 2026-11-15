#include <scissor/scissor.hpp>
// scissor/batch.hpp is included transitively via scissor/loader.hpp

#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <system_error>
#include <unistd.h>
#include <vector>

// ======================================================
// test_loader.cpp
//
// Integration tests for scissor::Loader.
//
// Most tests use a lightweight MockReader that implements
// scissor::IReader and reads directly from seg.fd via pread(),
// which keeps them free of io_uring requirements and exercises
// scissor's full pipeline.  test_shipped_readers runs the same
// two-file pattern through PosixReader and MmapReader.
// ======================================================

using namespace scissor;

// ======================================================
// MockReader: minimal scissor::IReader backed by pread().
// Uses seg.fd from each Segment. The caller owns the fds.
// ======================================================
class MockReader : public IReader {
public:
    int enqueue_reads(const std::vector<Segment>& segs) override {
        // Synchronous: pread each segment immediately
        for (const auto& seg : segs) {
            assert(seg.buffer != nullptr);
            size_t done = 0;
            while (done < seg.size) {
                ssize_t n = pread(seg.fd,
                                  seg.buffer + done,
                                  seg.size - done,
                                  static_cast<off_t>(seg.offset + done));
                if (n <= 0) return -errno;
                done += static_cast<size_t>(n);
            }
        }
        return 0;
    }
    int    wait(size_t /*id*/) override { return 0; }
    size_t wait_any()          override { return 0; }
    int    wait_all()          override { return 0; }
};

// Choose a temp directory that is likely to be on a real filesystem supporting
// O_DIRECT.  Priority: /local/scratch/$USER → /local/scratch → $TMPDIR → /tmp.
// tmpfs (typically $TMPDIR on PBS/SLURM nodes) does not support O_DIRECT.
static std::string choose_tmpdir() {
    const char* user = std::getenv("USER");
    if (user && *user) {
        std::string d = std::string("/local/scratch/") + user;
        if (access(d.c_str(), W_OK) == 0) return d;
    }
    if (access("/local/scratch", W_OK) == 0) return "/local/scratch";
    const char* td = std::getenv("TMPDIR");
    if (td && *td) return std::string(td);
    return "/tmp";
}

// Return true if the filesystem hosting 'path' supports O_DIRECT.
static bool supports_o_direct(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd >= 0) { close(fd); return true; }
    return errno != EINVAL;
}

// Write 'n_chunks' chunks of 'chunk_size' bytes to a temp file.
// Each chunk is filled with its chunk index (byte value, truncated to uint8_t).
// Returns the temp file path.
static std::string make_temp_file(size_t n_chunks, size_t chunk_size) {
    std::string tmpdir = choose_tmpdir();
    std::string path = tmpdir + "/scissor_loader_test_XXXXXX";

    // Use mkstemp for a unique name and posix_memalign for aligned writes
    std::vector<char> buf(path.size() + 1);
    std::memcpy(buf.data(), path.c_str(), path.size() + 1);
    int fd = mkstemp(buf.data());
    assert(fd >= 0 && "mkstemp failed");
    path = std::string(buf.data());
    close(fd);

    // Re-open with O_DIRECT for aligned write
    fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    assert(fd >= 0 && "failed to open temp file for write");

    void* aligned_buf = nullptr;
    int ret = posix_memalign(&aligned_buf, 4096, chunk_size);
    assert(ret == 0 && "posix_memalign failed");

    for (size_t c = 0; c < n_chunks; ++c) {
        uint8_t fill = static_cast<uint8_t>(c & 0xFF);
        memset(aligned_buf, fill, chunk_size);
        ssize_t nw = write(fd, aligned_buf, chunk_size);
        assert(nw == static_cast<ssize_t>(chunk_size) && "short write");
    }
    free(aligned_buf);
    close(fd);
    return path;
}

// ======================================================
// Test 1: single-file sequential load.
// Verify all bytes are read correctly.
// ======================================================
static void test_single_file_sequential() {
    constexpr size_t CHUNK_SIZE  = 4096;
    constexpr size_t N_CHUNKS    = 8;
    constexpr size_t TOTAL_BYTES = CHUNK_SIZE * N_CHUNKS;

    std::string path = make_temp_file(N_CHUNKS, CHUNK_SIZE);

    if (!supports_o_direct(path)) {
        std::cout << "[SKIP] test_single_file_sequential: O_DIRECT not supported on "
                  << path << " (tmpfs or network filesystem)\n";
        remove(path.c_str());
        return;
    }

    MockReader reader;
    Loader loader(reader, {false, 0, 4});
    LoadHandle h = loader.load(path, TOTAL_BYTES, CHUNK_SIZE,
                               TransferType::FileToHost);

    // Collect all batches and verify byte pattern
    size_t total_read = 0;
    size_t chunk_idx  = 0;
    common::Batch* batch = nullptr;
    while ((batch = loader.next(h, TransferType::FileToHost)) != nullptr) {
        for (size_t i = 0; i < batch->count; ++i) {
            const Segment& seg = batch->data[i];
            assert(seg.buffer != nullptr && "segment buffer must be set after I/O");
            uint8_t expected = static_cast<uint8_t>(chunk_idx & 0xFF);
            for (size_t b = 0; b < CHUNK_SIZE; ++b) {
                assert(seg.buffer[b] == expected &&
                       "Read byte does not match expected chunk fill value");
            }
            total_read += seg.size;
            ++chunk_idx;
        }
    }

    assert(total_read == TOTAL_BYTES &&
           "Total bytes read must equal total file size");

    loader.close(h);
    remove(path.c_str());

    std::cout << "[PASS] test_single_file_sequential  ("
              << N_CHUNKS << " chunks x " << CHUNK_SIZE << " B = "
              << TOTAL_BYTES << " B)\n";
}

// ======================================================
// Test 2: dual-file scattered load via FileRequest API.
// Two files identical in content. Load with scattered
// offsets and verify chunk indices match.
// ======================================================
static void test_dual_file_scattered() {
    constexpr size_t CHUNK_SIZE = 4096;
    constexpr size_t N_CHUNKS   = 32;

    std::string path0 = make_temp_file(N_CHUNKS, CHUNK_SIZE);
    std::string path1 = make_temp_file(N_CHUNKS, CHUNK_SIZE);

    if (!supports_o_direct(path0)) {
        std::cout << "[SKIP] test_dual_file_scattered: O_DIRECT not supported on "
                  << path0 << " (tmpfs or network filesystem)\n";
        remove(path0.c_str());
        remove(path1.c_str());
        return;
    }

    // Scattered: every 4th chunk
    std::vector<size_t> offsets;
    for (size_t i = 0; i < N_CHUNKS; i += 4) offsets.push_back(i);

    MockReader reader;
    Loader loader(reader, {false, 0, 8});
    LoadHandle h = loader.load(path0, path1, offsets, CHUNK_SIZE,
                               TransferType::FileToHost,
                               FileSrc::PFS, /*nthreads=*/1);

    size_t batch_count = 0;
    common::Batch* batch = nullptr;
    while ((batch = loader.next(h, TransferType::FileToHost)) != nullptr) {
        ++batch_count;
    }
    assert(batch_count > 0 && "At least one batch must be returned");

    loader.close(h);
    remove(path0.c_str());
    remove(path1.c_str());

    std::cout << "[PASS] test_dual_file_scattered  ("
              << offsets.size() << " offsets, "
              << batch_count << " batches)\n";
}

// ======================================================
// FailingReader: reports an I/O error from wait_all().
// ======================================================
class FailingReader : public IReader {
public:
    int    enqueue_reads(const std::vector<Segment>&) override { return 0; }
    int    wait(size_t)  override { return -EIO; }
    size_t wait_any()    override { return 0; }
    int    wait_all()    override { return -EIO; }
};

// Check that every segment of 'batch' holds the chunk index it was read from.
static void check_batch_contents(const common::Batch* batch, size_t chunk_size) {
    for (size_t i = 0; i < batch->count; ++i) {
        const Segment& seg = batch->data[i];
        for (size_t c = 0; c < seg.size / chunk_size; ++c) {
            uint8_t expected = static_cast<uint8_t>((seg.offset / chunk_size + c) & 0xFF);
            assert(seg.buffer[c * chunk_size] == expected &&
                   "Batch byte does not match the chunk it was read from");
        }
    }
}

// ======================================================
// Test 3: batch size groups segments into batches.
// 16 isolated offsets, no coalescing, 4 chunks per batch:
// 4 batches of 4 segments per file, files interleaved.
// ======================================================
static void test_batch_grouping() {
    constexpr size_t CHUNK_SIZE = 4096;
    constexpr size_t N_CHUNKS   = 64;

    std::string path0 = make_temp_file(N_CHUNKS, CHUNK_SIZE);
    std::string path1 = make_temp_file(N_CHUNKS, CHUNK_SIZE);
    if (!supports_o_direct(path0)) {
        std::cout << "[SKIP] test_batch_grouping: O_DIRECT not supported\n";
        remove(path0.c_str());
        remove(path1.c_str());
        return;
    }

    std::vector<size_t> offsets;
    for (size_t i = 0; i < N_CHUNKS; i += 4) offsets.push_back(i);

    MockReader reader;
    Loader loader(reader, {false, 0, 8});
    LoadHandle h = loader.load(path0, path1, offsets, CHUNK_SIZE,
                               TransferType::FileToHost, FileSrc::PFS,
                               /*nthreads=*/1, /*gap=*/0,
                               /*block_size=*/4 * CHUNK_SIZE);
    assert(loader.iop_count(h) == offsets.size() && "gap=0 must not coalesce");

    size_t batch_count = 0;
    common::Batch* batch = nullptr;
    while ((batch = loader.next(h, TransferType::FileToHost)) != nullptr) {
        assert(batch->count == 8 && "Each batch must hold 4 segments for each of 2 files");
        for (size_t i = 0; i + 1 < batch->count; i += 2) {
            assert(batch->data[i].offset == batch->data[i + 1].offset &&
                   "Reads of the same offset in both files must be adjacent");
            assert(batch->data[i].fd != batch->data[i + 1].fd);
        }
        check_batch_contents(batch, CHUNK_SIZE);
        ++batch_count;
    }
    assert(batch_count == 4 && "16 offsets in batches of 4 must give 4 batches");

    loader.close(h);
    remove(path0.c_str());
    remove(path1.c_str());
    std::cout << "[PASS] test_batch_grouping\n";
}

// ======================================================
// Test 4: invalid arguments and missing files throw.
// ======================================================
static void test_errors_throw() {
    constexpr size_t CHUNK_SIZE = 4096;
    std::string path = make_temp_file(8, CHUNK_SIZE);

    MockReader reader;
    Loader loader(reader, {false, 0, 4});

    bool thrown = false;
    try {
        loader.load("/nonexistent/scissor_missing_file", 8 * CHUNK_SIZE, CHUNK_SIZE,
                    TransferType::FileToHost);
    } catch (const std::system_error& e) {
        thrown = (e.code().value() == ENOENT);
    }
    assert(thrown && "A missing file must throw std::system_error(ENOENT)");

    thrown = false;
    try {
        loader.load(path, path, {0, 1}, /*seg_size=*/1000,
                    TransferType::FileToHost, FileSrc::PFS);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    assert(thrown && "A chunk size that is not a multiple of 4096 must throw");

    thrown = false;
    try {
        loader.load(path, path, {3, 1}, CHUNK_SIZE,
                    TransferType::FileToHost, FileSrc::PFS);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    assert(thrown && "Unsorted offsets must throw");

    if (supports_o_direct(path)) {
        FailingReader failing;
        Loader bad(failing, {false, 0, 4});
        LoadHandle h = bad.load(path, path, {0, 2}, CHUNK_SIZE,
                                TransferType::FileToHost, FileSrc::PFS);
        thrown = false;
        try {
            bad.next(h, TransferType::FileToHost);
        } catch (const std::system_error& e) {
            thrown = (e.code().value() == EIO);
        }
        assert(thrown && "A failed read must make next() throw std::system_error(EIO)");
        bad.close(h);
    }

    remove(path.c_str());
    std::cout << "[PASS] test_errors_throw\n";
}

// ======================================================
// Test 5: closing or destroying with undelivered batches
// returns (no hang) and frees them.
// ======================================================
static void test_close_without_drain() {
    constexpr size_t CHUNK_SIZE = 4096;
    std::string path = make_temp_file(32, CHUNK_SIZE);
    if (!supports_o_direct(path)) {
        std::cout << "[SKIP] test_close_without_drain: O_DIRECT not supported\n";
        remove(path.c_str());
        return;
    }
    std::vector<size_t> offsets = {0, 4, 8, 12, 16, 20};
    {
        MockReader reader;
        Loader loader(reader, {false, 0, 4});
        LoadHandle h = loader.load(path, path, offsets, CHUNK_SIZE,
                                   TransferType::FileToHost, FileSrc::PFS);
        assert(loader.next(h, TransferType::FileToHost) != nullptr);
        loader.close(h);                      // closed after one of six batches
        loader.load(path, path, offsets, CHUNK_SIZE,
                    TransferType::FileToHost, FileSrc::PFS);
    }                                         // destroyed with an untouched handle
    remove(path.c_str());
    std::cout << "[PASS] test_close_without_drain\n";
}

// ======================================================
// Test 6: next() calls on two open handles may be
// interleaved. Each handle's batch stays valid until that
// handle's next call.
// ======================================================
static void test_interleaved_handles() {
    constexpr size_t CHUNK_SIZE = 4096;
    constexpr size_t N_CHUNKS   = 32;
    std::string path = make_temp_file(N_CHUNKS, CHUNK_SIZE);
    if (!supports_o_direct(path)) {
        std::cout << "[SKIP] test_interleaved_handles: O_DIRECT not supported\n";
        remove(path.c_str());
        return;
    }
    std::vector<size_t> even, odd;
    for (size_t i = 0; i < N_CHUNKS; i += 4) { even.push_back(i); odd.push_back(i + 2); }

    MockReader reader;
    Loader loader(reader, {false, 0, 4});
    LoadHandle ha = loader.load(path, path, even, CHUNK_SIZE,
                                TransferType::FileToHost, FileSrc::PFS);
    LoadHandle hb = loader.load(path, path, odd, CHUNK_SIZE,
                                TransferType::FileToHost, FileSrc::PFS);

    size_t na = 0, nb = 0;
    while (true) {
        common::Batch* a = loader.next(ha, TransferType::FileToHost);
        common::Batch* b = loader.next(hb, TransferType::FileToHost);
        if (!a && !b) break;
        if (a) {
            assert(a->data[0].offset == even[na] * CHUNK_SIZE);
            check_batch_contents(a, CHUNK_SIZE);   // still valid after next(hb)
            ++na;
        }
        if (b) {
            assert(b->data[0].offset == odd[nb] * CHUNK_SIZE);
            check_batch_contents(b, CHUNK_SIZE);
            ++nb;
        }
    }
    assert(na == even.size() && nb == odd.size() && "Each handle must deliver all its batches");

    loader.close(ha);
    loader.close(hb);
    remove(path.c_str());
    std::cout << "[PASS] test_interleaved_handles\n";
}

// ======================================================
// Test 7: PosixReader and MmapReader serve both files of a
// two-file load through Segment::fd.
// ======================================================
template <typename Reader>
static void run_shipped_reader(const char* name) {
    constexpr size_t CHUNK_SIZE = 4096;
    constexpr size_t N_CHUNKS   = 64;

    std::string path0 = make_temp_file(N_CHUNKS, CHUNK_SIZE);
    std::string path1 = make_temp_file(N_CHUNKS, CHUNK_SIZE);
    if (!supports_o_direct(path0)) {
        std::cout << "[SKIP] test_shipped_readers (" << name << "): O_DIRECT not supported\n";
        remove(path0.c_str());
        remove(path1.c_str());
        return;
    }

    std::vector<size_t> offsets = {1, 2, 3, 10, 11, 40, 41, 42, 63};

    Reader reader(2);
    Loader loader(reader, {false, 0, 8});
    LoadHandle h = loader.load(path0, path1, offsets, CHUNK_SIZE,
                               TransferType::FileToHost, FileSrc::PFS,
                               /*nthreads=*/1, /*gap=*/1,
                               /*block_size=*/2 * CHUNK_SIZE);
    size_t chunks = 0;
    common::Batch* batch = nullptr;
    while ((batch = loader.next(h, TransferType::FileToHost)) != nullptr) {
        check_batch_contents(batch, CHUNK_SIZE);
        for (size_t i = 0; i < batch->count; ++i)
            chunks += batch->data[i].size / CHUNK_SIZE;
    }
    assert(chunks == 2 * offsets.size() && "Every offset must be read from both files");

    loader.close(h);
    remove(path0.c_str());
    remove(path1.c_str());
    std::cout << "[PASS] test_shipped_readers (" << name << ")\n";
}

static void test_shipped_readers() {
    run_shipped_reader<PosixReader>("PosixReader");
    run_shipped_reader<MmapReader>("MmapReader");
}

int main() {
    std::cout << "=== Loader Tests ===\n";
    test_single_file_sequential();
    test_dual_file_scattered();
    test_batch_grouping();
    test_errors_throw();
    test_close_without_drain();
    test_interleaved_handles();
    test_shipped_readers();
    std::cout << "All Loader tests passed.\n";
    return 0;
}
