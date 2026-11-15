#include <scissor/readers.hpp>

#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

// ======================================================
// test_readers.cpp
//
// Unit tests for the shipped readers: reads stopping at the
// end of the file, read errors reported by wait() and
// wait_all(), and MmapReader reads of a file truncated after
// it was mapped.  The files are opened without O_DIRECT, so any
// filesystem works.  LiburingReader tests are skipped when
// io_uring is unavailable.
// ======================================================

using namespace scissor;

constexpr size_t CHUNK = 4096;
constexpr uint8_t UNTOUCHED = 0xEE;

static uint8_t pattern(size_t pos) { return static_cast<uint8_t>(pos % 251); }

// Write 'size' bytes of pattern() to a temp file and return its path
static std::string make_file(size_t size) {
    const char* td = std::getenv("TMPDIR");
    std::string path = std::string(td && *td ? td : "/tmp") + "/scissor_reader_test_XXXXXX";
    std::vector<char> name(path.begin(), path.end());
    name.push_back('\0');
    int fd = mkstemp(name.data());
    assert(fd >= 0 && "mkstemp failed");
    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < size; i++)
        data[i] = pattern(i);
    assert(write(fd, data.data(), size) == static_cast<ssize_t>(size) && "short write");
    close(fd);
    return std::string(name.data());
}

// Bytes [0, valid) of buf hold the file content at 'offset', the rest is untouched
static bool check(const std::vector<uint8_t>& buf, size_t offset, size_t valid) {
    for (size_t i = 0; i < buf.size(); i++) {
        uint8_t expected = (i < valid) ? pattern(offset + i) : UNTOUCHED;
        if (buf[i] != expected)
            return false;
    }
    return true;
}

static std::unique_ptr<IReader> make_reader(const std::string& name) {
    if (name == "PosixReader")
        return std::make_unique<PosixReader>(2);
    if (name == "MmapReader")
        return std::make_unique<MmapReader>(2);
    try {
        return std::make_unique<LiburingReader>(64, 16);
    } catch (const std::system_error& e) {
        std::cout << "[SKIP] LiburingReader: " << e.what() << "\n";
        return nullptr;
    }
}

// ======================================================
// Test 1: a segment crossing the end of the file is read up
// to the end, a segment past the end reads nothing, and
// neither is an error.
// ======================================================
static void test_end_of_file(const std::string& name) {
    auto reader = make_reader(name);
    if (!reader) return;
    const size_t fsize = 3 * CHUNK + 100;
    std::string path = make_file(fsize);
    int fd = open(path.c_str(), O_RDONLY);
    assert(fd >= 0);

    std::vector<std::vector<uint8_t>> bufs(3, std::vector<uint8_t>(CHUNK, UNTOUCHED));
    std::vector<Segment> segs = {
        {0, 0,         CHUNK, fd, bufs[0].data()},
        {1, 3 * CHUNK, CHUNK, fd, bufs[1].data()},  // 100 bytes left
        {2, 4 * CHUNK, CHUNK, fd, bufs[2].data()},  // past the end
    };
    assert(reader->enqueue_reads(segs) == 0);
    assert(reader->wait_all() == 0 && "Reaching the end of the file is not an error");
    assert(check(bufs[0], 0, CHUNK));
    assert(check(bufs[1], 3 * CHUNK, 100) && "Read must stop at the end of the file");
    assert(check(bufs[2], 4 * CHUNK, 0) && "Read past the end must leave the buffer untouched");

    close(fd);
    remove(path.c_str());
    std::cout << "[PASS] test_end_of_file (" << name << ")\n";
}

// ======================================================
// Test 2: a failed read is returned by wait() for its id and
// by the next wait_all(), the other segments succeed.
// ======================================================
static void test_wait_errors(const std::string& name) {
    auto reader = make_reader(name);
    if (!reader) return;
    std::string path = make_file(2 * CHUNK);
    int fd = open(path.c_str(), O_RDONLY);
    assert(fd >= 0);

    std::vector<uint8_t> good(CHUNK, UNTOUCHED), bad(CHUNK, UNTOUCHED);
    std::vector<Segment> segs = {
        {3, CHUNK, CHUNK, fd, good.data()},
        {7, 0,     CHUNK, -1, bad.data()},  // no file
    };
    assert(reader->enqueue_reads(segs) == 0);
    assert(reader->wait(7) == -EBADF && "wait() must return the segment's error");
    assert(reader->wait(3) == 0);
    assert(check(good, CHUNK, CHUNK));
    assert(reader->wait_all() == -EBADF && "wait_all() must report the failed read");
    assert(reader->wait_all() == 0 && "wait_all() reports a failure once");

    close(fd);
    remove(path.c_str());
    std::cout << "[PASS] test_wait_errors (" << name << ")\n";
}

// ======================================================
// Test 3: MmapReader reads a file truncated after it was
// mapped without touching pages past the new end.
// ======================================================
static void test_mmap_truncated() {
    MmapReader reader(1);
    std::string path = make_file(4 * CHUNK);
    int fd = open(path.c_str(), O_RDONLY);
    assert(fd >= 0);

    std::vector<uint8_t> first(CHUNK, UNTOUCHED);
    std::vector<Segment> segs = {{0, 2 * CHUNK, CHUNK, fd, first.data()}};
    assert(reader.enqueue_reads(segs) == 0);
    assert(reader.wait_all() == 0);
    assert(check(first, 2 * CHUNK, CHUNK));

    assert(truncate(path.c_str(), CHUNK + 10) == 0);
    std::vector<uint8_t> head(CHUNK, UNTOUCHED), cut(CHUNK, UNTOUCHED), gone(CHUNK, UNTOUCHED);
    segs = {
        {1, 0,         CHUNK, fd, head.data()},
        {2, CHUNK,     CHUNK, fd, cut.data()},   // 10 bytes left
        {3, 2 * CHUNK, CHUNK, fd, gone.data()},  // removed by the truncation
    };
    assert(reader.enqueue_reads(segs) == 0);
    assert(reader.wait_all() == 0);
    assert(check(head, 0, CHUNK));
    assert(check(cut, CHUNK, 10) && "Read must stop at the new end of the file");
    assert(check(gone, 2 * CHUNK, 0) && "Truncated pages must not be read");

    close(fd);
    remove(path.c_str());
    std::cout << "[PASS] test_mmap_truncated\n";
}

int main() {
    std::cout << "=== Reader Tests ===\n";
    for (const char* name : {"LiburingReader", "PosixReader", "MmapReader"})
        test_end_of_file(name);
    for (const char* name : {"LiburingReader", "PosixReader"})
        test_wait_errors(name);
    test_mmap_truncated();
    std::cout << "All Reader tests passed.\n";
    return 0;
}
