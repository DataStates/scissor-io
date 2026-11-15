#ifndef SCISSOR_READERS_HPP
#define SCISSOR_READERS_HPP

// ======================================================
// scissor/readers.hpp
//
// I/O backends implementing scissor::IReader:
//   LiburingReader -- io_uring (SQPOLL) asynchronous reads
//   PosixReader    -- pread() thread pool
//   MmapReader     -- mmap() + memcpy thread pool (MADV_RANDOM)
//
// All readers read from Segment::fd, so one instance can serve
// segments from many files, and identify segments by
// Segment::id in wait()/wait_n()/wait_any().  PosixReader and
// MmapReader constructed with a file name also serve segments
// whose fd is -1 from that file.  Call wait_all() before
// enqueueing the next group of segments on PosixReader and
// MmapReader.
//
// Errors: constructors throw std::system_error when the ring,
// file or mapping cannot be set up.  wait(id) returns the
// negative errno of that segment's failed read, 0 otherwise.
// wait_all() returns the negative errno of the first failed
// read since the previous wait_all(), 0 otherwise.
// ======================================================

#include <scissor/io_reader.hpp>

#include <liburing.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <sys/types.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace scissor {

class LiburingReader : public IReader {
  size_t MAX_RING_SIZE = DEFAULT_RING_SIZE;
  size_t BATCH_SIZE = DEFAULT_BATCH_SIZE;
  size_t req_submitted, req_completed, req_consumed;
  std::queue<Segment> submissions;
  std::unordered_set<size_t> completions;
  io_uring ring;
  struct io_uring_params ring_params{};
  int first_error = 0; // negative errno of the first failed read since the last wait_all()
  std::unordered_map<size_t, int> failed; // negative errno of failed ids not yet waited for

  uint32_t request_completion();
  uint32_t submit_more();

  public:
    static constexpr size_t DEFAULT_RING_SIZE  = 32768;
    static constexpr size_t DEFAULT_BATCH_SIZE = 1024;

    LiburingReader();
    // qdepth: io_uring ring entries, cqdepth: completion batch size
    LiburingReader(size_t qdepth, size_t cqdepth);
    ~LiburingReader() override;
    int enqueue_reads(const std::vector<Segment>& segments) override; // Add segments to read queue
    int wait(size_t id) override; // Wait for id to finish
    int wait_all() override; // wait for all pending reads to finish
    size_t wait_any() override; // wait for any available read to finish
    std::vector<size_t> wait_n(size_t n=1); // wait for any N available reads to finish
};

class PosixReader : public IReader {
  size_t fsize = 0;
  int fd = -1, num_threads = 1;
  std::string fname;
  std::vector<Segment> reads;
  std::vector<bool> segment_status;
  std::vector<int> segment_error; // negative errno of each segment's read, 0 on success
  std::vector<std::future<int>> futures;
  size_t completed_count=0, submitted_count=0;
  std::mutex status_mutex;
  std::atomic<int> first_error{0}; // negative errno of the first failed read since the last wait_all()

  int read_data(size_t beg, size_t end);

  public:
    explicit PosixReader(size_t nthreads=1); // read through Segment::fd
    PosixReader(const std::string& name, size_t nthreads=1); // open file
    ~PosixReader() override;
    int enqueue_reads(const std::vector<Segment>& segments) override; // Add segments to read queue
    int wait(size_t id) override; // Wait for id to finish
    std::vector<size_t> wait_n(size_t n=1); // wait for any N available reads to finish
    int wait_all() override; // wait for all pending reads to finish
    size_t wait_any() override; // wait for any available read to finish
    size_t size() { return fsize; } // retrieve size of the file to read
};

class MmapReader : public IReader {
  size_t fsize = 0;
  int fd = -1, num_threads = 1;
  std::string fname;
  uint8_t* buffer = nullptr;
  std::vector<Segment> reads;
  std::vector<const uint8_t*> sources; // mapping each queued segment is copied from
  std::map<std::pair<dev_t, ino_t>, std::pair<uint8_t*, size_t>> mappings; // files mapped through Segment::fd
  std::vector<bool> segment_status;
  std::vector<std::future<int>> futures;
  size_t completed_count=0, submitted_count=0;
  std::mutex status_mutex;
  std::atomic<int> first_error{0}; // negative errno of the first failed read since the last wait_all()

  int read_data(size_t beg, size_t end);

  public:
    explicit MmapReader(size_t nthreads=1); // read through Segment::fd
    MmapReader(const std::string& name, size_t nthreads=1); // open file
    ~MmapReader() override;
    int enqueue_reads(const std::vector<Segment>& segments) override; // Add segments to read queue
    int wait(size_t id) override; // Wait for id to finish
    std::vector<size_t> wait_n(size_t n=1); // wait for any N available reads to finish
    int wait_all() override; // wait for all pending reads to finish
    size_t wait_any() override; // wait for any available read to finish
    size_t size() { return fsize; } // retrieve size of the file to read
};

} // namespace scissor

#endif // SCISSOR_READERS_HPP
