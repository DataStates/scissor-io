#ifndef SCISSOR_IO_READER_HPP
#define SCISSOR_IO_READER_HPP

// ======================================================
// scissor/io_reader.hpp
//
// Scissor's own I/O abstractions.  Any I/O backend can
// implement IReader.  The io_uring, POSIX and mmap
// backends shipped with scissor are declared in
// readers.hpp.
// ======================================================

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scissor {

// ======================================================
// Segment: one contiguous read request.
// ======================================================
struct Segment {
    size_t   id;      // caller-assigned identifier
    size_t   offset;  // byte offset within the file
    size_t   size;    // number of bytes to read
    int      fd;      // open file descriptor (O_RDONLY)
    uint8_t* buffer;  // destination buffer (must be 4 KiB aligned for O_DIRECT)
};

// ======================================================
// IReader: pure abstract I/O interface.
//
// Any backend (liburing, POSIX pread pool, mmap, DAOS,
// network, in-memory simulation) implements this interface.
// Callers submit a batch of Segment descriptors, then wait
// for completions.  The interface is deliberately minimal:
// no file paths, no lifecycle management beyond the virtual
// destructor.  File descriptors are owned by the caller
// (typically the Loader's LRU FD cache).
// ======================================================
class IReader {
public:
    // Enqueue a batch of segments for asynchronous reading.
    // Returns 0 on success, negative errno on failure.
    virtual int enqueue_reads(const std::vector<Segment>& segments) = 0;

    // Block until the segment with the given id is complete.
    // Returns 0 on success, negative errno on failure.
    virtual int wait(size_t id) = 0;

    // Block until any one segment completes.
    // Returns the id of the completed segment.
    virtual size_t wait_any() = 0;

    // Block until all currently enqueued segments complete.
    // Returns 0 on success, negative errno on failure.
    virtual int wait_all() = 0;

    virtual ~IReader() = default;
};

} // namespace scissor

#endif // SCISSOR_IO_READER_HPP
