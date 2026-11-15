#include <scissor/readers.hpp>
#include <scissor/log.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <system_error>
#include <thread>
#include <unistd.h>

namespace scissor {

PosixReader::PosixReader(size_t nthreads) {
    num_threads = nthreads;
}

PosixReader::~PosixReader() {
    if (fd >= 0)
        close(fd);
}

PosixReader::PosixReader(const std::string& name, size_t nthreads) {
    num_threads = nthreads;
    fname       = name;
    fd          = open(name.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::system_error(errno, std::generic_category(),
                                "scissor: cannot open " + fname);
    }
    fsize = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
}

int PosixReader::read_data(size_t beg, size_t end) {
    SCISSOR_TIMER_START(read);
    for (size_t i = beg; i < end; i++) {
        Segment& seg = reads[i];
        size_t len = 0;
        int seg_error = 0;

        // A read reaching the end of the file stops the segment there, as
        // with io_uring. Sizes are left unclamped so that O_DIRECT file
        // descriptors keep aligned request lengths.
        while (len < seg.size) {
            ssize_t data_read = pread(seg.fd,
                                      static_cast<void*>(seg.buffer + len),
                                      seg.size - len,
                                      static_cast<off_t>(seg.offset + len));
            if (data_read == 0)
                break;
            if (data_read < 0) {
                int err = errno;
                SCISSOR_ERROR("pread failed",
                              "fd",          seg.fd,
                              "buf_offset",  len,
                              "size",        seg.size - len,
                              "file_offset", seg.offset + len,
                              "error",       std::strerror(err),
                              "errno",       err);
                seg_error = -err;
                int expected = 0;
                first_error.compare_exchange_strong(expected, -err);
                break;
            } else {
                len += data_read;
            }
        }

        status_mutex.lock();
        segment_error[i] = seg_error;
        segment_status[i] = true;
        status_mutex.unlock();
    }
    SCISSOR_TIMER_PERF(read, "Read segments",
                       "beg", beg,
                       "end", end);
    return 0;
}

int PosixReader::enqueue_reads(const std::vector<Segment>& segments) {
    SCISSOR_TIMER_START(enqueue);
    // Segments with fd -1 are read from the file given to the constructor
    std::vector<Segment> resolved(segments.begin(), segments.end());
    for (Segment& seg : resolved) {
        if (seg.fd < 0)
            seg.fd = fd;
    }
    status_mutex.lock();
    segment_status.insert(segment_status.end(), resolved.size(), false);
    segment_error.insert(segment_error.end(), resolved.size(), 0);
    status_mutex.unlock();
    reads.insert(reads.end(), resolved.begin(), resolved.end());
    SCISSOR_TIMER_PERF(enqueue, "Enqueue segments",
                       "count", segments.size());

    SCISSOR_TIMER_START(launch_threads);
    size_t per_thread = segments.size() / num_threads;
    if (per_thread * num_threads < segments.size())
        per_thread += 1;
    for (size_t i = 0; i < (size_t)num_threads; i++) {
        size_t beg = i * per_thread;
        size_t end = (i + 1) * per_thread;
        if (end > segments.size())
            end = segments.size();
        futures.push_back(std::async(std::launch::async | std::launch::deferred,
                                     &PosixReader::read_data, this, beg, end));
        submitted_count += end - beg;
    }
    SCISSOR_TIMER_PERF(launch_threads, "Launch threads for reading",
                       "num_threads", num_threads);
    return 0;
}

int PosixReader::wait(size_t id) {
    size_t pos = 0;
    for (pos = 0; pos < reads.size(); pos++) {
        if (reads[pos].id == id)
            break;
    }
    if (pos == reads.size())
        return -1;
    while (segment_status[pos] == false) {
        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(10));
    }
    status_mutex.lock();
    segment_status[pos] = false;
    int err = segment_error[pos];
    status_mutex.unlock();
    completed_count++;
    return err;
}

int PosixReader::wait_all() {
    for (uint32_t i = 0; i < futures.size(); i++) {
        futures[i].get();
    }
    completed_count = submitted_count;
    futures.clear();
    reads.clear();
    status_mutex.lock();
    segment_status.clear();
    segment_error.clear();
    status_mutex.unlock();
    return first_error.exchange(0);
}

std::vector<size_t> PosixReader::wait_n(size_t n) {
    size_t nwait = std::min(n, submitted_count - completed_count);
    std::vector<size_t> ids;
    size_t finished = 0;
    while (finished < nwait) {
        for (size_t pos = 0; pos < segment_status.size(); pos++) {
            if (segment_status[pos]) {
                status_mutex.lock();
                segment_status[pos] = false;
                status_mutex.unlock();
                finished++;
                completed_count++;
                ids.push_back(reads[pos].id);
                if (finished >= nwait) {
                    return ids;
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(10));
    }
    return ids;
}

size_t PosixReader::wait_any() {
    bool finished = false;
    while (!finished) {
        for (size_t pos = 0; pos < segment_status.size(); pos++) {
            if (segment_status[pos]) {
                status_mutex.lock();
                segment_status[pos] = false;
                status_mutex.unlock();
                completed_count++;
                finished = true;
                return reads[pos].id;
            }
        }
        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(10));
    }
    SCISSOR_FATAL("wait_any: no segment became ready, should have returned earlier");
    return std::numeric_limits<size_t>::max();
}

} // namespace scissor
