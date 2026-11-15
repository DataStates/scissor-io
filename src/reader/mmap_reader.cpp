#include <scissor/readers.hpp>
#include <scissor/log.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <system_error>
#include <thread>
#include <unistd.h>

namespace scissor {

MmapReader::MmapReader(size_t nthreads) {
    num_threads = nthreads;
}

MmapReader::~MmapReader() {
    if (buffer)
        munmap(buffer, fsize);
    for (auto& m : mappings)
        munmap(m.second.first, m.second.second);
}

MmapReader::MmapReader(const std::string& name, size_t nthreads) {
    num_threads = nthreads;
    fname       = name;
    fd          = open(name.c_str(), O_RDONLY | O_DIRECT);
    if (fd == -1)
        throw std::system_error(errno, std::generic_category(),
                                "scissor: cannot open " + fname);
    fsize  = lseek(fd, 0, SEEK_END);
    void* mapped = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    int   err    = errno;
    close(fd);
    if (mapped == MAP_FAILED)
        throw std::system_error(err, std::generic_category(),
                                "scissor: mmap failed for " + fname);
    buffer = static_cast<uint8_t*>(mapped);
    madvise(buffer, fsize, MADV_RANDOM);
}

int MmapReader::read_data(size_t beg, size_t end) {
    SCISSOR_TIMER_START(read);
    for (size_t i = beg; i < end; i++) {
        Segment& seg = reads[i];
        memcpy(seg.buffer, sources[i] + seg.offset, seg.size);
        status_mutex.lock();
        segment_status[i] = true;
        status_mutex.unlock();
    }
    SCISSOR_TIMER_PERF(read, "Read segments",
                       "beg", beg,
                       "end", end);
    return 0;
}

int MmapReader::enqueue_reads(const std::vector<Segment>& segments) {
    SCISSOR_TIMER_START(enqueue);
    // Resolve the mapping of each segment (the file behind Segment::fd,
    // mapped once per file, or the file given to the constructor when it
    // is -1) and clamp reads to the file size
    std::vector<Segment> resolved(segments.begin(), segments.end());
    std::vector<const uint8_t*> resolved_src(segments.size());
    for (size_t i = 0; i < resolved.size(); i++) {
        Segment& seg = resolved[i];
        uint8_t* base = buffer;
        size_t   len  = fsize;
        if (seg.fd >= 0) {
            struct stat st;
            if (fstat(seg.fd, &st) < 0)
                return -errno;
            auto key = std::make_pair(st.st_dev, st.st_ino);
            auto it  = mappings.find(key);
            if (it == mappings.end()) {
                size_t fs = static_cast<size_t>(st.st_size);
                void* mapped = mmap(NULL, fs, PROT_READ, MAP_PRIVATE, seg.fd, 0);
                if (mapped == MAP_FAILED)
                    return -errno;
                madvise(mapped, fs, MADV_RANDOM);
                it = mappings.emplace(key, std::make_pair(static_cast<uint8_t*>(mapped), fs)).first;
            }
            base = it->second.first;
            len  = it->second.second;
        }
        if (seg.offset + seg.size > len)
            seg.size = (len > seg.offset) ? len - seg.offset : 0;
        resolved_src[i] = base;
    }
    status_mutex.lock();
    segment_status.insert(segment_status.end(), resolved.size(), false);
    status_mutex.unlock();
    reads.insert(reads.end(), resolved.begin(), resolved.end());
    sources.insert(sources.end(), resolved_src.begin(), resolved_src.end());
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
                                     &MmapReader::read_data, this, beg, end));
        submitted_count += end - beg;
    }
    SCISSOR_TIMER_PERF(launch_threads, "Launch threads for reading",
                       "num_threads", num_threads);
    return 0;
}

int MmapReader::wait(size_t id) {
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
    status_mutex.unlock();
    completed_count++;
    return 0;
}

int MmapReader::wait_all() {
    for (uint32_t i = 0; i < futures.size(); i++) {
        futures[i].get();
    }
    completed_count = submitted_count;
    futures.clear();
    reads.clear();
    sources.clear();
    status_mutex.lock();
    segment_status.clear();
    status_mutex.unlock();
    return first_error.exchange(0);
}

std::vector<size_t> MmapReader::wait_n(size_t n) {
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

size_t MmapReader::wait_any() {
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
