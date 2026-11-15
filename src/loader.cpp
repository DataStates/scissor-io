#include <scissor/loader.hpp>
#include <scissor/log.hpp>

#include "common/host_cache.hpp"
#include <scissor/batch.hpp>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <stdexcept>
#include <system_error>
#include <unistd.h>

namespace scissor {

namespace {

void validate_tier(TransferType tier) {
    if (tier != TransferType::FileToHost && tier != TransferType::FileToDevice)
        throw std::invalid_argument(
            "scissor: load() requires TransferType::FileToHost or FileToDevice");
}

void validate_seg_size(size_t seg_size) {
    if (seg_size == 0 || seg_size % 4096 != 0)
        throw std::invalid_argument(
            "scissor: seg_size must be a non-zero multiple of 4096 (O_DIRECT)");
}

} // namespace

// ======================================================
// Constructor
// ======================================================
Loader::Loader(IReader& reader, Config cfg)
    : reader_(reader), cfg_(cfg)
{
    // Resolve LRU FD cache capacity: env var overrides zero default
    if (cfg_.max_open_fds == 0) {
        const char* env = std::getenv("SCISSOR_MAX_OPEN_FDS");
        lru_cap_ = (env && *env) ? static_cast<size_t>(std::stoull(env)) : 32768;
    } else {
        lru_cap_ = cfg_.max_open_fds;
    }

    // Resolve max_staging_bytes: env var in MB
    if (cfg_.max_staging_bytes == 0) {
        const char* env = std::getenv("SCISSOR_MAX_STAGING_MB");
        if (env && *env) {
            cfg_.max_staging_bytes =
                static_cast<size_t>(std::stoull(env)) * 1024 * 1024;
        }
    }

    // Construct the HostCache
    host_cache_ = new common::HostCache();

    SCISSOR_DEBUG("Loader initialized",
        "use_gpu",         cfg_.use_gpu,
        "max_staging_mb",  cfg_.max_staging_bytes / (1024 * 1024),
        "max_open_fds",    lru_cap_);
}

// ======================================================
// Destructor: close all cached FDs
// ======================================================
Loader::~Loader() {
    // Wait for in-flight reads and free batches before closing the
    // file descriptors the reads use.
    for (auto& entry : total_batches_) {
        host_cache_->drop(entry.first);
    }
    for (auto& entry : lru_list_) {
        if (entry.second >= 0) ::close(entry.second);
    }
    lru_list_.clear();
    lru_map_.clear();
    delete host_cache_;
    SCISSOR_DEBUG("Loader destroyed");
}

// ======================================================
// get_or_open_fd: LRU FD cache
// ======================================================
int Loader::get_or_open_fd(const std::string& path) {
    auto it = lru_map_.find(path);
    if (it != lru_map_.end()) {
        // Cache hit: move to front
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
        return it->second->second;
    }

    // Cache miss: evict LRU entry if at capacity
    if (lru_list_.size() >= lru_cap_) {
        auto& lru = lru_list_.back();
        SCISSOR_DEBUG("evicting LRU fd", "path", lru.first, "fd", lru.second);
        ::close(lru.second);
        lru_map_.erase(lru.first);
        lru_list_.pop_back();
    }

    // Open the new file
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        int err = errno;
        if (err == EMFILE || err == ENFILE) {
            SCISSOR_WARN("open() failed: FD limit hit despite LRU eviction",
                "path", path, "errno_msg", std::strerror(err),
                "tip", "raise SCISSOR_MAX_OPEN_FDS or system ulimit -n");
        }
        throw std::system_error(err, std::generic_category(),
                                "scissor: cannot open " + path);
    }

    lru_list_.emplace_front(path, fd);
    lru_map_[path] = lru_list_.begin();
    return fd;
}

// ======================================================
// enqueue_reads_sequential: stage sequential batches
// for a single-file read (no scatter coalescing).
// Ported from data_loader_t::enqueue_reads().
// ======================================================
void Loader::enqueue_reads_sequential(int id, int fd, size_t seg_size,
                                      size_t n_segs_per_read,
                                      size_t total_n_segs,
                                      size_t total_read_size) {
    size_t n_read_call = (total_n_segs + n_segs_per_read - 1) / n_segs_per_read;
    size_t staged_size = 0;
    size_t batch_len   = n_segs_per_read;

    auto* batch = new common::Batch(batch_len, 0, seg_size);
    for (size_t i = 0; i < total_n_segs; ++i) {
        if (i % n_segs_per_read == 0 && i > 0) {
            SCISSOR_DEBUG("staging sequential batch",
                "id", id,
                "batch_num", i / n_segs_per_read - 1,
                "batch_len", batch_len);
            host_cache_->stage_in(id, batch);
            n_read_call--;
            size_t remaining = total_n_segs - i;
            batch_len = (n_read_call == 1) ? remaining : batch_len;
            batch     = new common::Batch(batch_len, 0, seg_size);
        }
        size_t offset          = i * seg_size;
        size_t current_seg_size = seg_size;
        if (i == total_n_segs - 1) {
            // Align to 4K for O_DIRECT on the last segment
            constexpr size_t ALIGN = 4096;
            size_t tail      = total_read_size - staged_size;
            current_seg_size = (tail + ALIGN - 1) & ~(ALIGN - 1);
        }
        Segment seg;
        seg.id     = i;
        seg.fd     = fd;
        seg.offset = offset;
        seg.size   = current_seg_size;
        seg.buffer = nullptr;
        batch->push(seg);
        staged_size += current_seg_size;
    }
    SCISSOR_DEBUG("staging final sequential batch",
        "id", id, "batch_len", batch_len);
    host_cache_->stage_in(id, batch);
}

// ======================================================
// load: single-file sequential
// ======================================================
LoadHandle Loader::load(const std::string& file,
                        size_t             total_read_size,
                        size_t             seg_size,
                        TransferType       tier) {
    SCISSOR_TIMER_START(load_seq);
    validate_tier(tier);
    validate_seg_size(seg_size);
    if (total_read_size == 0)
        throw std::invalid_argument("scissor: total_read_size must be > 0");

    const int fd = get_or_open_fd(file);

    int id          = instance_count_++;
    ready_count_[id]   = 0;
    wasted_bytes_[id]  = 0;
    iop_count_[id]     = 0;
    total_batches_[id] = 0;

    SCISSOR_DEBUG("load sequential",
        "id", id, "path", file,
        "total_bytes", total_read_size, "seg_size", seg_size);

    size_t total_n_segs    = (total_read_size + seg_size - 1) / seg_size;
    size_t n_segs_per_read = std::min(size_t{2}, total_n_segs);
    // Compute number of batches that will be staged so next() knows when to stop
    total_batches_[id] = (total_n_segs + n_segs_per_read - 1) / n_segs_per_read;

    enqueue_reads_sequential(id, fd, seg_size, n_segs_per_read,
                             total_n_segs, total_read_size);

    host_cache_->set_reader(id, &reader_);

    SCISSOR_TIMER_PERF(load_seq, "sequential load staged",
        "id", id, "total_segs", total_n_segs);

    return LoadHandle{id};
}

// ======================================================
// load: two-file comparison
// ======================================================
LoadHandle Loader::load(const std::string&  file0,
                        const std::string&  file1,
                        std::vector<size_t> offsets,
                        size_t              seg_size,
                        TransferType        tier,
                        FileSrc             src,
                        int                 nthreads,
                        uint32_t            gap,
                        size_t              block_size) {
    FileRequest req0{file0, offsets};
    FileRequest req1{file1, offsets};
    return load({req0, req1}, seg_size, tier, src, nthreads, gap, block_size);
}

// ======================================================
// load: generalized N-file scatter
// ======================================================
LoadHandle Loader::load(const std::vector<FileRequest>& requests,
                        size_t                          seg_size,
                        TransferType                    tier,
                        FileSrc                         src,
                        int                             nthreads,
                        uint32_t                        gap,
                        size_t                          block_size) {
    SCISSOR_TIMER_START(load_scatter);
    validate_tier(tier);
    validate_seg_size(seg_size);
    if (requests.empty())
        throw std::invalid_argument("scissor: load() requires at least one FileRequest");

    // All files are read at the first request's offsets
    const auto& offsets = requests[0].offsets;
    for (const auto& req : requests) {
        if (req.offsets != offsets)
            throw std::invalid_argument(
                "scissor: every FileRequest must use the same offsets");
    }
    if (std::adjacent_find(offsets.begin(), offsets.end(),
                           std::greater_equal<size_t>()) != offsets.end())
        throw std::invalid_argument(
            "scissor: offsets must be sorted ascending without duplicates");

    // Open all file descriptors
    std::vector<int> fds;
    fds.reserve(requests.size());
    for (const auto& req : requests) {
        fds.push_back(get_or_open_fd(req.path));
    }

    int id         = instance_count_++;
    ready_count_[id]   = 0;
    wasted_bytes_[id]  = 0;
    iop_count_[id]     = 0;
    total_batches_[id] = 0;  // set below after aggregation

    SCISSOR_DEBUG("load scatter",
        "id", id, "files", requests.size(),
        "seg_size", seg_size, "nthreads", nthreads);

    // Decide gap and block_size via Regressor if threading
    if (nthreads > 1) {
        Regressor reg(src);
        auto [opt_gap, opt_block] = reg.optimize(offsets, seg_size, nthreads);
        gap        = static_cast<uint32_t>(opt_gap);
        block_size = opt_block;
        SCISSOR_DEBUG("regressor result",
            "id", id, "gap", gap, "block_size_mb", block_size >> 20);
    }
    if (block_size == 0) block_size = seg_size;

    // Clamp block_size to max_staging_bytes if set
    if (cfg_.max_staging_bytes > 0 && block_size > cfg_.max_staging_bytes) {
        SCISSOR_WARN("block_size exceeds max_staging_bytes, clamping",
            "block_size_mb",       block_size >> 20,
            "max_staging_bytes_mb", cfg_.max_staging_bytes >> 20);
        block_size = cfg_.max_staging_bytes;
    }

    int    n_readers       = static_cast<int>(requests.size());
    size_t used_chks       = (block_size + seg_size - 1) / seg_size;
    auto   agg_result      = aggregator_.coalesce(offsets, seg_size, gap, used_chks);

    wasted_bytes_[id]  = agg_result.wasted_bytes;
    iop_count_[id]     = agg_result.iop_count;
    total_batches_[id] = agg_result.batch_ends.size();
    params_[id]        = LoadParams{gap, block_size};

    // Stage one batch per Aggregator group (at least used_chks requested
    // chunks each).  Every segment is pushed once per file, so the reads
    // for the same offsets in different files are submitted next to each other.
    size_t beg = 0;
    for (size_t b = 0; b < agg_result.batch_ends.size(); ++b) {
        size_t end   = agg_result.batch_ends[b];
        auto*  batch = new common::Batch((end - beg) * n_readers,
                                         agg_result.batch_chunks[b],
                                         seg_size, n_readers);
        for (size_t si = beg; si < end; ++si) {
            const Segment& proto = agg_result.segments[si];
            for (int j = 0; j < n_readers; ++j) {
                Segment seg;
                seg.id     = proto.id;
                seg.fd     = fds[j];
                seg.offset = proto.offset;
                seg.size   = proto.size;
                seg.buffer = nullptr;
                batch->push(seg);
            }
        }
        host_cache_->stage_in(id, batch);
        beg = end;
    }

    SCISSOR_DEBUG("all batches staged",
        "id", id, "segments", agg_result.segments.size(),
        "iop_count", iop_count_[id], "wasted_bytes", wasted_bytes_[id]);

    // Set readers: single or dual depending on n_readers
    if (n_readers == 1) {
        host_cache_->set_reader(id, &reader_);
    } else {
        // For dual-file the Loader holds one IReader, and the caller would supply two.
        // With a single IReader we pass it for both slots. The caller can
        // override by constructing two adapters over two readers.
        host_cache_->set_reader(id, &reader_);
    }

    SCISSOR_TIMER_PERF(load_scatter, "scatter load staged",
        "id", id, "iop_count", iop_count_[id], "wasted_bytes", wasted_bytes_[id]);

    return LoadHandle{id};
}

// ======================================================
// next: retrieve the next completed batch
// ======================================================
common::Batch* Loader::next(LoadHandle handle, TransferType /*tier*/) {
    int  id = handle.id;
    auto tb = total_batches_.find(id);
    if (tb == total_batches_.end())
        throw std::invalid_argument("scissor: next() called on an unknown or closed handle");
    size_t call_count = ready_count_[id]++;

    SCISSOR_TIMER_START(next);

    // Free the batch returned by the previous call on this handle
    if (holding_[id]) {
        host_cache_->release(id);
        holding_[id] = false;
    }

    // All batches have been returned: signal end of stream
    if (call_count >= tb->second) {
        SCISSOR_DEBUG("next: all batches delivered", "id", id,
            "total_batches", tb->second);
        return nullptr;
    }

    common::Batch* batch = host_cache_->get_completed(id);
    holding_[id] = true;

    SCISSOR_TIMER_PERF(next, "next batch ready",
        "id", id, "batch_size", batch->size,
        "seg_count", batch->count);

    return batch;
}

// ======================================================
// close
// ======================================================
void Loader::close(LoadHandle handle) {
    int id = handle.id;
    host_cache_->drop(id);
    wasted_bytes_.erase(id);
    iop_count_.erase(id);
    ready_count_.erase(id);
    total_batches_.erase(id);
    holding_.erase(id);
    params_.erase(id);
    SCISSOR_DEBUG("handle closed", "id", id);
}

// ======================================================
// Statistics
// ======================================================
size_t Loader::wasted_bytes(LoadHandle handle) const {
    auto it = wasted_bytes_.find(handle.id);
    return (it != wasted_bytes_.end()) ? it->second : 0;
}

size_t Loader::iop_count(LoadHandle handle) const {
    auto it = iop_count_.find(handle.id);
    return (it != iop_count_.end()) ? it->second : 0;
}

LoadParams Loader::params(LoadHandle handle) const {
    auto it = params_.find(handle.id);
    return (it != params_.end()) ? it->second : LoadParams{0, 0};
}

} // namespace scissor
