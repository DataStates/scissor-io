#ifndef SCISSOR_LOG_HPP
#define SCISSOR_LOG_HPP

/*
 * scissor/log.hpp
 * Header-only structured JSONL logger for scissor.
 *
 * Usage:
 *   #include <scissor/log.hpp>
 *   SCISSOR_INFO("stage_in complete", "batch_id", id, "bytes", n);
 *   SCISSOR_TIMER_START(load);
 *   SCISSOR_TIMER_PERF(load, "load complete", "bytes", n);
 *
 * The component field ("comp") of every entry is "scissor::" + the
 * enclosing function name, and "src" is file:line of the call site.
 *
 * Runtime env vars:
 *   SCISSOR_LOG_LEVEL      : min level name (DEBUG/PERF/INFO/WARN/ERROR/FATAL), default INFO
 *   SCISSOR_LOG_FILE       : output path, use %r for rank (e.g. logs/rank_%r.jsonl), default stderr
 *   SCISSOR_LOG_RANK0_ONLY : if set, ranks > 0 suppress below WARN
 *
 * Compile-time:
 *   SCISSOR_MIN_LOG_LEVEL  : integer 0-5, log calls below this level are compiled out (default 2)
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>

#ifndef SCISSOR_MIN_LOG_LEVEL
#  define SCISSOR_MIN_LOG_LEVEL 2  /* INFO */
#endif

namespace scissor {
namespace log {

/* ==========================================================================
 * Level enum
 * ==========================================================================*/
enum class Level : int {
    DEBUG = 0,
    PERF  = 1,
    INFO  = 2,
    WARN  = 3,
    ERROR = 4,
    FATAL = 5,
};

inline const char* level_name(Level l) noexcept {
    switch (l) {
        case Level::DEBUG: return "DEBUG";
        case Level::PERF:  return "PERF";
        case Level::INFO:  return "INFO";
        case Level::WARN:  return "WARN";
        case Level::ERROR: return "ERROR";
        case Level::FATAL: return "FATAL";
    }
    return "UNKNOWN";
}

inline Level level_from_string(const char* s) noexcept {
    if (!s) return Level::INFO;
    if (std::strcmp(s, "DEBUG") == 0) return Level::DEBUG;
    if (std::strcmp(s, "PERF")  == 0) return Level::PERF;
    if (std::strcmp(s, "INFO")  == 0) return Level::INFO;
    if (std::strcmp(s, "WARN")  == 0) return Level::WARN;
    if (std::strcmp(s, "ERROR") == 0) return Level::ERROR;
    if (std::strcmp(s, "FATAL") == 0) return Level::FATAL;
    return Level::INFO;
}

/* ==========================================================================
 * Timing helpers
 * ==========================================================================*/
using TimePoint = std::chrono::steady_clock::time_point;

inline TimePoint now() noexcept {
    return std::chrono::steady_clock::now();
}

inline double elapsed_ms(TimePoint t) noexcept {
    using ms = std::chrono::duration<double, std::milli>;
    return std::chrono::duration_cast<ms>(now() - t).count();
}

inline double elapsed_s(TimePoint t) noexcept {
    using sec = std::chrono::duration<double>;
    return std::chrono::duration_cast<sec>(now() - t).count();
}

/* ==========================================================================
 * Logger singleton
 * ==========================================================================*/
class Logger {
public:
    static Logger& instance() {
        /* C++11 function-local static: guaranteed one-time init, thread-safe,
         * and safe across DSOs that share this header. */
        static Logger inst;
        return inst;
    }

    /* Public configuration API.  Call before first log to override env vars. */
    void set_level(Level l)              { runtime_level_ = l; }
    void set_rank(int r)                 { rank_ = r; }
    void set_output(std::ostream& os)    { out_ = &os; }
    void set_output(const std::string& path) {
        std::string resolved = replace_rank(path, rank_);
        file_out_.open(resolved, std::ios::app);
        if (file_out_.is_open())
            out_ = &file_out_;
    }

    Level runtime_level() const noexcept { return runtime_level_; }
    int   rank()          const noexcept { return rank_; }
    bool  rank0_only()    const noexcept { return rank0_only_; }

    /* Core emit: builds JSONL in a thread-local buffer, then writes under mutex. */
    template<typename... KV>
    void emit(Level level, std::string_view comp, std::string_view src,
              std::string_view msg, KV&&... kv) {
        if (level < runtime_level_) return;
        if (rank0_only_ && rank_ > 0 && level < Level::WARN) return;

        thread_local std::ostringstream tl_buf;
        tl_buf.str("");
        tl_buf.clear();

        using ms = std::chrono::duration<long long, std::milli>;
        auto epoch_ms = std::chrono::duration_cast<ms>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        tl_buf << "{"
               << "\"ts\":"     << epoch_ms
               << ",\"level\":\"" << level_name(level) << "\""
               << ",\"rank\":"  << rank_
               << ",\"comp\":\"" << escape(comp) << "\""
               << ",\"src\":\""  << escape(src)  << "\""
               << ",\"msg\":\""  << escape(msg)  << "\"";

        append_kv(tl_buf, std::forward<KV>(kv)...);
        tl_buf << "}\n";

        {
            std::lock_guard<std::mutex> lk(write_mutex_);
            out_->write(tl_buf.str().data(),
                        static_cast<std::streamsize>(tl_buf.str().size()));
            out_->flush();
        }
    }

private:
    Logger() {
        rank_          = detect_rank();
        runtime_level_ = Level::INFO;
        out_           = &std::cerr;
        rank0_only_    = false;
        init();
    }

    Logger(const Logger&)            = delete;
    Logger& operator=(const Logger&) = delete;

    /* =======================================================================
     * Rank auto-detection from launcher env vars.
     *
     * Priority order:
     *   PMI_RANK            : PMI2 protocol (Cray/MPICH-based launchers)
     *   PMIX_RANK           : PMIx protocol (Open MPI, Slurm native)
     *   SLURM_PROCID        : Slurm process ID (fallback when PMI is absent)
     *   OMPI_COMM_WORLD_RANK: Open MPI direct env (no PMI launcher)
     *   LSB_RANKID          : IBM LSF (Summit/OLCF)
     *
     * Note: PBS_VNODENUM is a node index, not a global rank, so it is omitted.
     * =======================================================================*/
    static int detect_rank() noexcept {
        for (const char* var : {"PMI_RANK", "PMIX_RANK", "SLURM_PROCID",
                                 "OMPI_COMM_WORLD_RANK", "LSB_RANKID"}) {
            const char* v = std::getenv(var);
            if (v && *v) return std::atoi(v);
        }
        return 0;
    }

    void init() {
        if (const char* lv = std::getenv("SCISSOR_LOG_LEVEL"))
            runtime_level_ = level_from_string(lv);

        if (std::getenv("SCISSOR_LOG_RANK0_ONLY"))
            rank0_only_ = true;

        if (const char* fp = std::getenv("SCISSOR_LOG_FILE")) {
            std::string resolved = replace_rank(fp, rank_);
            file_out_.open(resolved, std::ios::app);
            if (file_out_.is_open())
                out_ = &file_out_;
        }
    }

    static std::string replace_rank(std::string_view tmpl, int rank) {
        std::string result(tmpl);
        std::string rank_str = std::to_string(rank);
        std::string::size_type pos;
        while ((pos = result.find("%r")) != std::string::npos)
            result.replace(pos, 2, rank_str);
        return result;
    }

    /* =======================================================================
     * JSON value helper.
     *
     * Anything convertible to std::string_view (string literals, char*,
     * std::string, std::string_view) is emitted as a quoted, escaped JSON
     * string, bool as true/false, and everything else via operator<<.
     * Dispatch is on the decayed type so that string arguments are quoted
     * regardless of value category or array type.
     * =======================================================================*/
    template<typename T>
    static void json_val(std::ostringstream& os, const T& v) {
        using D = std::decay_t<T>;
        if constexpr (std::is_same_v<D, bool>) {
            os << (v ? "true" : "false");
        } else if constexpr (std::is_convertible_v<const T&, std::string_view>) {
            os << '"' << escape(std::string_view(v)) << '"';
        } else {
            os << v;
        }
    }

    /* =======================================================================
     * KV pair unpacking (alternating key, value, key, value, ...)
     * =======================================================================*/
    static void append_kv(std::ostringstream&) {}

    template<typename K, typename V, typename... Rest>
    static void append_kv(std::ostringstream& os, K&& key, V&& val, Rest&&... rest) {
        os << ",\"" << key << "\":";
        json_val(os, val);
        append_kv(os, std::forward<Rest>(rest)...);
    }

    /* =======================================================================
     * JSON string escape (handles ASCII control characters and specials)
     * =======================================================================*/
    static std::string escape(std::string_view s) {
        std::string out;
        out.reserve(s.size());
        for (char c : s) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x",
                                      static_cast<unsigned>(c));
                        out += buf;
                    } else {
                        out += c;
                    }
                    break;
            }
        }
        return out;
    }

    Level          runtime_level_;
    int            rank_;
    bool           rank0_only_;
    std::ostream*  out_;
    std::ofstream  file_out_;
    std::mutex     write_mutex_;
};

}  // namespace log
}  // namespace scissor

/* ==========================================================================
 * Stringify helper (two-level expansion required for __LINE__)
 * ==========================================================================*/
#define SCISSOR_STRINGIFY_INNER(x) #x
#define SCISSOR_STRINGIFY(x)       SCISSOR_STRINGIFY_INNER(x)

/* ==========================================================================
 * Core macro with compile-time gate via if constexpr.
 *
 * level  : bare level name (INFO, DEBUG, etc. without scope prefix)
 * comp   : component identifier string (e.g. "scissor::load")
 * src    : source location string (e.g. "file:line")
 * msg    : message string
 * ...    : optional key-value pairs (key, value, key, value, ...)
 * ==========================================================================*/
#define SCISSOR_LOG(level, comp, src, msg, ...)                                      \
    do {                                                                              \
        if constexpr (SCISSOR_MIN_LOG_LEVEL <=                                       \
                      static_cast<int>(::scissor::log::Level::level)) {              \
            ::scissor::log::Logger::instance().emit(                                  \
                ::scissor::log::Level::level,                                         \
                (comp), (src), (msg), ##__VA_ARGS__);                               \
        }                                                                             \
    } while (0)

/* ==========================================================================
 * Source location macro
 * ==========================================================================*/
#define _SCISSOR_SRC (__FILE__ ":" SCISSOR_STRINGIFY(__LINE__))

/* ==========================================================================
 * Core per-level convenience macros.
 * Component field is "scissor::" + enclosing function name
 * (__func__ is evaluated at the call site).
 *
 * Signature: SCISSOR_LEVEL(msg [, key, value, ...])
 *   msg   -- short human-readable description (string literal or std::string)
 *   key   -- string literal key for a structured field
 *   value -- any type supported by scissor::log::Logger::json_val()
 * ==========================================================================*/
#define SCISSOR_DEBUG(msg, ...) \
    SCISSOR_LOG(DEBUG, ("scissor::" + std::string(__func__)).c_str(), _SCISSOR_SRC, msg, ##__VA_ARGS__)

#define SCISSOR_PERF(msg, ...) \
    SCISSOR_LOG(PERF, ("scissor::" + std::string(__func__)).c_str(), _SCISSOR_SRC, msg, ##__VA_ARGS__)

#define SCISSOR_INFO(msg, ...) \
    SCISSOR_LOG(INFO, ("scissor::" + std::string(__func__)).c_str(), _SCISSOR_SRC, msg, ##__VA_ARGS__)

#define SCISSOR_WARN(msg, ...) \
    SCISSOR_LOG(WARN, ("scissor::" + std::string(__func__)).c_str(), _SCISSOR_SRC, msg, ##__VA_ARGS__)

#define SCISSOR_ERROR(msg, ...) \
    SCISSOR_LOG(ERROR, ("scissor::" + std::string(__func__)).c_str(), _SCISSOR_SRC, msg, ##__VA_ARGS__)

// SCISSOR_FATAL emits a FATAL log entry then calls std::abort().
// Wrapped in do{}while(0) so the macro is safe in single-statement branches
// (e.g. if (cond) SCISSOR_FATAL(...)) without the bare std::abort() executing
// unconditionally on the next line.
#define SCISSOR_FATAL(msg, ...) \
    do { \
        SCISSOR_LOG(FATAL, ("scissor::" + std::string(__func__)).c_str(), \
                    _SCISSOR_SRC, msg, ##__VA_ARGS__); \
        std::abort(); \
    } while(0)

// ======================================================
// Timer helpers: start a named timer, then emit a PERF
// log entry with elapsed_ms when done.
//
// SCISSOR_TIMER_START(name)         -- captures current time
// SCISSOR_TIMER_PERF(name, msg, ...) -- emits PERF log with elapsed_ms
//
// 'name' is a bare identifier (not a string).  Multiple
// timers with different names may coexist in one scope.
// ======================================================
#define SCISSOR_TIMER_START(name) \
    auto _scissor_t_##name = ::scissor::log::now()

#define SCISSOR_TIMER_PERF(name, msg, ...) \
    SCISSOR_PERF(msg, "elapsed_ms", ::scissor::log::elapsed_ms(_scissor_t_##name), ##__VA_ARGS__)

#endif // SCISSOR_LOG_HPP
