# Makefile: AI IO Benchmark (all C++, with optional scissor engine)

CXX      = g++
CXXFLAGS = -O2 -Wall -Wextra -std=c++17 -D_GNU_SOURCE -I. -pthread

LDFLAGS  = -pthread -lm

# ── Optional: scissor-io install prefix ──────────────────────────────
# SCISSOR_PREFIX is the CMake install prefix of scissor-io
# (cmake --install ... --prefix $SCISSOR_PREFIX).  It provides
# include/scissor/*.hpp, lib/libscissor.a and, when scissor-io built its
# own liburing, lib/liburing.a plus include/liburing.h.
SCISSOR_PREFIX ?=

ifneq ($(SCISSOR_PREFIX),)
SCISSOR_LIB := $(firstword $(wildcard \
    $(SCISSOR_PREFIX)/lib64/libscissor.a \
    $(SCISSOR_PREFIX)/lib/libscissor.a))
SCISSOR_INC := $(firstword $(wildcard $(SCISSOR_PREFIX)/include/scissor/scissor.hpp))
SCISSOR_INC := $(if $(SCISSOR_INC),$(SCISSOR_PREFIX)/include)
URING_STATIC_LIB := $(firstword $(wildcard \
    $(SCISSOR_PREFIX)/lib64/liburing.a \
    $(SCISSOR_PREFIX)/lib/liburing.a))
endif

# ── Detect available libraries ────────────────────────────────────────
# libaio: try pkg-config first, then fall back to a compiler header probe so
# that installations without a .pc file (e.g. Ubuntu libaio-dev) are found.
HAS_LIBAIO_PC  := $(shell pkg-config --exists libaio 2>/dev/null && echo 1)
HAS_LIBAIO_HDR := $(shell $(CXX) -x c++ -include libaio.h -E /dev/null > /dev/null 2>&1 && echo 1)
HAS_LIBAIO     := $(or $(HAS_LIBAIO_PC),$(HAS_LIBAIO_HDR))

# liburing: use the static liburing.a bundled in SCISSOR_PREFIX if present
# (no runtime LD_LIBRARY_PATH needed), otherwise a system liburing found by
# pkg-config (set PKG_CONFIG_PATH for non-standard locations).
ifneq ($(URING_STATIC_LIB),)
  HAS_URING     := 1
  URING_CFLAGS  := -I$(SCISSOR_PREFIX)/include
  URING_LIBS    := $(URING_STATIC_LIB)
else
  HAS_URING     := $(shell pkg-config --exists liburing 2>/dev/null && echo 1)
  URING_CFLAGS  := $(if $(HAS_URING),$(shell pkg-config --cflags liburing))
  URING_LIBS    := $(if $(HAS_URING),$(shell pkg-config --libs liburing))
endif

HAS_SCISSOR := $(shell test -n "$(SCISSOR_LIB)" -a -n "$(SCISSOR_INC)" -a -n "$(HAS_URING)" && echo 1)

# ── Source list ───────────────────────────────────────────────────────
SRCS = bench.cpp engine_pread.cpp engine_mmap.cpp

ifeq ($(HAS_LIBAIO),1)
  SRCS     += engine_libaio.cpp
  CXXFLAGS += -DHAS_LIBAIO $(if $(HAS_LIBAIO_PC),$(shell pkg-config --cflags libaio))
  LDFLAGS  += $(if $(HAS_LIBAIO_PC),$(shell pkg-config --libs libaio),-laio)
  $(info  [Makefile] libaio: ENABLED)
else
  $(info  [Makefile] libaio: NOT FOUND, skipping)
endif

ifeq ($(HAS_URING),1)
  SRCS     += engine_uring.cpp
  CXXFLAGS += -DHAS_URING $(URING_CFLAGS)
  LDFLAGS  += $(URING_LIBS)
  $(info  [Makefile] io_uring: ENABLED ($(URING_LIBS)))
else
  $(info  [Makefile] io_uring: NOT FOUND, skipping)
endif

ifeq ($(HAS_SCISSOR),1)
  SRCS     += engine_scissor.cpp engine_scissor_sr.cpp
  CXXFLAGS += -DHAS_SCISSOR -I$(SCISSOR_INC)
  # libscissor.a must precede liburing on the link line
  LDFLAGS  += $(SCISSOR_LIB) $(URING_LIBS) -lstdc++
  $(info  [Makefile] scissor: ENABLED ($(SCISSOR_LIB)))
else
  $(info  [Makefile] scissor: NOT FOUND, skipping)
  $(info  [Makefile]   (set SCISSOR_PREFIX=<scissor-io install prefix>, which needs include/scissor and lib/libscissor.a, plus liburing))
endif

OBJS = $(SRCS:.cpp=.o)

.PHONY: all clean dirs

all: dirs bench

dirs:
	mkdir -p traces results

bench: $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)
	@echo "[Makefile] Built: $@"

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o bench
	@echo "[Makefile] Cleaned."

# Run quick self-test with the small 4 GB config against the mmap engine.
# Use TEST_CFG to override: make test TEST_CFG=../config/resnet50.yaml
# TEST_DATA is the directory the test dataset is written to.
TEST_CFG  ?= ../config/test_small.yaml
TEST_DATA ?= /tmp/ai_bench_data
test: bench
	python3 ../scripts/gen_trace.py   --config $(TEST_CFG)
	python3 ../scripts/gen_dataset.py --config $(TEST_CFG) --outdir $(TEST_DATA)
	./bench \
	    --trace   traces/test_small_trace.bin \
	    --manifest $(TEST_DATA)/manifest.txt \
	    --engine  mmap \
	    --queue-depth 64 \
	    --compute-mean-ms 28.5 \
	    --compute-std-ms  4.2 \
	    --compute-dist    normal \
	    --report-json results/test_mmap.json \
	    --verbose
