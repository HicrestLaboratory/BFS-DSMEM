# Build rules for the DGX Spark (GB10) microbenchmarks.
#
#   make            build every .cu in this directory that has a main()
#   make l2_topology  build a single program
#   make clean      remove every built executable
#   make list       show what "make all" would build
#
# sm_121 is mandatory: the default nvcc target silently lacks thread block
# cluster support and fails with "__cluster_dims__ is not supported".

NVCC      ?= nvcc
ARCH      ?= sm_121
NVCCFLAGS ?= -O3 -arch=$(ARCH)
EXTRA     ?=

SRCS := $(wildcard *.cu)

# Only sources with a main() are linkable; placeholders and fragments are
# skipped so one empty file cannot fail the whole build.
# (no parentheses in the pattern: make counts them inside $(shell ...))
MAINS := $(shell grep -l -E '^[[:space:]]*int[[:space:]]+main' $(SRCS) 2>/dev/null)
BINS  := $(MAINS:.cu=)

.PHONY: all clean list
.DEFAULT_GOAL := all

all: $(BINS)

# Compile one .cu straight to an executable of the same name.
# 2>&1 folds nvcc's diagnostics into stdout so they survive a pipe or a tee.
%: %.cu
	$(NVCC) $(NVCCFLAGS) $(EXTRA) $< -o $@ 2>&1

# Clean every basename derived from a .cu, not just the ones in BINS, so
# stale binaries left over from a renamed or emptied source are removed too.
clean:
	$(RM) $(SRCS:.cu=)

list:
	@echo $(BINS) | tr ' ' '\n'
