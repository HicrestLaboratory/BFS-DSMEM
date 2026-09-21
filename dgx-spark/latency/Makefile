# Build rules for the latency microbenchmarks (one program per scenario).
#
#   make            build every .cu in this directory
#   make dsmem_remote   build one program
#   make clean      remove every built executable
#
# sm_121 is mandatory: the default nvcc target silently lacks thread block
# cluster support and fails with "cluster misconfiguration" at launch.

NVCC      ?= nvcc
ARCH      ?= sm_121
NVCCFLAGS ?= -O3 -arch=$(ARCH)
EXTRA     ?=

SRCS := $(wildcard *.cu)
BINS := $(SRCS:.cu=)

.PHONY: all clean list
.DEFAULT_GOAL := all

all: $(BINS)

# Every program includes ../common.cuh, so a change there rebuilds all of them.
%: %.cu ../common.cuh
	$(NVCC) $(NVCCFLAGS) $(EXTRA) $< -o $@ 2>&1

clean:
	$(RM) $(BINS)

list:
	@echo $(BINS) | tr ' ' '\n'
