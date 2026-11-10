# One Makefile for the whole repository. Every program is a single .cu file
# with a main(); each is compiled straight to build/<name>.
#
#   make                build everything
#   make l2             build one program        -> build/l2
#   make latency        build one directory      (latency, throughput, extra, unified-dram, dgx-spark, bfs)
#   make list           show every program and its source
#   make clean          rm -rf build/
#
#   make ARCH=sm_90     another GPU;  make EXTRA=-lineinfo   extra nvcc flags
#
# sm_121 is mandatory on GB10: the default nvcc target silently lacks thread
# block cluster support. cuda_micro_bench/ is a third-party clone with its
# own multi-file build and is not handled here.

NVCC      ?= nvcc
ARCH      ?= sm_121
NVCCFLAGS ?= -O3 -arch=$(ARCH)
EXTRA     ?= 

BUILD    := build
SRC_DIRS := dgx-spark bfs

# Only sources with a main() are programs; headers-only or fragment .cu files
# are skipped. (no parentheses in the pattern: make counts them in $(shell))
SRCS  := $(shell find $(SRC_DIRS) -name '*.cu' -not -path '*/SbatchMan/*' \
                 -exec grep -l -E '^[[:space:]]*int[[:space:]]+main' {} +)
NAMES := $(notdir $(SRCS:.cu=))
BINS  := $(addprefix $(BUILD)/,$(NAMES))

# A flat build/ needs unique program names: fail loudly instead of letting
# one source silently overwrite another's binary.
DUPES := $(shell printf '%s\n' $(NAMES) | sort | uniq -d)
ifneq ($(DUPES),)
  $(error two sources share a name, rename one: $(DUPES))
endif

vpath %.cu $(sort $(dir $(SRCS)))

.PHONY: all clean list $(NAMES)
.DEFAULT_GOAL := all

all: $(BINS)

# -MMD -MP writes build/<name>.d listing the headers a program includes, so
# editing dgx-spark/common.cuh rebuilds exactly the programs that use it.
$(BUILD)/%: %.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(EXTRA) -MMD -MP -MF $@.d $< -o $@ 2>&1

$(BUILD):
	mkdir -p $@

# `make l2` as a shortcut for `make build/l2`
$(NAMES): %: $(BUILD)/%

# `make latency` builds every program whose source lives in a directory of
# that name (the directory itself, not its subdirectories).
dirname_of = $(notdir $(patsubst %/,%,$(dir $(1))))
DIR_GROUPS := $(sort $(foreach s,$(SRCS),$(call dirname_of,$(s))))
.PHONY: $(DIR_GROUPS)
$(foreach s,$(SRCS),$(eval $(call dirname_of,$(s)): $(BUILD)/$(notdir $(basename $(s)))))

list:
	@for s in $(SRCS); do n=$$(basename $$s .cu); printf '%-24s %s\n' "$$n" "$$s"; done | sort

clean:
	rm -rf $(BUILD)

-include $(BINS:=.d)
