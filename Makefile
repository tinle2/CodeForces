SHELL := /bin/bash

TEMPLATE := template.cpp
SRC      := submit.cpp
INP      := input.txt

# ---------------------------
# Smoothness defaults
# ---------------------------
# Parallel by default (can override: make -j1 ...)
MAKEFLAGS += -j$(shell nproc)

# Put build outputs into tmpfs (RAM) if RAM=1 and /mnt/rambuild exists
RAM ?= 0
BUILD_BASE := build
ifeq ($(RAM),1)
  ifneq ("$(wildcard /mnt/rambuild)","")
    BUILD_BASE := /mnt/rambuild/$(notdir $(CURDIR))
  endif
endif

BUILD_DIR := $(BUILD_BASE)/build
APP       := $(BUILD_DIR)/app
APP_LIVE  := $(BUILD_DIR)/app.live
APP_ASAN  := $(BUILD_DIR)/app.asan

# ---------------------------
# Compiler + linker
# ---------------------------
CXX := ccache g++

# Prefer mold, then lld, else default
LDOPT := $(shell if command -v mold >/dev/null 2>&1; then echo -fuse-ld=mold; \
             elif command -v ld.lld >/dev/null 2>&1; then echo -fuse-ld=lld; \
             else echo ""; fi)
LDFLAGS := $(LDOPT)

# ---------------------------
# PCH (real .gch)
# ---------------------------
PCH_DIR := $(HOME)/.cache/cpp_pch
PCH_H   := $(PCH_DIR)/pch.hpp
PCH_GCH := $(PCH_DIR)/pch.hpp.gch
PCH ?= 1

ifeq ($(PCH),1)
  USE_PCH := -include $(PCH_H)
  PCH_DEPS := $(PCH_GCH)
else
  USE_PCH :=
  PCH_DEPS :=
endif

# ---------------------------
# Flags
# ---------------------------
CXXRUN  := -std=gnu++23 -O2 -pipe -DLOCAL
CXXLIVE := -std=gnu++23 -O2 -pipe -DLOCAL -DDEBUG_AUTO_FLUSH
CXXASAN := -std=gnu++23 -O1 -g -pipe -DLOCAL -DDEBUG_AUTO_FLUSH \
          -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS \
          -fno-omit-frame-pointer -fno-pie -no-pie \
          -fsanitize=address,undefined

ASAN_OPTS  := halt_on_error=1:abort_on_error=1:verbosity=0:print_summary=0:fast_unwind_on_malloc=1:malloc_context_size=0
UBSAN_OPTS := print_stacktrace=1:halt_on_error=1

.PHONY: run run-live run-debug run-debug-live new clean pch stats

# ---------------------------
# Directories / files
# ---------------------------
$(BUILD_DIR):
	@mkdir -p '$(BUILD_DIR)'

$(PCH_H):
	@mkdir -p '$(PCH_DIR)'
	@printf '#pragma once\n#include <bits/stdc++.h>\n' > '$(PCH_H)'

# Build real PCH once (this is what makes -include fast on later compiles)
$(PCH_GCH): $(PCH_H)
	@echo "Building PCH..."
	@$(CXX) -std=gnu++23 -O2 -pipe -x c++-header '$(PCH_H)' -o '$(PCH_GCH)'

pch: $(PCH_GCH)

# ---------------------------
# Build targets
# ---------------------------
$(APP): $(SRC) | $(BUILD_DIR) $(PCH_H) $(PCH_DEPS)
	@echo "Compiling..."
	@$(CXX) $(CXXRUN) $(USE_PCH) '$(SRC)' -o '$@' $(LDFLAGS)

$(APP_LIVE): $(SRC) | $(BUILD_DIR) $(PCH_H) $(PCH_DEPS)
	@echo "Compiling (LIVE)..."
	@$(CXX) $(CXXLIVE) $(USE_PCH) '$(SRC)' -o '$@' $(LDFLAGS)

$(APP_ASAN): $(SRC) | $(BUILD_DIR) $(PCH_H) $(PCH_DEPS)
	@echo "Compiling (ASAN)..."
	@$(CXX) $(CXXASAN) $(USE_PCH) '$(SRC)' -o '$@' $(LDFLAGS)

# ---------------------------
# Run targets
# ---------------------------
run: $(APP)
	@'$(APP)' < '$(INP)'

run-live: $(APP_LIVE)
	@echo "Please type input (Ctrl+D to finish):"
	@script -q -e -c "'$(APP_LIVE)'" /dev/null

run-debug: $(APP_ASAN)
	@ASAN_SO="$$(g++ -print-file-name=libasan.so)"; \
	LDPRE=""; \
	if [[ "$$ASAN_SO" != "libasan.so" && -n "$$ASAN_SO" ]]; then LDPRE="LD_PRELOAD=$$ASAN_SO"; fi; \
	ulimit -s 2097152; \
	env $$LDPRE ASAN_OPTIONS='$(ASAN_OPTS)' UBSAN_OPTIONS='$(UBSAN_OPTS)' '$(APP_ASAN)' < '$(INP)'

run-debug-live: $(APP_ASAN)
	@echo "Please type input (Ctrl+D to finish):"
	@ASAN_SO="$$(g++ -print-file-name=libasan.so)"; \
	LDPRE=""; \
	if [[ "$$ASAN_SO" != "libasan.so" && -n "$$ASAN_SO" ]]; then LDPRE="LD_PRELOAD=$$ASAN_SO"; fi; \
	ulimit -s 2097152; \
	script -q -e -c "env $$LDPRE ASAN_OPTIONS='$(ASAN_OPTS)' UBSAN_OPTIONS='$(UBSAN_OPTS)' '$(APP_ASAN)'" /dev/null

# ---------------------------
# Convenience
# ---------------------------
new: | $(BUILD_DIR)
	@cp -f '$(TEMPLATE)' '$(SRC)'
	@: > '$(INP)'

clean:
	@rm -rf '$(BUILD_DIR)'
	@rm -f '$(APP)' '$(APP_LIVE)' '$(APP_ASAN)'

# Show ccache stats
stats:
	@ccache -s




