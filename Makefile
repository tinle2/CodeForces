SHELL := /bin/bash

TEMPLATE  := template.cpp
SUBMIT    := submit.cpp
GENERATOR := generator.cpp
INPUT     := input.txt

BUILD_DIR   := build
BIN_DEBUG   := $(BUILD_DIR)/app
BIN_RELEASE := $(BUILD_DIR)/app.release
GENBIN      := $(BUILD_DIR)/generator

CXX := g++

LDOPT := $(shell command -v mold >/dev/null 2>&1 && echo -fuse-ld=mold || (command -v ld.lld >/dev/null 2>&1 && echo -fuse-ld=lld))
LDFLAGS := $(LDOPT)

CXXDEBUG := -std=gnu++23 -O0 -g -DLOCAL -pipe
CXXREL   := -std=gnu++23 -O2 -DNDEBUG -pipe -march=native

# ===== Global PCH (assume it's already built) =====
PCH ?= 1
PCH_H := $(HOME)/.cache/cpp_pch/pch.hpp

ifeq ($(PCH),1)
  USE_PCH := -include $(PCH_H)
else
  USE_PCH :=
endif
# ================================================

.PHONY: new run run-fast run-interactive make-release run-release clean pch

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# OPTIONAL: build PCH only when you explicitly run `make pch`
pch:
	@mkdir -p '$(HOME)/.cache/cpp_pch'
	@printf '#pragma once\n#include <bits/stdc++.h>\n' > '$(HOME)/.cache/cpp_pch/pch.hpp'
	@echo "[PCH] building ~/.cache/cpp_pch/pch.hpp.gch..."
	@$(CXX) -std=gnu++23 -O2 -pipe -x c++-header '$(HOME)/.cache/cpp_pch/pch.hpp' -o '$(HOME)/.cache/cpp_pch/pch.hpp.gch'

# Build binaries in ONE step (compile+link), like `just`
$(BIN_DEBUG): $(SUBMIT) | $(BUILD_DIR)
	@$(CXX) $(CXXDEBUG) $(USE_PCH) '$(SUBMIT)' -o '$@' $(LDFLAGS)

$(BIN_RELEASE): $(SUBMIT) | $(BUILD_DIR)
	@$(CXX) $(CXXREL) $(USE_PCH) '$(SUBMIT)' -o '$@' $(LDFLAGS)

$(GENBIN): $(GENERATOR) | $(BUILD_DIR)
	@$(CXX) $(CXXREL) $(USE_PCH) '$(GENERATOR)' -o '$@' $(LDFLAGS)

run: $(BIN_DEBUG)
	@'$(BIN_DEBUG)' < '$(INPUT)'

run-interactive: $(BIN_DEBUG)
	@'$(BIN_DEBUG)'

run-fast: $(BIN_RELEASE)
	@'$(BIN_RELEASE)' < '$(INPUT)'

make-release: $(BIN_RELEASE)
	@:

run-release: $(BIN_RELEASE)
	@'$(BIN_RELEASE)' < '$(INPUT)'

new: | $(BUILD_DIR)
	@cp -f '$(TEMPLATE)' '$(SUBMIT)' && > '$(INPUT)'

clean:
	@rm -rf '$(BUILD_DIR)'

