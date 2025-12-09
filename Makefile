SHELL := /bin/bash

TEMPLATE  := template.cpp
SUBMIT    := submit.cpp
GENERATOR := generator.cpp
INPUT     := input.txt

BUILD_DIR   := build
BIN_DEBUG   := $(BUILD_DIR)/app
BIN_RELEASE := $(BUILD_DIR)/app.release
GENBIN      := $(BUILD_DIR)/generator

CXX ?= ccache clang++

LDOPT := $(shell command -v mold >/dev/null 2>&1 && echo -fuse-ld=mold || (command -v ld.lld >/dev/null 2>&1 && echo -fuse-ld=lld))

# Debug vs Release flags (no sanitizers, no debug-STL, no warnings)
CXXDEBUG := -std=gnu++23 -O0 -g -DLOCAL -pipe
CXXREL   := -std=gnu++23 -O2 -DNDEBUG -pipe -march=native
LDFLAGS  := $(LDOPT)

# PCH (on by default)
PCH ?= 1
PCHDBG_H := $(BUILD_DIR)/pch.debug.hpp
PCHDBG_G := $(PCHDBG_H).gch
PCHREL_H := $(BUILD_DIR)/pch.rel.hpp
PCHREL_G := $(PCHREL_H).gch

ifeq ($(PCH),1)
  USE_PCH_DBG := -include $(PCHDBG_H)
  USE_PCH_REL := -include $(PCHREL_H)
  DEPS_PCH_DBG := $(PCHDBG_G)
  DEPS_PCH_REL := $(PCHREL_G)
else
  USE_PCH_DBG :=
  USE_PCH_REL :=
  DEPS_PCH_DBG :=
  DEPS_PCH_REL :=
endif

.PHONY: new run run-fast run-interactive make-release run-release clean

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(PCHDBG_H): | $(BUILD_DIR)
	@printf '#include <bits/stdc++.h>\n' > '$@'
$(PCHREL_H): | $(BUILD_DIR)
	@printf '#include <bits/stdc++.h>\n' > '$@'

$(PCHDBG_G): $(PCHDBG_H) | $(BUILD_DIR)
	@$(CXX) $(CXXDEBUG) -x c++-header '$<' -o '$@'
$(PCHREL_G): $(PCHREL_H) | $(BUILD_DIR)
	@$(CXX) $(CXXREL) -x c++-header '$<' -o '$@'

$(BIN_DEBUG): $(SUBMIT) $(DEPS_PCH_DBG) | $(BUILD_DIR)
	@$(CXX) $(CXXDEBUG) $(USE_PCH_DBG) -o '$@' '$(SUBMIT)' $(LDFLAGS)

$(BIN_RELEASE): $(SUBMIT) $(DEPS_PCH_REL) | $(BUILD_DIR)
	@$(CXX) $(CXXREL) $(USE_PCH_REL) -o '$@' '$(SUBMIT)' $(LDFLAGS)

$(GENBIN): $(GENERATOR) $(DEPS_PCH_REL) | $(BUILD_DIR)
	@$(CXX) $(CXXREL) $(USE_PCH_REL) -o '$@' '$(GENERATOR)' $(LDFLAGS)

# Debug run
run: $(BIN_DEBUG)
	@ulimit -s 2097152; '$(BIN_DEBUG)' < '$(INPUT)'

run-interactive: $(BIN_DEBUG)
	@'$(BIN_DEBUG)'

# Fast run (no debug)
run-fast: $(BIN_RELEASE)
	@ulimit -s 2097152; '$(BIN_RELEASE)' < '$(INPUT)'

# Build release only
make-release: $(BIN_RELEASE)
	@:

# Compatibility: release run
run-release: $(BIN_RELEASE)
	@ulimit -s 2097152; '$(BIN_RELEASE)' < '$(INPUT)'

new: | $(BUILD_DIR)
	@cp -f '$(TEMPLATE)' '$(SUBMIT)' && > '$(INPUT)'

clean:
	@rm -rf '$(BUILD_DIR)'

