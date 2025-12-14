set shell := ["bash", "-uc"]

PCH      := env_var("HOME") + "/.cache/cpp_pch/pch.hpp"
TEMPLATE := "template.cpp"
SRC      := "submit.cpp"
INP      := "input.txt"

CXX      := "ccache g++"

APP      := "app"
APP_DBG  := "app.dbg"

ASAN_OPTS  := "halt_on_error=1:abort_on_error=1:verbosity=0:print_summary=0:fast_unwind_on_malloc=1:malloc_context_size=0"
UBSAN_OPTS := "print_stacktrace=1:halt_on_error=1"

run:
    @if [[ ! -f "{{PCH}}" ]]; then \
        mkdir -p "$(dirname "{{PCH}}")"; \
        printf '#include <bits/stdc++.h>\n' > "{{PCH}}"; \
    fi; \
    if [[ ! -f {{APP}} || {{APP}} -ot {{SRC}} ]]; then \
        echo "Compiling..."; \
        LDOPT=""; \
        if command -v mold >/dev/null 2>&1; then LDOPT="-fuse-ld=mold"; \
        elif command -v ld.lld >/dev/null 2>&1; then LDOPT="-fuse-ld=lld"; fi; \
        {{CXX}} -std=gnu++23 -O2 -pipe -DLOCAL $LDOPT -include "{{PCH}}" {{SRC}} -o {{APP}}; \
    fi; \
    ./{{APP}} < {{INP}}

run-debug:
    @if [[ ! -f "{{PCH}}" ]]; then \
        mkdir -p "$(dirname "{{PCH}}")"; \
        printf '#include <bits/stdc++.h>\n' > "{{PCH}}"; \
    fi; \
    if [[ ! -f {{APP_DBG}} || {{APP_DBG}} -ot {{SRC}} ]]; then \
        echo "Compiling (ASAN)..."; \
        LDOPT=""; \
        if command -v mold >/dev/null 2>&1; then LDOPT="-fuse-ld=mold"; \
        elif command -v ld.lld >/dev/null 2>&1; then LDOPT="-fuse-ld=lld"; fi; \
        {{CXX}} -std=gnu++23 -O1 -g -pipe -DLOCAL -DDEBUG_AUTO_FLUSH \
          -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS \
          -fno-omit-frame-pointer -fno-pie -no-pie \
          -fsanitize=address,undefined \
          $LDOPT -include "{{PCH}}" {{SRC}} -o {{APP_DBG}}; \
    fi; \
    LDPRE=""; \
    ASAN_SO="$(g++ -print-file-name=libasan.so)"; \
    if [[ "$ASAN_SO" != "libasan.so" && -n "$ASAN_SO" ]]; then LDPRE="LD_PRELOAD=$ASAN_SO"; fi; \
    set -o pipefail; \
    ulimit -s 2097152; \
    if env $LDPRE ASAN_OPTIONS="{{ASAN_OPTS}}" UBSAN_OPTIONS="{{UBSAN_OPTS}}" ./{{APP_DBG}} < {{INP}}; then \
        :; \
    else \
        out="$( env $LDPRE ASAN_OPTIONS="{{ASAN_OPTS}}:detect_leaks=0" UBSAN_OPTIONS="{{UBSAN_OPTS}}" \
          gdb -q -batch -ex "set pagination off" -ex "set print frame-arguments none" \
          -ex "run < {{INP}}" -ex "bt 32" --args ./{{APP_DBG}} 2>&1 )"; \
        echo "$out" | sed -n 's/.* at \(.*\.cpp:[0-9]\+\).*/\1/p' | head -n1 || printf "%s\n" "$out"; \
        exit 2; \
    fi

run-interactive:
    @if [[ ! -f "{{PCH}}" ]]; then \
        mkdir -p "$(dirname "{{PCH}}")"; \
        printf '#include <bits/stdc++.h>\n' > "{{PCH}}"; \
    fi; \
    if [[ ! -f {{APP_DBG}} || {{APP_DBG}} -ot {{SRC}} ]]; then \
        echo "Compiling (ASAN)..."; \
        LDOPT=""; \
        if command -v mold >/dev/null 2>&1; then LDOPT="-fuse-ld=mold"; \
        elif command -v ld.lld >/dev/null 2>&1; then LDOPT="-fuse-ld=lld"; fi; \
        {{CXX}} -std=gnu++23 -O1 -g -pipe -DLOCAL -DDEBUG_AUTO_FLUSH \
          -D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS \
          -fno-omit-frame-pointer -fno-pie -no-pie \
          -fsanitize=address,undefined \
          $LDOPT -include "{{PCH}}" {{SRC}} -o {{APP_DBG}}; \
    fi; \
    LDPRE=""; \
    ASAN_SO="$(g++ -print-file-name=libasan.so)"; \
    if [[ "$ASAN_SO" != "libasan.so" && -n "$ASAN_SO" ]]; then LDPRE="LD_PRELOAD=$ASAN_SO"; fi; \
    ulimit -s 2097152; \
    env $LDPRE ASAN_OPTIONS="{{ASAN_OPTS}}" UBSAN_OPTIONS="{{UBSAN_OPTS}}" ./{{APP_DBG}}

new:
    @cp -f "{{TEMPLATE}}" "{{SRC}}"
    @: > "{{INP}}"

clean:
    @rm -f "{{APP}}" "{{APP_DBG}}"

