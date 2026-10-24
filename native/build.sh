#!/bin/bash
set -euo pipefail
LF_NATIVE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
LF_NATIVE_TMP=$(mktemp "$LF_NATIVE_DIR/kernel.XXXXXX.so")
trap 'rm -f -- "$LF_NATIVE_TMP"' EXIT
gcc -shared -fPIC -O3 -mavx2 -ffp-contract=off -fno-fast-math \
  "$LF_NATIVE_DIR/kernel.c" -o "$LF_NATIVE_TMP" -lm
mv -- "$LF_NATIVE_TMP" "$LF_NATIVE_DIR/kernel.so"
