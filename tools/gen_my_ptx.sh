#!/usr/bin/env bash
# Compile every problems/*/*.ir.json to <entry>.my.ptx next to it with our compiler
#
# If compilation fails, the .my.ptx file just contains "Failed" (stderr goes to the
# terminal). Always overwrites.
#
# usage: tools/gen_my_ptx.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
COMPILER="${COMPILER:-compile.py}"
ok=0
failed=0

# Open problems
for ir in problems/*/*.ir.json; do
    ptx="${ir%.ir.json}.my.ptx"
    if "$PYTHON" "$COMPILER" "$ir" > "$ptx"; then
        echo "wrote  $ptx"
        ok=$((ok + 1))
    else
        echo "Failed" > "$ptx"
        echo "Failed $ptx"
        failed=$((failed + 1))
    fi
done

echo "Open Problems: $ok ok, $failed failed"

ok=0
failed=0
for ir in problems/closed/*/*.ir.json; do
    ptx="${ir%.ir.json}.my.ptx"
    if "$PYTHON" "$COMPILER" "$ir" > "$ptx"; then
        echo "wrote  $ptx"
        ok=$((ok + 1))
    else
        echo "Failed" > "$ptx"
        echo "Failed $ptx"
        failed=$((failed + 1))
    fi
done

echo "Closed Problems: $ok ok, $failed failed"
