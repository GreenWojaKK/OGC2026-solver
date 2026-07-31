#!/usr/bin/env bash
# Build geoexact.so (linux x86-64, glibc forward-compatible via static
# libstdc++/libgcc).  -ffp-contract=off is REQUIRED for Shewchuk predicate
# exactness (no FMA contraction); do not add -ffast-math/-Ofast.
set -e
cd "$(dirname "$0")"
g++ -O2 -shared -fPIC -ffp-contract=off -fno-math-errno \
    -Wall -Wextra \
    -static-libstdc++ -static-libgcc \
    -o geoexact.so geoexact.cpp
cp geoexact.so ../../solver/geoexact.so
echo "== GLIBC version symbols (max must be <= server glibc 2.39) =="
objdump -T geoexact.so | grep -o 'GLIBC_[0-9.]*' | sort -Vu
echo "== GLIBCXX dynamic symbols (must be none: statically linked) =="
objdump -T geoexact.so | grep GLIBCXX || echo "(none)"
echo "== needed shared libs =="
readelf -d geoexact.so | grep NEEDED
ls -la geoexact.so
