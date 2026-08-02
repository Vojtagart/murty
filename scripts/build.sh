#!/usr/bin/env bash
set -e

BUILD_DIR=${1:-"build"}
BUILD_TYPE=${2:-"Release"}

IF_NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "=================================================="
echo " Configuring Murty Algorithm (${BUILD_TYPE} Mode)"
echo "=================================================="

cmake -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DMURTY_BUILD_BENCHMARKS=ON \
    -DMURTY_BUILD_TESTS=ON \
    -DMURTY_BUILD_EXPERIMENTS=ON

echo ""
echo "=================================================="
echo " Building Targets"
echo "=================================================="

cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" -j "${IF_NPROC}"

echo ""
echo "[SUCCESS] Build completed in '${BUILD_DIR}/'."