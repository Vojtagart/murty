#!/usr/bin/env bash
set -e

BUILD_DIR=${1:-"build"}

DENSE_BIN="${BUILD_DIR}/benchmark/benchmark_dense"
SPARSE_BIN="${BUILD_DIR}/benchmark/benchmark_sparse"

if [ ! -d "${BUILD_DIR}" ]; then
    echo "[ERROR] Build directory '${BUILD_DIR}' does not exist. Run 'scripts/build.sh' first."
    exit 1
fi

echo "=================================================="
echo " Running Dense Benchmark"
echo "=================================================="
if [ -f "${DENSE_BIN}" ]; then
    "${DENSE_BIN}"
else
    echo "[ERROR] Executable '${DENSE_BIN}' not found."
    exit 1
fi

echo "=================================================="
echo " Running Sparse Benchmark"
echo "=================================================="
if [ -f "${SPARSE_BIN}" ]; then
    "${SPARSE_BIN}"
else
    echo "[ERROR] Executable '${SPARSE_BIN}' not found."
    exit 1
fi