#!/usr/bin/env bash
set -e

BUILD_DIR=${1:-"build"}

BIN="${BUILD_DIR}/experiments/experiment_sparsification"

if [ ! -d "${BUILD_DIR}" ]; then
    echo "[ERROR] Build directory '${BUILD_DIR}' does not exist. Run 'scripts/build.sh' first."
    exit 1
fi

echo "=================================================="
echo " Running Experiment"
echo "=================================================="
if [ -f "${BIN}" ]; then
    "${BIN}"
else
    echo "[ERROR] Executable '${BIN}' not found."
    exit 1
fi
