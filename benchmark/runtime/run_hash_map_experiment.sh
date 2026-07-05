#!/usr/bin/env bash
#
# Build-matrix driver for the hash map backend experiment.
#
# For each backend (STD, BOOST) this script configures a dedicated Release build
# directory, builds the incremental mapping benchmark, and runs it, emitting one
# JSON result file per backend. It then prints a comparison of end-to-end
# runtime and peak RSS relative to the STD baseline.
#
# Usage:
#   benchmark/runtime/run_hash_map_experiment.sh [backend ...]
#
# Environment:
#   COLMAP_BENCHMARK_DATABASE_PATH  Real database to benchmark
#                                   (default: ~/data/south-building/database.db)
#   BACKENDS                        Space-separated backends (default: STD BOOST)
#   BENCH_FILTER                    Google-benchmark --benchmark_filter regex
#   CMAKE_EXTRA_ARGS                Extra args forwarded to the configure step
#
# Run from the repository root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

BACKENDS="${*:-${BACKENDS:-STD BOOST}}"
RESULT_DIR="${REPO_ROOT}/benchmark_hashmap_results"
mkdir -p "${RESULT_DIR}"

# Lean, production-representative build: Release + IPO, no GUI/CUDA/ONNX/tests.
COMMON_CMAKE_ARGS=(
  -GNinja
  -DCMAKE_BUILD_TYPE=Release
  -DBENCHMARK_ENABLED=ON
  -DTESTS_ENABLED=OFF
  -DGUI_ENABLED=OFF
  -DCUDA_ENABLED=OFF
  -DONNX_ENABLED=OFF
  -DCGAL_ENABLED=OFF
  -DMVS_ENABLED=OFF
)

for backend in ${BACKENDS}; do
  build_dir="${REPO_ROOT}/build_hash_${backend}"
  echo "=============================================================="
  echo " Configuring + building backend: ${backend}"
  echo "=============================================================="
  cmake -S . -B "${build_dir}" \
    "${COMMON_CMAKE_ARGS[@]}" \
    -DCOLMAP_HASH_MAP_BACKEND="${backend}" \
    ${CMAKE_EXTRA_ARGS:-}
  cmake --build "${build_dir}" --target benchmark_incremental_mapping -j

  bench_bin="${build_dir}/benchmark/runtime/benchmark_incremental_mapping"
  echo "--------------------------------------------------------------"
  echo " Running benchmark: ${backend}"
  echo "--------------------------------------------------------------"
  "${bench_bin}" \
    ${BENCH_FILTER:+--benchmark_filter="${BENCH_FILTER}"} \
    --benchmark_out="${RESULT_DIR}/results_${backend}.json" \
    --benchmark_out_format=json
done

echo
echo "=============================================================="
echo " Summary (see ${RESULT_DIR}/results_*.json for full detail)"
echo "=============================================================="
python3 "${REPO_ROOT}/benchmark/runtime/summarize_hash_map_experiment.py" \
  "${RESULT_DIR}"/results_*.json || {
  echo "(Install python3 to render the comparison table; raw JSON is in ${RESULT_DIR})"
}
