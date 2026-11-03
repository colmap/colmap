#!/usr/bin/env bash
# Mutation test for the cusift matcher: prove the test suite can actually FAIL.
#
# WHY THIS EXISTS. Two real bugs shipped through a green suite:
#   1. the cross-check only confirmed mutual-best agreement, omitting the
#      reverse-direction ratio test that both of COLMAP's matchers apply;
#   2. descriptors were L2-normalized per row, where the reference ranks on the
#      raw inner product with a FIXED 512^2 constant.
# Both survived because the fixtures sampled only unambiguous, uniform-norm
# descriptors -- the regime where neither bug changes a decision. One of the
# tests even asserted `strict subset-of loose`, which passes vacuously when the
# two modes agree.
#
# A test that cannot fail is not evidence. This script reintroduces each bug and
# asserts the suite goes red, then restores the source and asserts it goes green.
# Run it after touching the matcher's decision logic, and especially before the
# integer-GEMM follow-up, which introduces new threshold arithmetic.
#
# Usage: scripts/shell/cusift_matcher_mutation_test.sh [build_dir]
#
# Not registered with ctest deliberately: it EDITS SOURCE FILES and rebuilds, so
# running it inside a parallel `ctest -j` would race the build directory. Invoke
# it by hand after touching the matcher's decision logic.
set -uo pipefail

BUILD_DIR="${1:-$HOME/work/colmap/build_fresh_verify}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WRAPPER="$REPO_ROOT/src/colmap/feature/cusift_matcher.cc"
KERNEL="$REPO_ROOT/src/thirdparty/cusift/src/matcher_gemm.cu"

# Fail loudly rather than silently mutating nothing if the layout moves.
for f in "$WRAPPER" "$KERNEL"; do
    [ -f "$f" ] || { echo "cannot find $f -- has the tree layout changed?" >&2; exit 1; }
done
TEST_BIN="$BUILD_DIR/src/colmap/feature/cusift_matcher_test"
TMP="$(mktemp -d)"
trap 'cp -f "$TMP/cusift_matcher.cc" "$WRAPPER" 2>/dev/null;
      cp -f "$TMP/matcher_gemm.cu" "$KERNEL" 2>/dev/null; rm -rf "$TMP"' EXIT

cp "$WRAPPER" "$TMP/cusift_matcher.cc"
cp "$KERNEL" "$TMP/matcher_gemm.cu"

build() { ninja -C "$BUILD_DIR" -j "$(nproc)" cusift_matcher_test >"$TMP/build.log" 2>&1; }
restore() { cp -f "$TMP/cusift_matcher.cc" "$WRAPPER"; cp -f "$TMP/matcher_gemm.cu" "$KERNEL"; }

fail=0

# Each mutation names the test(s) that MUST catch it. A mutation that builds but
# leaves the suite green is a coverage hole, reported as such.
run_mutation() {
    local name="$1" expect="$2" file="$3"
    echo "--- mutation: $name  (must be caught by: $expect)"
    # A replacement whose pattern no longer matches leaves the source untouched,
    # the suite green, and the result indistinguishable from a coverage hole.
    # Refuse to interpret that as a finding.
    if cmp -s "$file" "$TMP/$(basename "$file")"; then
        echo "    *** MUTATION DID NOT APPLY: pattern is stale, fix the script ***"
        fail=1; restore; return
    fi
    if ! build; then
        echo "    SKIP: mutation does not compile"
        restore; return
    fi
    if "$TEST_BIN" >"$TMP/test.log" 2>&1; then
        echo "    *** COVERAGE HOLE: suite still PASSES with this bug injected ***"
        fail=1
    else
        local caught
        caught=$(grep -oE "^\[  FAILED  \] [A-Za-z0-9_/.]+" "$TMP/test.log" |
                 sed 's/^\[  FAILED  \] //' | sort -u | tr '\n' ' ')
        echo "    OK: suite fails. Caught by: ${caught:-<unnamed>}"
    fi
    restore
}

# STRUCTURAL GUARD: a vanishing test cannot be caught by any assertion inside a
# test. This bit us for real -- a block replacement whose start and end anchors
# were different tests silently deleted six of them, and the suite stayed green
# with fewer tests. Pin the count. Raising it must be a deliberate edit, and the
# annoyance of having to edit it IS the alarm.
EXPECTED_TEST_CASES=20

echo "=== structural guard: test-case count ==="
actual=$(cd "$BUILD_DIR" && "$TEST_BIN" --gtest_list_tests 2>/dev/null |
         grep -cE "^  [A-Za-z]")
if [ "$actual" != "$EXPECTED_TEST_CASES" ]; then
    echo "    *** TEST COUNT CHANGED: expected $EXPECTED_TEST_CASES, found $actual ***"
    if [ "$actual" -lt "$EXPECTED_TEST_CASES" ]; then
        echo "    Tests were REMOVED. Restore them (git show <rev>:<path>) before"
        echo "    continuing -- removal needs explicit sign-off, never a silent diff."
        fail=1
    else
        echo "    Tests were added. If intended, bump EXPECTED_TEST_CASES to $actual."
        fail=1
    fi
else
    echo "    OK: $actual test cases, as pinned"
fi

echo "=== baseline: unmutated source must PASS ==="
build || { echo "baseline build FAILED"; exit 1; }
if "$TEST_BIN" >"$TMP/test.log" 2>&1; then
    echo "    OK: $(grep -oE '\[  PASSED  \] [0-9]+ test' "$TMP/test.log" | head -1)"
else
    echo "    baseline is already RED -- fix that before mutation testing"; exit 1
fi

# BUG 1: drop the reverse-direction ratio test, leaving mutual-best only.
python3 - "$KERNEL" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
s = s.replace("      if (reverseRatio && ratio_fails(", "      if (false && ratio_fails(")
s = s.replace("      if (reverseRatio) {", "      if (false) {")
p.write_text(s)
PY
run_mutation "cross-check reduced to mutual-best only (bug 1)" \
    "ReverseRatioRejectsAmbiguousReverseDirection, DifferentialAgainstExactCpuBruteForce" \
    "$KERNEL"

# BUG 2: rank on normalized distance instead of the raw inner product, i.e.
# always take the L2 key path regardless of the requested distance type.
python3 - "$KERNEL" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
s = s.replace("angular_mode ? nullptr : d2Sq", "d2Sq")
s = s.replace("angular_mode ? nullptr : d1Sq", "d1Sq")
p.write_text(s)
PY
run_mutation "ranking switched to normalized distance (bug 2)" \
    "DifferentialAgainstExactCpuBruteForce, NonCanonicalNormsDoNotChangeTheDecision" \
    "$KERNEL"

# BUG 3: max_num_matches truncation re-sorted by score instead of preserving
# ascending image1-index order, which is SiftMatchGPU's cap-binds rule.
python3 - "$WRAPPER" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
s = s.replace("""    if (raw_matches.size() >
        static_cast<size_t>(options_.max_num_matches)) {""",
"""    std::sort(raw_matches.begin(), raw_matches.end(),
              [](const ::Match& a, const ::Match& b) { return a.dist2 < b.dist2; });
    if (raw_matches.size() >
        static_cast<size_t>(options_.max_num_matches)) {""")
p.write_text(s)
PY
run_mutation "max_num_matches truncation re-sorted by score (bug 3)" \
    "MaxNumMatchesTruncatesByIndexOrder" \
    "$WRAPPER"

# BUG 4: revert the canonical tie-break to the old order-dependent rule -- resolve
# equal keys by visitation order rather than by index. That is what the column
# reduction did before Phase 1: ties went to the lowest threadIdx.x, i.e. the
# lowest (r % 256), not the lowest r.
python3 - "$KERNEL" <<'PYEOF'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
s = s.replace("return v > b || (v == b && (bi < 0 || i < bi));", "return v > b;")
p.write_text(s)
PYEOF
run_mutation "canonical tie-break reverted to order-dependent (bug 4)" \
    "ColumnTieBreakPrefersLowerRowIndex, ColumnTieBreakIsKeyModeIndependent" \
    "$KERNEL"

# BUG 5: invert the preference -- prefer the HIGHER index. Catches a test that
# only asserts "some deterministic winner" instead of which one.
python3 - "$KERNEL" <<'PYEOF'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
s = s.replace("return v > b || (v == b && (bi < 0 || i < bi));",
              "return v > b || (v == b && (bi < 0 || i > bi));")
p.write_text(s)
PYEOF
run_mutation "tie preference inverted to higher index (bug 5)" \
    "ColumnTieBreakPrefersLowerRowIndex, ColumnTieBreakIsKeyModeIndependent" \
    "$KERNEL"

echo "=== restoring and re-verifying ==="
restore
build || { echo "restore build FAILED"; exit 1; }
if "$TEST_BIN" >"$TMP/test.log" 2>&1; then
    echo "    OK: restored source PASSES"
else
    echo "    *** restored source is RED -- the script corrupted the tree ***"; fail=1
fi

if [ "$fail" = 0 ]; then
    echo; echo "MUTATION TEST PASSED: every injected bug was caught."
else
    echo; echo "MUTATION TEST FAILED: see coverage holes above."; exit 1
fi
