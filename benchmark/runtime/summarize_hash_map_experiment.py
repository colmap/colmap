#!/usr/bin/env python3
# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Summarize the hash map backend experiment: given the per-backend
# google-benchmark JSON files produced by run_hash_map_experiment.sh, print a
# comparison of end-to-end runtime and peak RSS per benchmark, relative to the
# STD baseline.
#
# Usage: summarize_hash_map_experiment.py results_STD.json results_BOOST.json ...

import json
import os
import sys
from collections import defaultdict


def backend_from_path(path):
    name = os.path.basename(path)
    for prefix, suffix in (("results_", ".json"),):
        if name.startswith(prefix) and name.endswith(suffix):
            return name[len(prefix):-len(suffix)]
    return name


def load(path):
    """Return {benchmark_name: {"time_ms": float, "peak_rss_mb": float,
    "num_reg_images": float}} using the mean aggregate when available."""
    with open(path) as f:
        data = json.load(f)

    # Prefer the *_mean aggregate; fall back to the single iteration.
    chosen = {}
    for bench in data.get("benchmarks", []):
        run_name = bench.get("run_name", bench.get("name", ""))
        aggregate = bench.get("aggregate_name", "")
        is_mean = aggregate == "mean"
        is_plain = bench.get("run_type", "iteration") == "iteration"
        if run_name not in chosen or is_mean:
            if is_mean or (is_plain and run_name not in chosen):
                chosen[run_name] = {
                    "time_ms": bench.get("real_time", float("nan")),
                    "peak_rss_mb": bench.get("peak_rss_mb", float("nan")),
                    "num_reg_images": bench.get("num_reg_images", float("nan")),
                }
    return chosen


def fmt_ratio(value, baseline):
    if baseline is None or baseline != baseline or baseline == 0:
        return "   n/a"
    return f"{value / baseline:5.2f}x"


def main(paths):
    results = {}  # backend -> {bench -> metrics}
    for path in paths:
        results[backend_from_path(path)] = load(path)

    backends = sorted(results, key=lambda b: (b != "STD", b))
    benches = sorted({b for r in results.values() for b in r})
    baseline = "STD" if "STD" in results else backends[0]

    print(f"\nBaseline: {baseline}   (lower ratio = faster / less memory)\n")
    for bench in benches:
        print(bench)
        base = results.get(baseline, {}).get(bench)
        header = f"  {'backend':<8} {'time_ms':>12} {'vs base':>9} " \
                 f"{'peak_rss_mb':>12} {'vs base':>9} {'reg_imgs':>9}"
        print(header)
        for backend in backends:
            m = results[backend].get(bench)
            if m is None:
                print(f"  {backend:<8} {'(missing)':>12}")
                continue
            base_t = base["time_ms"] if base else None
            base_r = base["peak_rss_mb"] if base else None
            print(
                f"  {backend:<8} {m['time_ms']:>12.1f} "
                f"{fmt_ratio(m['time_ms'], base_t):>9} "
                f"{m['peak_rss_mb']:>12.1f} "
                f"{fmt_ratio(m['peak_rss_mb'], base_r):>9} "
                f"{m['num_reg_images']:>9.0f}"
            )
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: summarize_hash_map_experiment.py results_*.json")
    main(sys.argv[1:])
