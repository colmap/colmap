# Benchmarking

## Installation

1. Install [google/benchmark](https://github.com/google/benchmark).
   For example, using homebrew on a Mac: `brew install google-benchmark`.
2. Build and run the benchmarking executables:

```bash
cmake .. -DBENCHMARK_ENABLED=ON
ninja benchmark/runtime/benchmark_cost_functions
```

To reduce the variance, consider setting up your system appropriately [following these instructions](https://github.com/google/benchmark/blob/main/docs/reducing_variance.md).

## Running the benchmarks

Cost functions:

```bash
./benchmark/runtime/benchmark_cost_functions --benchmark_display_aggregates_only=true --benchmark_repetitions=50
```
