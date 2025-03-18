# Benchmarking

## Installation

1. Install [google/benchmark](https://github.com/google/benchmark).
2. Build the benchmarking executables:
```bash
mkdir build && cd build
cmake .. -GNinja && ninja
```

To reduce the variance, consider setting up your system appropriately [following these instructions](https://github.com/google/benchmark/blob/main/docs/reducing_variance.md).

## Running the benchmarks

Cost functions:
```bash
./benchmark_cost_functions --benchmark_display_aggregates_only=true --benchmark_repetitions=50
```
