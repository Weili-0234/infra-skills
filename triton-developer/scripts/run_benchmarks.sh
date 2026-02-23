#!/bin/bash

# Run all Triton kernel benchmark scripts and save results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-.}"

echo "=========================================="
echo "Running Triton kernel benchmarks"
echo "=========================================="
echo "Script directory: $SCRIPT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Track failed benchmarks
declare -a FAILED_BENCHMARKS

# Find and run all benchmark scripts
benchmark_count=0
for file in "$SCRIPT_DIR"/bench_*.py; do
    if [ -f "$file" ]; then
        benchmark_count=$((benchmark_count + 1))
        benchmark_name=$(basename "$file" .py)
        output_file="$OUTPUT_DIR/${benchmark_name}_results.txt"

        echo "[$benchmark_count] Running $benchmark_name..."

        # Run benchmark and capture output
        if python3 "$file" 2>&1 | tee "$output_file"; then
            echo "  $benchmark_name completed successfully"
        else
            echo "  $benchmark_name failed"
            FAILED_BENCHMARKS+=("$benchmark_name")
        fi

        echo ""
    fi
done

if [ $benchmark_count -eq 0 ]; then
    echo "No benchmark scripts found (looking for bench_*.py in $SCRIPT_DIR)"
    echo "Create benchmark scripts named bench_<name>.py"
    exit 0
fi

# Summary
echo "=========================================="
echo "Benchmark Summary"
echo "=========================================="
echo "Total benchmarks run: $benchmark_count"

if [ ${#FAILED_BENCHMARKS[@]} -eq 0 ]; then
    echo "All benchmarks completed successfully"
    exit 0
else
    echo "${#FAILED_BENCHMARKS[@]} benchmark(s) failed:"
    for failed in "${FAILED_BENCHMARKS[@]}"; do
        echo "  - $failed"
    done
    exit 1
fi
