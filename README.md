# Computational Benchmark Suite

A comprehensive benchmark that measures machine performance across different computational workloads while monitoring system metrics.

## 🚀 Quick Start

```bash
# Run medium workload (15-45 minutes)
python benchmark.py --workload medium --path ./results

# For SLURM jobs
python benchmark.py --workload large --path ./results_${SLURM_JOB_NAME}
```

## 📊 Benchmark Tasks

### 1. **Prime Factorization** 🔢
- **What**: Factors large semi-prime numbers using trial division
- **Bound**: **CPU-bound** 
- **Characteristics**: Pure computational work, minimal memory, no I/O
- **Workload**: 5K-100K factorizations depending on size

### 2. **Monte Carlo Pi Estimation** 🎯  
- **What**: Estimates π by random sampling points in unit circle
- **Bound**: **CPU-bound**
- **Characteristics**: Intensive random number generation, simple arithmetic
- **Workload**: 500M-50B random points depending on size

### 3. **Matrix Operations** 🧮
- **What**: Matrix multiplication, inversion, eigenvalues, SVD, QR decomposition  
- **Bound**: **CPU + Memory-bound**
- **Characteristics**: Heavy floating-point operations, large memory allocations
- **Workload**: 500-5K operations on 200×200 to 500×500 matrices

### 4. **Dataset Analysis** 📊
- **What**: Statistical analysis, FFT, correlation matrices on large datasets
- **Bound**: **Memory-bound**
- **Characteristics**: Large memory allocations (GB-scale), memory-intensive algorithms
- **Workload**: 100-1K analyses on 2M-20M data points each

## 🔧 Workload Sizes

| Size | Duration | Prime Factors | Monte Carlo | Matrix Ops | Dataset Size |
|------|----------|---------------|-------------|------------|--------------|
| `small` | 5-15 min | 5K | 500M points | 500 ops | 2M points |
| `medium` | 15-45 min | 20K | 2B points | 1K ops | 5M points |
| `large` | 45-120 min | 50K | 10B points | 2K ops | 10M points |
| `extreme` | 2-6 hours | 100K | 50B points | 5K ops | 20M points |

## 📁 Output Files

- **Machine snapshots**: `YYYYMMDD_HHMMSS.json` (every 10 seconds)
- **Benchmark results**: `benchmark_results.json` (final summary)
- **Performance score**: Higher = faster machine

## 🎯 Use Cases

- **Machine comparison**: Same workload across different systems
- **Performance monitoring**: Track system metrics during heavy computation  
- **Resource planning**: Understand CPU vs memory vs I/O bottlenecks
- **HPC benchmarking**: Validate cluster node performance
