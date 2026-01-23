# Connected Components — GPU (CUDA) Implementation

<p align="center">
  <img src="https://img.shields.io/badge/acceleration-CUDA-green" alt="CUDA">
</p>

Assignment #3 of the **Parallel and Distributed Systems** coursework:  
[parallel-distributed-systems](https://github.com/dimgerasimou/parallel-distributed-systems)

A **GPU-based connected components detection** project targeting **single-node NVIDIA GPUs** using **CUDA**, with a strong emphasis on performance analysis, memory behavior, and reproducible benchmarking.

---

## Overview

This project studies the connected components problem on GPUs using CUDA.

The main objectives are:
- Exploit massive GPU parallelism for graph analytics
- Evaluate different CUDA work-distribution strategies
- Compare GPU implementations against a CPU baseline
- Analyze performance bottlenecks on real-world sparse graphs

The implementation and evaluation closely follow the methodology described in the accompanying technical report.

---

## Features

- CUDA-based connected components implementations
- CPU **sequential baseline** for correctness and comparison
- Multiple CUDA variants exploring different execution strategies:
  - **Warp-per-row**
  - **Block-per-row**
  - **Afforest-inspired GPU variant**
- Sparse graph representation optimized for GPU access
- Automated benchmarking with warmup and trial runs
- Detailed **JSON output** including timing, throughput, and memory usage

---

## Algorithm

The GPU implementations are based on **iterative label propagation**, adapted for execution on CUDA-capable devices.

High-level algorithm:
1. Initialize one label per vertex
2. Iteratively relax labels along edges in parallel
3. Use atomic operations to enforce correctness
4. Detect convergence when no labels change
5. Compute the final number of connected components

Each CUDA variant differs in how work is assigned to threads (e.g. warp-level vs block-level parallelism), allowing exploration of the trade-offs between parallelism, atomic contention, and memory locality.

---

## Graph Representation

Graphs are stored in a sparse matrix format derived from **Matrix Market (`.mtx`)** inputs.

Internally, graphs are converted to compressed sparse representations suitable for GPU execution, balancing memory footprint and access efficiency.

---

## Build

### Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit
- C compiler (`gcc` or `clang`)
- `make`

### Compile
```bash
make
```

Produces:
```
bin/connected_components_cuda
```

---

## Usage

```bash
./bin/connected_components_cuda [options] matrix.mtx
```

Available options control:
- Implementation variant
- Number of trials
- Number of warmup runs
- CPU vs GPU execution

Run with `-h` to see all supported options.

---

## Experimental Setup

Benchmarks are executed using:
- Multiple warmup iterations to eliminate startup effects
- Multiple timed trials per configuration
- Large real-world graphs (e.g. LiveJournal, Orkut, Hollywood, MAWI)

All measurements are reported in a machine-readable JSON format for post-processing and plotting.

---

## Performance Results

Detailed benchmark results and plots are provided in:

```
performance.md
```

The analysis focuses on:
- GPU speedup over the CPU baseline
- Impact of graph structure on convergence
- Memory-bandwidth and atomic-operation limits
- Scalability within single-GPU memory constraints

---

## Notes on Performance

- Performance is typically **memory-bound**
- Atomic contention becomes significant on high-degree vertices
- Different CUDA variants excel on different graph structures
- GPU memory capacity limits the maximum graph size

---

<p align="center"><sub>January 2026 • Aristotle University of Thessaloniki</sub></p>
