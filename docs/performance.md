# Performance

This report summarizes the benchmark runs produced by the current implementation.

## Benchmark Environment

All benchmarks were executed on a Google Colab environment with a single GPU.

- **CPU**: Intel(R) Xeon(R) CPU @ 2.00 GHz
- **RAM**: ~12.7 GiB
- **GPU**: NVIDIA Tesla T4 (Compute Capability 7.5)
- **VRAM**: ~14.7 GiB

## Datasets

> **Note**: Reported `nnz` values correspond to **half the actual edge count**, since graphs are treated as undirected.

| Name              | Nodes       | nnz         | Edges       |
|-------------------|------------:|------------:|------------:|
| dictionary28      | 52,652      | 89,038      | 178,076     |
| hollywood-2009    | 1,139,905   | 56,375,711  | 112,751,422 |
| com-LiveJournal   | 3,997,962   | 34,681,189  | 69,362,378  |
| com-Orkut         | 3,072,441   | 117,185,083 | 234,370,166 |
| mawi_201512020330 | 226,196,185 | 240,023,945 | 480,047,890 |

## Results

Speedup Calculation:

Speedup = (CPU mean time) / (Implementation mean time)

### dictionary28

| Implementation           | Mean time (ms) | Speedup vs CPU | Throughput (MEdges/s) |
|--------------------------|---------------:|---------------:|----------------------:|
| CPU: Sequential          |          1.041 |          1.00× |                 171.1 |
| CUDA: Thread-per-Vertex  |          0.809 |          1.29× |                 220.1 |
| CUDA: Warp-per-Row       |          0.782 |          1.33× |                 227.7 |
| CUDA: Block-per-Row      |          0.796 |          1.31× |                 223.7 |
| CUDA: Afforest           |          1.006 |          1.03× |                 177.0 |

Very small graph; kernel launch overhead dominates.

**Observation**:

GPU versions offer marginal benefit; Warp-per-Row performs best but gains are limited by graph size.

### hollywood-2009

| Implementation           | Mean time (ms) | Speedup vs CPU | Throughput (MEdges/s) |
|--------------------------|---------------:|---------------:|----------------------:|
| CPU: Sequential          |          231.4 |          1.00× |                 487.3 |
| CUDA: Thread-per-Vertex  |           61.6 |          3.76× |               1,830.9 |
| CUDA: Warp-per-Row       |           57.8 |          4.01× |               1,951.7 |
| CUDA: Block-per-Row      |           61.1 |          3.79× |               1,846.5 |
| CUDA: Afforest           |           62.0 |          3.73× |               1,818.4 |

Medium-sized graph with good GPU utilization.

**Observation**:

Warp-per-Row is consistently the fastest, achieving ~4× speedup over CPU.

### com-LiveJournal

| Implementation           | Mean time (ms) | Speedup vs CPU | Throughput (MEdges/s) |
|--------------------------|---------------:|---------------:|----------------------:|
| CPU: Sequential          |          207.8 |          1.00× |                 333.8 |
| CUDA: Thread-per-Vertex  |           42.2 |          4.93× |               1,645.5 |
| CUDA: Warp-per-Row       |           45.3 |          4.59× |               1,531.0 |
| CUDA: Block-per-Row      |           58.9 |          3.53× |               1,178.5 |
| CUDA: Afforest           |           67.1 |          3.10× |               1,033.9 |

Large social network; GPU-friendly sparsity pattern.

**Observation**:

Thread-per-Vertex slightly outperforms Warp-per-Row here, likely due to better vertex-level
parallelism and memory coalescing.

### com-Orkut

| Implementation           | Mean time (ms) | Speedup vs CPU | Throughput (MEdges/s) |
|--------------------------|---------------:|---------------:|----------------------:|
| CPU: Sequential          |          620.2 |          1.00× |                 377.9 |
| CUDA: Thread-per-Vertex  |          132.7 |          4.67× |               1,766.4 |
| CUDA: Warp-per-Row       |          131.0 |          4.73× |               1,789.0 |
| CUDA: Block-per-Row      |          139.8 |          4.44× |               1,677.0 |
| CUDA: Afforest           |          145.7 |          4.26× |               1,608.1 |

Denser social graph with higher memory pressure.

**Observation**:

Warp-per-Row regains the lead on denser graphs, showing better load balance across warps.

### mawi_201512020330

| Implementation           | Mean time (s) | Speedup vs CPU | Throughput (MEdges/s) |
|--------------------------|--------------:|---------------:|----------------------:|
| CPU: Sequential          |         2.255 |          1.00× |                 212.9 |
| CUDA: Thread-per-Vertex  |        53.821 |          0.04× |                 8.919 |
| CUDA: Warp-per-Row       |         3.303 |          0.68× |                 145.3 |
| CUDA: Block-per-Row      |         1.370 |          1.65× |                 350.5 |
| CUDA: Afforest           |         1.704 |          1.32× |                 281.7 |

Extremely large graph stressing both memory and traversal depth.

**Observation:**

Thread-per-Vertex collapses completely due to extreme imbalance.

Block-per-Row is the clear winner, outperforming even the CPU baseline.

### Overall observations

- Warp-per-Row is the most consistently strong GPU strategy across datasets.
- Thread-per-Vertex performs well on mid-sized graphs but fails catastrophically on MAWI.
- Block-per-Row is the most robust choice for very large graphs.
- Afforest is competitive but rarely the fastest in this GPU setting.
- GPU acceleration delivers 3–5× speedup on realistic social-network graphs.
