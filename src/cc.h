/**
 * @file cc.h
 * @brief Connected components counting algorithms for sparse binary matrices.
 *
 * Provides multiple implementations of connected components algorithms:
 * - Sequential CPU implementation using union-find
 * - CUDA GPU implementations with different parallelization strategies
 *
 * All algorithms operate on undirected graphs represented as sparse
 * binary matrices in CSC format.
 */

#ifndef CC_H
#define CC_H

#include "benchmark.h"
#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------- */
/*                    Connected Components Implementations                   */
/* ------------------------------------------------------------------------- */

/**
 * @brief Sequential CPU connected components using union-find.
 *
 * Strategy: Union-find with path halving optimization.
 * Single-threaded algorithm suitable for baseline comparison.
 *
 * @param[in] mtx Sparse binary matrix in CSC format (adjacency matrix).
 *
 * @return Number of connected components, or -1 on error.
 */
int connected_components_sequential(const Matrix *mtx);

/**
 * @brief Connected components using one CUDA thread per vertex.
 *
 * Strategy: Each thread processes one vertex, iterating until convergence.
 * Grid configuration: One thread per vertex.
 * Best for graphs with uniform vertex degrees.
 *
 * @param[in] mtx Sparse binary matrix in CSC format (adjacency matrix).
 *
 * @return Number of connected components, or -1 on error.
 */
int connected_components_cuda_thread_per_vertex(const Matrix *mtx);

/**
 * @brief Connected components using one warp per row.
 *
 * Strategy: Each warp (32 threads) collaboratively processes one vertex.
 * Threads in a warp work together on neighbors of the same vertex.
 * Best for medium-degree vertices (roughly 32-1024 neighbors).
 *
 * @param[in] mtx Sparse binary matrix in CSC format (adjacency matrix).
 *
 * @return Number of connected components, or -1 on error.
 */
int connected_components_cuda_warp_per_row(const Matrix *mtx);

/**
 * @brief Connected components using one block per row.
 *
 * Strategy: Each thread block processes one vertex using shared memory.
 * All threads in block collaborate on neighbors of the same vertex.
 * Best for high-degree vertices (1024+ neighbors).
 *
 * @param[in] mtx Sparse binary matrix in CSC format (adjacency matrix).
 *
 * @return Number of connected components, or -1 on error.
 */
int connected_components_cuda_block_per_row(const Matrix *mtx);

/**
 * @brief Connected components using the Afforest algorithm.
 *
 * Strategy: Hybrid approach combining sampling and neighbor processing.
 * Adapts to graph structure for improved performance.
 *
 * @param[in] mtx Sparse binary matrix in CSC format (adjacency matrix).
 *
 * @return Number of connected components, or -1 on error.
 */
int connected_components_cuda_afforest(const Matrix *mtx);

/* ------------------------------------------------------------------------- */
/*                         CUDA Utility Functions                            */
/* ------------------------------------------------------------------------- */

/**
 * @brief Captures CUDA device information.
 *
 * Queries the first CUDA device (device 0) for properties including
 * name, compute capability, and VRAM size.
 *
 * @param[out] info CudaDeviceInfo struct to be populated.
 *
 * @return 0 on success, non-zero on failure.
 */
int get_cuda_device_info(CudaDeviceInfo *info);

/**
 * @brief Captures peak GPU memory usage during a trial.
 *
 * Records the maximum GPU memory allocation that occurred during
 * the current trial.
 *
 * @param[out] result Result struct to populate with memory metrics.
 *
 * @return 0 on success, non-zero on failure.
 */
int set_cuda_memory_metrics(Result *result);

/* ------------------------------------------------------------------------- */
/*                         Algorithm Dispatcher                              */
/* ------------------------------------------------------------------------- */

/**
 * @brief Dispatcher to execute correct algorithm based on implementation type.
 *
 * Selects and runs the appropriate connected components implementation
 * based on the provided implementation index.
 *
 * @param[in] mtx Sparse binary matrix in CSC format.
 * @param[in] im  Implementation index (IMPL_SEQUENTIAL, IMPL_CUDA_*, etc).
 *
 * @return Number of connected components, or -1 on error.
 */
static inline int
connected_components(const Matrix *mtx, const unsigned int im)
{
	switch (im) {
	case IMPL_SEQUENTIAL:
		return connected_components_sequential(mtx);
	
	case IMPL_CUDA_THREAD_PER_VERTEX:
		return connected_components_cuda_thread_per_vertex(mtx);
	
	case IMPL_CUDA_WARP_PER_ROW:
		return connected_components_cuda_warp_per_row(mtx);
	
	case IMPL_CUDA_BLOCK_PER_ROW:
		return connected_components_cuda_block_per_row(mtx);
	
	case IMPL_CUDA_AFFOREST:
		return connected_components_cuda_afforest(mtx);
	
	default:
		break;
	}

	return -1;
}

#ifdef __cplusplus
}
#endif

#endif /* CC_H */