/**
 * @file cc.h
 * @brief Connected components counting algorithms for sparse binary matrices.
 */

#ifndef CC_H
#define CC_H

#include "benchmark.h"
#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

int connected_components_sequential(const Matrix *mtx);

/**
 * @brief Connected components using one thread per vertex.
 * 
 * Strategy: Each thread processes one vertex, iterating until convergence.
 * Grid configuration: One thread per vertex.
 * 
 * @param[in] mtx  Sparse binary matrix in CSC format (adjacency matrix).
 * @return Number of connected components, or -1 on error.
 */
int connected_components_cuda_thread_per_vertex(const Matrix *mtx);

/**
 * @brief Connected components using one warp per row.
 * 
 * Strategy: Each warp (32 threads) collaboratively processes one vertex.
 * Threads in a warp work together on neighbors of the same vertex.
 * Good for medium-degree vertices.
 * 
 * @param[in] mtx  Sparse binary matrix in CSC format (adjacency matrix).
 * @return Number of connected components, or -1 on error.
 */
int connected_components_cuda_warp_per_row(const Matrix *mtx);

/**
 * @brief Connected components using one block per row.
 * 
 * Strategy: Each thread block processes one vertex using shared memory.
 * All threads in block collaborate on neighbors of the same vertex.
 * Good for high-degree vertices.
 * 
 * @param[in] mtx  Sparse binary matrix in CSC format (adjacency matrix).
 * @return Number of connected components, or -1 on error.
 */
int connected_components_cuda_block_per_row(const Matrix *mtx);

int connected_components_cuda_afforest(const Matrix *mtx);

/**
 * @brief Captures in struct cuda device info.
 *
 * @param[out] info  CudaDeviceInfo struct to be populated.
 *
 * @return     0 on success, non-zero on failure.
 */
int getcudadeviceinfo(CudaDeviceInfo *info);

/**
 * @brief Dispatcher to execute correct algorithm based on implementation type.
 *
 * @param[in] mtx  Sparse binary matrix in CSC format.
 * @param[in] im   Implementation to run.
 *
 * @return  Number of connected components, or -1 on error
 */
static inline int connected_components(const Matrix *mtx, const unsigned int im) {
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
