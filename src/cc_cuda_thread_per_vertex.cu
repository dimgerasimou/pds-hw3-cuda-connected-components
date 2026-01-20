/**
 * @file cc_cuda_thread_per_vertex.cu
 * @brief CUDA implementation: One thread per vertex
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "cc.h"
#include "error.h"

#define CUDA_CHECK(call) do { \
	cudaError_t err = (call); \
	if (err != cudaSuccess) { \
		DERRF("CUDA error: %s", cudaGetErrorString(err)); \
		return -1; \
	} \
} while (0)

/**
 * @brief CUDA kernel: One thread per vertex
 *
 * Each thread processes its assigned vertex, checking all neighbors
 * and propagating the minimum label.
 *
 * @param[in]     colptr   Column pointers (ncols+1)
 * @param[in]     rowi     Row indices (nnz)
 * @param[in,out] label    Label array (nrows)
 * @param[out]    changed  Flag indicating if any label changed
 * @param[in]     nrows    Number of vertices
 */
__global__ void
cc_kernel_thread_per_vertex(const uint32_t *colptr, const uint32_t *rowi,
                            uint32_t *label, int *changed, uint32_t nrows)
{
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= nrows)
		return;

	uint32_t my_label  = label[tid];
	uint32_t min_label = my_label;

	uint32_t start = colptr[tid];
	uint32_t end   = colptr[tid + 1];

	for (uint32_t i = start; i < end; i++) {
		uint32_t neighbor       = rowi[i];
		uint32_t neighbor_label = label[neighbor];
		if (neighbor_label < min_label)
			min_label = neighbor_label;
	}

	if (min_label < my_label) {
		label[tid] = min_label;
		/* avoid data race (many threads may set changed) */
		atomicExch(changed, 1);
	}
}

/**
 * @brief Count unique components using bitmap on GPU
 *
 * NOTE: Use unsigned long long for atomicOr overload compatibility.
 */
__global__ void
cc_count_components(const uint32_t *label, unsigned long long *bitmap, uint32_t nrows)
{
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= nrows)
		return;

	uint32_t val = label[tid];
	unsigned int word = val >> 6;          /* /64 */
	unsigned int bit  = val & 63u;         /* %64 */

	atomicOr(&bitmap[word], 1ULL << bit);
}

/**
 * @brief Count set bits in bitmap using popcount
 */
__global__ void
cc_popcount(const unsigned long long *bitmap, uint32_t *count, uint32_t bitmap_size)
{
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= bitmap_size)
		return;

	/* __popcll returns number of set bits in 64-bit value */
	atomicAdd(count, (uint32_t)__popcll(bitmap[tid]));
}

int
connected_components_cuda_thread_per_vertex(const Matrix *mtx)
{
	if (!mtx || mtx->nrows != mtx->ncols) {
		DERRF("expected square adjacency matrix");
		return -1;
	}

	uint32_t nrows = mtx->nrows;
	uint32_t nnz   = mtx->nnz;

	uint32_t *d_colptr = NULL;
	uint32_t *d_rowi   = NULL;
	uint32_t *d_label  = NULL;
	int      *d_changed = NULL;

	CUDA_CHECK(cudaMalloc((void **)&d_colptr, (nrows + 1) * sizeof(uint32_t)));
	CUDA_CHECK(cudaMalloc((void **)&d_rowi,   nnz * sizeof(uint32_t)));
	CUDA_CHECK(cudaMalloc((void **)&d_label,  nrows * sizeof(uint32_t)));
	CUDA_CHECK(cudaMalloc((void **)&d_changed, sizeof(int)));

	CUDA_CHECK(cudaMemcpy(d_colptr, mtx->colptr, (nrows + 1) * sizeof(uint32_t),
	                      cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_rowi, mtx->rowi, nnz * sizeof(uint32_t),
	                      cudaMemcpyHostToDevice));

	uint32_t *h_label = (uint32_t *)malloc(nrows * sizeof(uint32_t));
	if (!h_label) {
		DERRNOF("malloc failed");
		cudaFree(d_colptr);
		cudaFree(d_rowi);
		cudaFree(d_label);
		cudaFree(d_changed);
		return -1;
	}

	for (uint32_t i = 0; i < nrows; i++)
		h_label[i] = i;

	CUDA_CHECK(cudaMemcpy(d_label, h_label, nrows * sizeof(uint32_t),
	                      cudaMemcpyHostToDevice));

	int threads_per_block = 256;
	int blocks = (int)((nrows + (uint32_t)threads_per_block - 1) / (uint32_t)threads_per_block);

	int h_changed;
	int iterations = 0;

	do {
		h_changed = 0;
		CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int),
		                      cudaMemcpyHostToDevice));

		cc_kernel_thread_per_vertex<<<blocks, threads_per_block>>>(
			d_colptr, d_rowi, d_label, d_changed, nrows
		);

		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int),
		                      cudaMemcpyDeviceToHost));

		iterations++;
	} while (h_changed);

	/* Count components using bitmap */
	uint32_t bitmap_size = (nrows + 63u) / 64u;
	unsigned long long *d_bitmap = NULL;
	uint32_t *d_count = NULL;

	CUDA_CHECK(cudaMalloc((void **)&d_bitmap, bitmap_size * sizeof(*d_bitmap)));
	CUDA_CHECK(cudaMalloc((void **)&d_count, sizeof(*d_count)));
	CUDA_CHECK(cudaMemset(d_bitmap, 0, bitmap_size * sizeof(*d_bitmap)));
	CUDA_CHECK(cudaMemset(d_count,  0, sizeof(*d_count)));

	blocks = (int)((nrows + (uint32_t)threads_per_block - 1) / (uint32_t)threads_per_block);
	cc_count_components<<<blocks, threads_per_block>>>(d_label, d_bitmap, nrows);

	blocks = (int)((bitmap_size + (uint32_t)threads_per_block - 1) / (uint32_t)threads_per_block);
	cc_popcount<<<blocks, threads_per_block>>>(d_bitmap, d_count, bitmap_size);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	uint32_t h_count = 0;
	CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(uint32_t),
	                      cudaMemcpyDeviceToHost));

	/* Cleanup */
	free(h_label);
	cudaFree(d_colptr);
	cudaFree(d_rowi);
	cudaFree(d_label);
	cudaFree(d_changed);
	cudaFree(d_bitmap);
	cudaFree(d_count);

	return (int)h_count;
}

