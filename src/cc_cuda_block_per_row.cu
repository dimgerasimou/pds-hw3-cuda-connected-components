/**
 * @file cc_cuda_block_per_row.cu
 * @brief CUDA Connected Components: Union-Find (single-pass), Block-per-Row (scalable grid-stride)
 *
 * Strategy C:
 * - Each block cooperatively processes one row (vertex) at a time.
 * - Blocks iterate over rows in a grid-stride loop: u = blockIdx.x; u += gridDim.x
 * - Threads within the block stride over the adjacency list of row u.
 *
 * Correctness:
 * - Union uses atomicCAS on roots only (safe parallel UF).
 * - Compression runs until stable.
 * - Root counting counts only true roots (parent[i] == i), matching CPU.
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>

#include "cc.h"
#include "error.h"

#define CUDA_CHECK(call) do { \
	cudaError_t err = (call); \
	if (err != cudaSuccess) { \
		DERRF("CUDA error: %s", cudaGetErrorString(err)); \
		return -1; \
	} \
} while (0)

#ifndef UF_COMPRESS_MAX_ITERS
#define UF_COMPRESS_MAX_ITERS 256UL
#endif

/* You can override this at compile time:
 * make NVCCFLAGS+='-DBPR_GRID_MULT=4'
 * grid.x = min(maxGridX, ceil(nrows/rowsPerBlock)), then clamped by BPR_GRID_MULT*SMcount.
 */
#ifndef BPR_GRID_MULT
#define BPR_GRID_MULT 8
#endif

/* ---------- device UF helpers ---------- */

static __device__ __forceinline__ uint32_t
uf_find_halving(uint32_t *parent, uint32_t x)
{
	while (1) {
		uint32_t p = parent[x];
		if (p == x)
			return x;
		uint32_t gp = parent[p];
		parent[x] = gp;
		x = gp;
	}
}

static __device__ __forceinline__ void
uf_union_by_index_cas(uint32_t *parent, uint32_t a, uint32_t b)
{
	uint32_t ra = uf_find_halving(parent, a);
	uint32_t rb = uf_find_halving(parent, b);

	while (ra != rb) {
		uint32_t hi = (ra > rb) ? ra : rb;
		uint32_t lo = (ra > rb) ? rb : ra;

		uint32_t old = (uint32_t)atomicCAS((unsigned int *)&parent[hi],
		                                   (unsigned int)hi,
		                                   (unsigned int)lo);
		if (old == hi)
			return;

		ra = uf_find_halving(parent, ra);
		rb = uf_find_halving(parent, rb);
	}
}

/* ---------- kernels ---------- */

static __global__ void
uf_init(uint32_t *parent, uint32_t n)
{
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
		parent[tid] = tid;
}

/*
 * Block-per-row (scalable):
 * Each block processes many rows (u) in grid-stride.
 * For each u, the block cooperatively traverses colptr[u]..colptr[u+1].
 */
static __global__ void
uf_union_block_per_row_stride(const uint32_t *colptr, const uint32_t *rowi,
                              uint32_t *parent, uint32_t nrows)
{
	for (uint32_t u = (uint32_t)blockIdx.x; u < nrows; u += (uint32_t)gridDim.x) {
		uint32_t start = colptr[u];
		uint32_t end   = colptr[u + 1];

		for (uint32_t i = start + (uint32_t)threadIdx.x; i < end; i += (uint32_t)blockDim.x) {
			uint32_t v = rowi[i];
			uf_union_by_index_cas(parent, u, v);
		}
	}
}

static __global__ void
uf_compress_all_changed(uint32_t *parent, int *changed, uint32_t nrows)
{
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nrows)
		return;

	uint32_t oldp = parent[tid];
	uint32_t root = uf_find_halving(parent, tid);

	if (oldp != root) {
		parent[tid] = root;
		atomicExch(changed, 1);
	}
}

/* Count only true roots: parent[i] == i */
static __global__ void
cc_count_roots_only(const uint32_t *parent, unsigned long long *bitmap, uint32_t nrows)
{
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nrows)
		return;

	if (parent[tid] != tid)
		return;

	unsigned int word = tid >> 6;
	unsigned int bit  = tid & 63u;
	atomicOr(&bitmap[word], 1ULL << bit);
}

static __global__ void
cc_popcount(const unsigned long long *bitmap, uint32_t *count, uint32_t bitmap_size)
{
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < bitmap_size)
		atomicAdd(count, (uint32_t)__popcll(bitmap[tid]));
}

/* ---------- host entry ---------- */

int
connected_components_cuda_block_per_row(const Matrix *mtx)
{
	if (!mtx || mtx->nrows != mtx->ncols) {
		DERRF("expected square adjacency matrix");
		return -1;
	}
	if (!mtx->rowi || !mtx->colptr) {
		DERRF("matrix storage is NULL");
		return -1;
	}

	if (mtx->nrows > (size_t)UINT32_MAX || mtx->ncols > (size_t)UINT32_MAX || mtx->nnz > (size_t)UINT32_MAX) {
		DERRF("matrix too large for 32-bit indices (n=%zu, nnz=%zu)", mtx->nrows, mtx->nnz);
		return -1;
	}

	const uint32_t nrows = (uint32_t)mtx->nrows;
	const uint32_t nnz   = (uint32_t)mtx->nnz;

	if (mtx->colptr[mtx->ncols] != nnz) {
		DERRF("invalid CSC: colptr[ncols]=%u != nnz=%u", mtx->colptr[mtx->ncols], nnz);
		return -1;
	}

	uint32_t *d_colptr = NULL;
	uint32_t *d_rowi   = NULL;
	uint32_t *d_parent = NULL;
	int      *d_changed = NULL;

	CUDA_CHECK(cudaMalloc((void **)&d_colptr, (size_t)(nrows + 1) * sizeof(uint32_t)));
	CUDA_CHECK(cudaMalloc((void **)&d_rowi,   (size_t)nnz * sizeof(uint32_t)));
	CUDA_CHECK(cudaMalloc((void **)&d_parent, (size_t)nrows * sizeof(uint32_t)));
	CUDA_CHECK(cudaMalloc((void **)&d_changed, sizeof(int)));

	CUDA_CHECK(cudaMemcpy(d_colptr, mtx->colptr, (size_t)(nrows + 1) * sizeof(uint32_t),
	                      cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_rowi, mtx->rowi, (size_t)nnz * sizeof(uint32_t),
	                      cudaMemcpyHostToDevice));

	/* threads per block for block-cooperative neighbor traversal */
	const int threads = 256;

	/* init parent */
	const int blocks_v = (int)((nrows + (uint32_t)threads - 1) / (uint32_t)threads);
	uf_init<<<blocks_v, threads>>>(d_parent, nrows);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	/* choose a sane grid.x:
	 * - not proportional to nrows (too huge),
	 * - but enough to fill the GPU.
	 */
	cudaDeviceProp prop;
	int dev = 0;
	CUDA_CHECK(cudaGetDevice(&dev));
	CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

	int grid_x = prop.multiProcessorCount * BPR_GRID_MULT;
	/* Avoid grid_x > nrows (waste) */
	if ((uint32_t)grid_x > nrows)
		grid_x = (int)nrows;
	if (grid_x < 1)
		grid_x = 1;

	/* union pass (single pass over all edges) */
	uf_union_block_per_row_stride<<<grid_x, threads>>>(d_colptr, d_rowi, d_parent, nrows);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	/* compress until stable */
	unsigned long comp_iters = 0;
	int h_changed = 1;

	while (h_changed) {
		if (comp_iters++ >= UF_COMPRESS_MAX_ITERS) {
			DERRF("compression did not converge within UF_COMPRESS_MAX_ITERS=%lu", (unsigned long)UF_COMPRESS_MAX_ITERS);
			cudaFree(d_colptr); cudaFree(d_rowi); cudaFree(d_parent); cudaFree(d_changed);
			return -1;
		}

		h_changed = 0;
		CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));

		uf_compress_all_changed<<<blocks_v, threads>>>(d_parent, d_changed, nrows);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
	}

	/* count roots */
	const uint32_t bitmap_size = (nrows + 63u) / 64u;
	unsigned long long *d_bitmap = NULL;
	uint32_t *d_count = NULL;

	CUDA_CHECK(cudaMalloc((void **)&d_bitmap, (size_t)bitmap_size * sizeof(*d_bitmap)));
	CUDA_CHECK(cudaMalloc((void **)&d_count, sizeof(*d_count)));
	CUDA_CHECK(cudaMemset(d_bitmap, 0, (size_t)bitmap_size * sizeof(*d_bitmap)));
	CUDA_CHECK(cudaMemset(d_count,  0, sizeof(*d_count)));

	cc_count_roots_only<<<blocks_v, threads>>>(d_parent, d_bitmap, nrows);
	CUDA_CHECK(cudaGetLastError());

	const int blocks_bm = (int)((bitmap_size + (uint32_t)threads - 1) / (uint32_t)threads);
	cc_popcount<<<blocks_bm, threads>>>(d_bitmap, d_count, bitmap_size);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	uint32_t h_count = 0;
	CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

	cudaFree(d_colptr);
	cudaFree(d_rowi);
	cudaFree(d_parent);
	cudaFree(d_changed);
	cudaFree(d_bitmap);
	cudaFree(d_count);

	return (int)h_count;
}
