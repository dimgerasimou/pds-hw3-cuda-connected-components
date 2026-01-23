/**
 * @file cc_cuda.cu
 * @brief CUDA implementations of connected components for large sparse graphs.
 *
 * Implements optimized atomic Union-Find (UF) for connected components
 * with three traversal strategies:
 *   - Thread-per-Vertex: One thread processes one vertex
 *   - Warp-per-Row: One warp (32 threads) collaboratively processes one vertex
 *   - Block-per-Row: One thread block processes one vertex using shared memory
 *
 * Additionally implements an Afforest-style preconditioning method:
 *   - A few sampling rounds (one neighbor per vertex per round)
 *   - Compression pass
 *   - Full UF pass (Block-per-Row strategy)
 *   - Final compression pass
 *
 * Key performance optimizations for MAWI-scale graphs:
 *   - Relaxed root finding (no atomics during traversal)
 *   - Grid-stride initialization and compression
 *   - GPU-side root counting (avoids copying parent[] back to host)
 *
 * Environment configuration knobs (optional):
 *   - CC_UF_THREADS  : Threads per block for UF kernels (default: 256)
 *   - CC_AFFO_ROUNDS : Afforest sampling rounds (default: 2)
 *   - CC_AFFO_SEED   : Afforest hash seed (default: 1)
 */

#include <cuda_runtime.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cc.h"
#include "error.h"
#include "matrix.h"

/* ------------------------------------------------------------------------- */
/*                            Global State Variables                         */
/* ------------------------------------------------------------------------- */

/**
 * @brief Tracks peak GPU memory usage for the most recent algorithm run.
 *
 * Updated by cc_cuda_run_uf() and queried by set_cuda_memory_metrics().
 */
static double g_last_peak_used_gb;

/* ------------------------------------------------------------------------- */
/*                          Environment Variable Helpers                     */
/* ------------------------------------------------------------------------- */

/**
 * @brief Reads an integer environment variable with bounds checking.
 *
 * @param[in] name    Environment variable name.
 * @param[in] defval  Default value if variable is unset or invalid.
 *
 * @return Parsed value clamped to [1, 1024], or defval if unset/invalid.
 */
static inline int
read_env_i(const char *name, int defval)
{
	const char *s = getenv(name);
	if (!s || !s[0])
		return defval;

	char *end = NULL;
	errno = 0;
	long v = strtol(s, &end, 10);

	if (errno || end == s)
		return defval;
	if (v < 1)
		return 1;
	if (v > 1024)
		return 1024;

	return (int)v;
}

/**
 * @brief Reads an unsigned 32-bit integer environment variable.
 *
 * @param[in] name    Environment variable name.
 * @param[in] defval  Default value if variable is unset or invalid.
 *
 * @return Parsed value, or defval if unset/invalid.
 */
static inline uint32_t
read_env_u32(const char *name, uint32_t defval)
{
	const char *s = getenv(name);
	if (!s || !s[0])
		return defval;

	char *end = NULL;
	errno = 0;
	unsigned long v = strtoul(s, &end, 10);

	if (errno || end == s)
		return defval;

	return (uint32_t)v;
}

/* ------------------------------------------------------------------------- */
/*                          CUDA Memory Tracking Helpers                     */
/* ------------------------------------------------------------------------- */

/**
 * @brief Updates minimum free memory tracker with current free memory.
 *
 * Queries CUDA for current free memory and updates the minimum if lower
 * than previously recorded value.
 *
 * @param[in,out] min_free Pointer to minimum free memory tracker (in bytes).
 */
static inline void
update_min_free_ull(unsigned long long *min_free)
{
	size_t free_b = 0, total_b = 0;

	if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess)
		return;

	unsigned long long f = (unsigned long long)free_b;
	if (*min_free == 0ULL || f < *min_free)
		*min_free = f;
}

/**
 * @brief Calculates number of CUDA blocks based on SM count and multiplier.
 *
 * Queries device properties to determine number of SMs (streaming
 * multiprocessors) and multiplies by the given factor, clamping to cap.
 *
 * @param[in] mul Multiplier for number of SMs.
 * @param[in] cap Maximum number of blocks (0 = no cap).
 *
 * @return Number of blocks to launch, or cap (or 65535) on error.
 */
static inline int
cuda_blocks_sm_mul(int mul, int cap)
{
	cudaDeviceProp prop;
	cudaError_t err;
	int blocks;

	err = cudaGetDeviceProperties(&prop, 0);
	if (err != cudaSuccess)
		return (cap > 0) ? cap : 65535;

	blocks = prop.multiProcessorCount * mul;
	if (blocks < 1)
		blocks = 1;
	if (cap > 0 && blocks > cap)
		blocks = cap;

	return blocks;
}

/**
 * @brief Macro for CUDA error checking with goto-based error handling.
 *
 * Checks CUDA call return value and jumps to label on error after
 * logging the error message.
 */
#define CUDA_CHECK_GOTO(call, label)                                    \
	do {                                                                \
		cudaError_t _e = (call);                                        \
		if (_e != cudaSuccess) {                                        \
			DERRF("%s failed: %s", #call, cudaGetErrorString(_e));      \
			goto label;                                                 \
		}                                                               \
	} while (0)

/* ------------------------------------------------------------------------- */
/*                      Device Union-Find Primitives                         */
/* ------------------------------------------------------------------------- */

/**
 * @brief Finds the root of a node with relaxed traversal and path halving.
 *
 * Uses relaxed (non-atomic) reads during traversal with opportunistic
 * path halving for compression. Correctness relies on atomicCAS only
 * for linking in uf_union().
 *
 * @param[in,out] parent Union-find parent array.
 * @param[in]     x      Node to find root for.
 *
 * @return Root of the set containing x.
 */
__device__ __forceinline__ uint32_t
uf_find_relaxed(uint32_t *parent, uint32_t x)
{
	uint32_t p = parent[x];
	while (p != x) {
		uint32_t gp = parent[p];
		if (gp != p)
			parent[x] = gp; /* Opportunistic path compression */
		x = p;
		p = parent[x];
	}
	return x;
}

/**
 * @brief Unites two nodes using atomic compare-and-swap with union-by-index.
 *
 * Implements union-by-index heuristic: always link higher index to lower
 * index to maintain canonical form. Uses atomicCAS for thread-safe linking.
 *
 * @param[in,out] parent Union-find parent array.
 * @param[in]     a      First node.
 * @param[in]     b      Second node.
 */
__device__ __forceinline__ void
uf_union(uint32_t *parent, uint32_t a, uint32_t b)
{
	while (true) {
		a = uf_find_relaxed(parent, a);
		b = uf_find_relaxed(parent, b);

		if (a == b)
			return;

		/* Union-by-index: link high -> low */
		if (a > b) {
			uint32_t t = a;
			a = b;
			b = t;
		}

		if (atomicCAS(&parent[b], b, a) == b)
			return;

		/* CAS failed -> retry */
	}
}

/* ------------------------------------------------------------------------- */
/*                         Common Utility Kernels                            */
/* ------------------------------------------------------------------------- */

/**
 * @brief Initializes parent array with identity mapping (each node is its own root).
 *
 * Uses grid-stride loop for efficient initialization across all nodes.
 *
 * @param[out] parent Parent array to initialize.
 * @param[in]  n      Number of nodes.
 */
__global__ void
k_init_parent(uint32_t *parent, uint32_t n)
{
	for (uint32_t i = (uint32_t)blockIdx.x * (uint32_t)blockDim.x + (uint32_t)threadIdx.x;
	     i < n;
	     i += (uint32_t)blockDim.x * (uint32_t)gridDim.x) {
		parent[i] = i;
	}
}

/**
 * @brief Compresses parent array by pointing each node directly to its root.
 *
 * Performs path compression to flatten the union-find tree structure
 * for faster subsequent queries. Uses grid-stride loop.
 *
 * @param[in,out] parent Parent array to compress.
 * @param[in]     n      Number of nodes.
 */
__global__ void
k_compress_parent(uint32_t *parent, uint32_t n)
{
	for (uint32_t i = (uint32_t)blockIdx.x * (uint32_t)blockDim.x + (uint32_t)threadIdx.x;
	     i < n;
	     i += (uint32_t)blockDim.x * (uint32_t)gridDim.x) {
		uint32_t r = i;
		while (parent[r] != r)
			r = parent[r];
		parent[i] = r;
	}
}

/**
 * @brief Counts the number of roots (connected components) in parent array.
 *
 * Uses block-level reduction followed by atomic accumulation to efficiently
 * count nodes where parent[i] == i (roots). Avoids copying parent array
 * back to host.
 *
 * @param[in]  parent Parent array (must be compressed).
 * @param[in]  n      Number of nodes.
 * @param[out] out    Output location for root count.
 */
__global__ void
k_count_roots(const uint32_t *parent, uint32_t n, unsigned long long *out)
{
	/* Block reduction + one atomicAdd per block */
	unsigned long long local = 0;

	for (uint32_t i = (uint32_t)blockIdx.x * (uint32_t)blockDim.x + (uint32_t)threadIdx.x;
	     i < n;
	     i += (uint32_t)blockDim.x * (uint32_t)gridDim.x) {
		local += (unsigned long long)(parent[i] == i);
	}

	/* Reduce within warp */
	for (int off = 16; off > 0; off >>= 1)
		local += __shfl_down_sync(0xffffffffu, local, off);

	/* Warp leaders write to shared, then first warp reduces */
	__shared__ unsigned long long warp_sums[32]; /* Max 1024 threads -> 32 warps */
	const int lane = (int)(threadIdx.x & 31);
	const int warp = (int)(threadIdx.x >> 5);

	if (lane == 0)
		warp_sums[warp] = local;
	__syncthreads();

	if (warp == 0) {
		unsigned long long sum = (lane < ((blockDim.x + 31) >> 5)) ? warp_sums[lane] : 0ULL;
		for (int off = 16; off > 0; off >>= 1)
			sum += __shfl_down_sync(0xffffffffu, sum, off);
		if (lane == 0)
			atomicAdd(out, sum);
	}
}

/* ------------------------------------------------------------------------- */
/*                        Union-Find Traversal Kernels                       */
/* ------------------------------------------------------------------------- */

/**
 * @brief Thread-per-vertex union-find kernel.
 *
 * Each thread processes one vertex, iterating over all its neighbors
 * and performing unions. Uses grid-stride loop to handle graphs larger
 * than the grid size.
 *
 * @param[in]     colptr Column pointer array (CSC format).
 * @param[in]     rowi   Row index array (CSC format).
 * @param[in,out] parent Union-find parent array.
 * @param[in]     ncols  Number of columns (vertices).
 */
__global__ void
k_uf_thread_per_vertex(const uint32_t *__restrict__ colptr,
                       const uint32_t *__restrict__ rowi,
                       uint32_t *__restrict__ parent,
                       uint32_t ncols)
{
	for (uint32_t u = (uint32_t)blockIdx.x * (uint32_t)blockDim.x + (uint32_t)threadIdx.x;
	     u < ncols;
	     u += (uint32_t)blockDim.x * (uint32_t)gridDim.x) {

		uint32_t s = colptr[u];
		uint32_t e = colptr[u + 1];

		for (uint32_t k = s; k < e; k++) {
			uint32_t v = rowi[k];
			uf_union(parent, u, v);
		}
	}
}

/**
 * @brief Warp-per-row union-find kernel.
 *
 * Each warp (32 threads) collaboratively processes one vertex. Threads
 * within a warp cooperate on the neighbors of the same vertex. Best for
 * medium-degree vertices (roughly 32-1024 neighbors).
 *
 * @param[in]     colptr Column pointer array (CSC format).
 * @param[in]     rowi   Row index array (CSC format).
 * @param[in,out] parent Union-find parent array.
 * @param[in]     ncols  Number of columns (vertices).
 */
__global__ void
k_uf_warp_per_row(const uint32_t *__restrict__ colptr,
                  const uint32_t *__restrict__ rowi,
                  uint32_t *__restrict__ parent,
                  uint32_t ncols)
{
	const uint32_t warps_per_grid = ((uint32_t)blockDim.x * (uint32_t)gridDim.x) >> 5;
	const uint32_t global_warp = (((uint32_t)blockIdx.x * (uint32_t)blockDim.x) + (uint32_t)threadIdx.x) >> 5;
	const uint32_t lane = (uint32_t)threadIdx.x & 31u;

	for (uint32_t u = global_warp; u < ncols; u += warps_per_grid) {
		uint32_t s = colptr[u];
		uint32_t e = colptr[u + 1];
		uint32_t deg = e - s;

		for (uint32_t i = lane; i < deg; i += 32u) {
			uint32_t v = rowi[s + i];
			uf_union(parent, u, v);
		}
	}
}

/**
 * @brief Block-per-row union-find kernel.
 *
 * Each thread block processes one vertex using all threads in the block.
 * Uses grid-stride loop over vertices. Best for high-degree vertices
 * (1024+ neighbors).
 *
 * @param[in]     colptr Column pointer array (CSC format).
 * @param[in]     rowi   Row index array (CSC format).
 * @param[in,out] parent Union-find parent array.
 * @param[in]     ncols  Number of columns (vertices).
 */
__global__ void
k_uf_block_per_row(const uint32_t *__restrict__ colptr,
                   const uint32_t *__restrict__ rowi,
                   uint32_t *__restrict__ parent,
                   uint32_t ncols)
{
	/* One block handles a row, blocks grid-stride over rows */
	for (uint32_t u = (uint32_t)blockIdx.x; u < ncols; u += (uint32_t)gridDim.x) {
		uint32_t s = colptr[u];
		uint32_t e = colptr[u + 1];
		uint32_t deg = e - s;

		for (uint32_t i = (uint32_t)threadIdx.x; i < deg; i += (uint32_t)blockDim.x) {
			uint32_t v = rowi[s + i];
			uf_union(parent, u, v);
		}
	}
}

/* ------------------------------------------------------------------------- */
/*                          Afforest Preconditioning                         */
/* ------------------------------------------------------------------------- */

/**
 * @brief Simple 32-bit integer hash function.
 *
 * Provides deterministic pseudo-random values for neighbor sampling
 * in Afforest algorithm.
 *
 * @param[in] x Input value to hash.
 *
 * @return Hashed value.
 */
__device__ __forceinline__ uint32_t
hash32(uint32_t x)
{
	/* Simple integer hash (deterministic, fast) */
	x ^= x >> 16;
	x *= 0x7feb352dU;
	x ^= x >> 15;
	x *= 0x846ca68bU;
	x ^= x >> 16;
	return x;
}

/**
 * @brief Afforest sampling kernel using warp-per-row strategy.
 *
 * Each warp processes one vertex and performs at most one union with
 * a pseudo-randomly selected neighbor for this sampling round. This
 * preconditioning step helps reduce work for the main UF pass.
 *
 * @param[in]     colptr Column pointer array (CSC format).
 * @param[in]     rowi   Row index array (CSC format).
 * @param[in,out] parent Union-find parent array.
 * @param[in]     ncols  Number of columns (vertices).
 * @param[in]     round  Current sampling round number.
 * @param[in]     seed   Hash seed for pseudo-random selection.
 */
__global__ void
k_afforest_sample_warp(const uint32_t *__restrict__ colptr,
                       const uint32_t *__restrict__ rowi,
                       uint32_t *__restrict__ parent,
                       uint32_t ncols,
                       uint32_t round,
                       uint32_t seed)
{
	/* One warp per row; each row performs (at most) one union in this round */
	const uint32_t warps_per_grid = ((uint32_t)blockDim.x * (uint32_t)gridDim.x) >> 5;
	const uint32_t global_warp = (((uint32_t)blockIdx.x * (uint32_t)blockDim.x) + (uint32_t)threadIdx.x) >> 5;
	const uint32_t lane = (uint32_t)threadIdx.x & 31u;

	for (uint32_t u = global_warp; u < ncols; u += warps_per_grid) {
		uint32_t s = colptr[u];
		uint32_t e = colptr[u + 1];
		uint32_t deg = e - s;

		if (deg == 0)
			continue;

		/* Pick a pseudo-random neighbor index for this round */
		uint32_t h = hash32(u ^ (seed + round * 0x9e3779b9U));
		uint32_t off = (deg == 1) ? 0 : (h % deg);

		if (lane == 0) {
			uint32_t v = rowi[s + off];
			uf_union(parent, u, v);
		}
	}
}

/* ------------------------------------------------------------------------- */
/*                          Host-side Common Runner                          */
/* ------------------------------------------------------------------------- */

/**
 * @brief Common CUDA union-find runner for all variants.
 *
 * Handles memory allocation, kernel launches, and cleanup for all
 * CUDA-based connected components implementations. Supports optional
 * Afforest preconditioning.
 *
 * @param[in] mtx         Input matrix in CSC format.
 * @param[in] variant     Kernel variant (0=TPV, 1=WPR, 2=BPR).
 * @param[in] do_afforest Enable Afforest preconditioning if non-zero.
 *
 * @return Number of connected components, or -1 on error.
 */
static int
cc_cuda_run_uf(const Matrix *mtx, int variant, int do_afforest)
{
	/* Track peak GPU memory used during this run using cudaMemGetInfo() */
	unsigned long long total_mem = 0ULL;
	unsigned long long min_free_mem = 0ULL;

	{
		size_t free_b = 0, total_b = 0;
		if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
			total_mem = (unsigned long long)total_b;
			min_free_mem = (unsigned long long)free_b;
		}
	}

	/* Declare all variables at the beginning to allow goto cleanup */
	size_t n_sz, nnz_sz;
	uint32_t n;
	size_t colptr_bytes, rowi_bytes, parent_bytes;

	uint32_t *d_colptr, *d_rowi, *d_parent;
	unsigned long long *d_count;
	unsigned long long h_count;

	cudaError_t err;
	int threads;
	int blocks_init, blocks_main, blocks_count;
	int rc;

	int affo_rounds;
	uint32_t affo_seed;

	/* Initialize variables */
	n_sz = 0;
	nnz_sz = 0;
	n = 0;
	colptr_bytes = rowi_bytes = parent_bytes = 0;

	d_colptr = NULL;
	d_rowi = NULL;
	d_parent = NULL;
	d_count = NULL;
	h_count = 0ULL;

	err = cudaSuccess;
	threads = 256;
	blocks_init = 0;
	blocks_main = 0;
	blocks_count = 0;
	rc = -1;

	affo_rounds = 2;
	affo_seed = 1;

	/* Validate input */
	if (!mtx || mtx->nrows != mtx->ncols) {
		DERRF("expected square adjacency matrix");
		return -1;
	}
	if (mtx->nrows > UINT32_MAX || mtx->nnz > UINT32_MAX) {
		DERRF("matrix too large for uint32_t indexing");
		return -1;
	}

	/* Read configuration from environment */
	threads = read_env_i("CC_UF_THREADS", 256);
	if (threads % 32 != 0)
		threads = (threads + 31) & ~31; /* Keep warp aligned */
	if (threads < 64) threads = 64;
	if (threads > 1024) threads = 1024;

	n_sz = mtx->nrows;
	nnz_sz = mtx->nnz;
	n = (uint32_t)n_sz;

	colptr_bytes = (n_sz + 1) * sizeof(uint32_t);
	rowi_bytes = nnz_sz * sizeof(uint32_t);
	parent_bytes = n_sz * sizeof(uint32_t);

	blocks_init = cuda_blocks_sm_mul(32, 65535);
	blocks_count = cuda_blocks_sm_mul(32, 65535);

	/* Main kernel blocks: for WPR/TPV we want lots of CTAs; for BPR we want up to 65535 */
	if (variant == 2) {
		blocks_main = (n < 65535u) ? (int)n : 65535;
	} else {
		/* Enough blocks to cover the grid-stride loops well */
		blocks_main = 65535;
	}

	if (do_afforest) {
		affo_rounds = read_env_i("CC_AFFO_ROUNDS", 2);
		if (affo_rounds > 8) affo_rounds = 8; /* Keep it light; UF does the real work */
		affo_seed = read_env_u32("CC_AFFO_SEED", 1);
	}

	/* Allocate device memory */
	CUDA_CHECK_GOTO(cudaMalloc((void **)&d_colptr, colptr_bytes), cleanup);
	CUDA_CHECK_GOTO(cudaMalloc((void **)&d_rowi, rowi_bytes), cleanup);
	CUDA_CHECK_GOTO(cudaMalloc((void **)&d_parent, parent_bytes), cleanup);
	CUDA_CHECK_GOTO(cudaMalloc((void **)&d_count, sizeof(unsigned long long)), cleanup);

	/* Copy input data to device */
	CUDA_CHECK_GOTO(cudaMemcpy(d_colptr, mtx->colptr, colptr_bytes, cudaMemcpyHostToDevice), cleanup);
	CUDA_CHECK_GOTO(cudaMemcpy(d_rowi, mtx->rowi, rowi_bytes, cudaMemcpyHostToDevice), cleanup);

	update_min_free_ull(&min_free_mem);

	/* Initialize parent array */
	k_init_parent<<<blocks_init, threads>>>(d_parent, n);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		DERRF("kernel launch failed: %s", cudaGetErrorString(err));
		goto cleanup;
	}

	/* Optional Afforest sampling rounds (preconditioning) */
	if (do_afforest && affo_rounds > 0) {
		for (int r = 0; r < affo_rounds; r++) {
			k_afforest_sample_warp<<<65535, 256>>>(d_colptr, d_rowi, d_parent, n, (uint32_t)r, affo_seed);
			err = cudaGetLastError();
			if (err != cudaSuccess) {
				DERRF("kernel launch failed: %s", cudaGetErrorString(err));
				goto cleanup;
			}
		}

		/* Compress once after sampling rounds */
		k_compress_parent<<<blocks_init, threads>>>(d_parent, n);
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			DERRF("kernel launch failed: %s", cudaGetErrorString(err));
			goto cleanup;
		}
	}

	/* Main union-find pass */
	if (variant == 0) {
		k_uf_thread_per_vertex<<<blocks_main, threads>>>(d_colptr, d_rowi, d_parent, n);
	} else if (variant == 1) {
		/* 8 warps per block for 256 threads; keep it stable */
		k_uf_warp_per_row<<<blocks_main, 256>>>(d_colptr, d_rowi, d_parent, n);
	} else {
		k_uf_block_per_row<<<blocks_main, threads>>>(d_colptr, d_rowi, d_parent, n);
	}

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		DERRF("kernel launch failed: %s", cudaGetErrorString(err));
		goto cleanup;
	}

	CUDA_CHECK_GOTO(cudaDeviceSynchronize(), cleanup);

	update_min_free_ull(&min_free_mem);

	/* Final compression (do twice; usually helps) */
	k_compress_parent<<<blocks_init, threads>>>(d_parent, n);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		DERRF("kernel launch failed: %s", cudaGetErrorString(err));
		goto cleanup;
	}

	k_compress_parent<<<blocks_init, threads>>>(d_parent, n);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		DERRF("kernel launch failed: %s", cudaGetErrorString(err));
		goto cleanup;
	}

	/* Count roots on GPU */
	CUDA_CHECK_GOTO(cudaMemset(d_count, 0, sizeof(unsigned long long)), cleanup);

	k_count_roots<<<blocks_count, threads>>>(d_parent, n, d_count);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		DERRF("kernel launch failed: %s", cudaGetErrorString(err));
		goto cleanup;
	}

	CUDA_CHECK_GOTO(cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost), cleanup);

	rc = (h_count > (unsigned long long)INT32_MAX) ? -1 : (int)h_count;

	/* Finalize per-run GPU peak used memory (GB) */
	update_min_free_ull(&min_free_mem);
	if (total_mem > 0ULL && min_free_mem > 0ULL && min_free_mem <= total_mem) {
		unsigned long long peak_used_bytes = total_mem - min_free_mem;
		g_last_peak_used_gb = (double)peak_used_bytes / 1024.0 / 1024.0 / 1024.0;
	} else {
		g_last_peak_used_gb = 0.0;
	}

cleanup:
	/* Best-effort cleanup */
	if (d_count)  cudaFree(d_count);
	if (d_parent) cudaFree(d_parent);
	if (d_rowi)   cudaFree(d_rowi);
	if (d_colptr) cudaFree(d_colptr);

	return rc;
}

/* ------------------------------------------------------------------------- */
/*                                Public API                                 */
/* ------------------------------------------------------------------------- */

/**
 * @brief Records peak CUDA memory usage in result structure.
 *
 * Copies the globally tracked peak memory usage from the most recent
 * algorithm run into the provided result structure.
 *
 * @param[out] result Result structure to update.
 *
 * @return 0 on success, -1 if result is NULL.
 */
int
set_cuda_memory_metrics(Result *result)
{
	if (!result)
		return -1;
	result->gpu_peak_used_gb = g_last_peak_used_gb;
	return 0;
}

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
int
get_cuda_device_info(CudaDeviceInfo *info)
{
	cudaDeviceProp prop;
	int devcount;

	if (!info) {
		DERRF("info struct is invalid");
		return -1;
	}

	memset(info, 0, sizeof(*info));
	info->available = 0;

	devcount = 0;
	CUDA_CHECK_GOTO(cudaGetDeviceCount(&devcount), infoerr);

	if (devcount <= 0) {
		DERRF("device count is invalid");
		return -1;
	}

	CUDA_CHECK_GOTO(cudaGetDeviceProperties(&prop, 0), infoerr);

	info->available = 1;
	snprintf(info->name, sizeof(info->name), "%s", prop.name);
	info->cc_major = prop.major;
	info->cc_minor = prop.minor;
	info->vram_gb = prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0;

	return 0;

infoerr:
	return -1;
}

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
int
connected_components_cuda_thread_per_vertex(const Matrix *mtx)
{
	return cc_cuda_run_uf(mtx, 0, 0);
}

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
int
connected_components_cuda_warp_per_row(const Matrix *mtx)
{
	return cc_cuda_run_uf(mtx, 1, 0);
}

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
int
connected_components_cuda_block_per_row(const Matrix *mtx)
{
	return cc_cuda_run_uf(mtx, 2, 0);
}

/**
 * @brief Connected components using the Afforest algorithm.
 *
 * Strategy: Hybrid approach combining sampling and neighbor processing.
 * Adapts to graph structure for improved performance.
 *
 * @param[in] mtx Sparse binary matrix in CSC format (adjacency matrix).
 *
 * @return Number of connected components, or -1 on error.
 */int
connected_components_cuda_afforest(const Matrix *mtx)
{
	return cc_cuda_run_uf(mtx, 2, 1);
}
