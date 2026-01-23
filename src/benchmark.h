/**
 * @file benchmark.h
 * @brief Benchmarking framework for parallel algorithms with CUDA support.
 *
 * Provides structures and functions to benchmark connected components
 * algorithms with comprehensive timing statistics, system information
 * capture, and JSON output formatting.
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

#ifndef PATH_MAX
#	include <linux/limits.h>
#endif
#ifndef PATH_MAX
#	define PATH_MAX 4096
#endif

#include "matrix.h"

/* ------------------------------------------------------------------------- */
/*                       Implementation Type Enumeration                     */
/* ------------------------------------------------------------------------- */

/**
 * @enum Implementation types
 * @brief Defines all the possible implementation types.
 *
 * @note IMPL_ALL provides the count of all implementations and is used
 *       to indicate "benchmark all implementations".
 */
enum {
	IMPL_SEQUENTIAL = 0,          /**< CPU: Sequential union-find */
	IMPL_CUDA_THREAD_PER_VERTEX,  /**< CUDA: One thread per vertex */
	IMPL_CUDA_WARP_PER_ROW,       /**< CUDA: One warp per row */
	IMPL_CUDA_BLOCK_PER_ROW,      /**< CUDA: One block per row */
	IMPL_CUDA_AFFOREST,           /**< CUDA: Afforest algorithm */
	IMPL_ALL                      /**< Sentinel: Total count of implementations */
};

/* ------------------------------------------------------------------------- */
/*                          Benchmark Data Structures                        */
/* ------------------------------------------------------------------------- */

/**
 * @struct Statistics
 * @brief Statistical summary of benchmark timing results.
 *
 * Provides comprehensive timing statistics computed across multiple
 * benchmark trials, including mean, standard deviation, median, min/max,
 * and total execution time.
 */
typedef struct {
	double mean_time_s;   /**< Mean execution time in seconds */
	double std_dev_s;     /**< Standard deviation of execution time in seconds */
	double median_time_s; /**< Median execution time in seconds */
	double min_time_s;    /**< Minimum execution time in seconds */
	double max_time_s;    /**< Maximum execution time in seconds */
	double total_time_s;  /**< Total execution time in seconds */
} Statistics;

/**
 * @struct Result
 * @brief Complete benchmark result for a single algorithm implementation.
 *
 * Contains all measured and computed metrics for one algorithm implementation,
 * including timing statistics, memory usage, and result validation data.
 */
typedef struct {
	char name[32];                     /**< Name of the implementation benchmarked */
	char timestamp[32];                /**< ISO 8601 timestamp of benchmark execution */
	unsigned int connected_components; /**< Number of connected components found */
	unsigned int stable_results;       /**< 1 if components match between all retries, else 0 */
	Statistics stats;                  /**< Timing statistics */
	double throughput_edges_per_sec;   /**< Processing throughput in edges per second */
	double *times;                     /**< Array of trial execution times in seconds */
	double cpu_peak_rss_gb;            /**< Peak CPU resident set size in GB during this implementation */
	double gpu_peak_used_gb;           /**< Peak GPU memory used in GB during this implementation */
} Result;

/**
 * @struct SystemInfo
 * @brief System information captured during benchmark execution.
 *
 * Contains details about the hardware and system configuration where
 * the benchmark was executed.
 */
typedef struct {
	char cpu_info[128]; /**< CPU model and specifications */
	double ram_gb;      /**< Total system RAM in gigabytes */
	double swap_gb;     /**< Total swap space in gigabytes */
} SystemInfo;

/**
 * @struct CudaDeviceInfo
 * @brief CUDA device information captured during benchmark execution.
 *
 * Contains information about the CUDA device where GPU implementations
 * were executed, including compute capability and memory size.
 */
typedef struct {
	int available;   /**< 1 if CUDA device is available, else 0 */
	char name[256];   /**< Device name (e.g., "NVIDIA GeForce RTX 3080") */
	int cc_major;    /**< Compute capability major version */
	int cc_minor;    /**< Compute capability minor version */
	double vram_gb;  /**< Total VRAM in gigabytes */
} CudaDeviceInfo;

/**
 * @struct MatrixInfo
 * @brief Information about the input matrix/graph.
 *
 * Describes the sparse matrix used as input for the connected
 * components algorithm, including its dimensions, sparsity, and
 * loading time.
 */
typedef struct {
	char path[PATH_MAX]; /**< File path to the matrix */
	unsigned int rows;   /**< Number of rows in the matrix */
	unsigned int cols;   /**< Number of columns in the matrix */
	unsigned int nnz;    /**< Number of non-zero elements (edges) */
	double load_time_s;  /**< Time it took to load the matrix into memory in seconds */
} MatrixInfo;

/**
 * @struct BenchmarkInfo
 * @brief Benchmark execution parameters.
 *
 * Contains the configuration parameters used for running the benchmark,
 * including trial counts and implementation selection.
 */
typedef struct {
	unsigned int trials;  /**< Number of benchmark trials performed */
	unsigned int wtrials; /**< Number of warmup trials before measuring performance */
	unsigned int imptype; /**< Implementation type index (or IMPL_ALL) */
} BenchmarkInfo;

/**
 * @struct Benchmark
 * @brief Master benchmark structure holding all results and metadata.
 *
 * Aggregates all benchmark data including system information, GPU info,
 * matrix properties, execution parameters, and per-implementation results.
 */
typedef struct {
	SystemInfo sys_info;          /**< System information */
	CudaDeviceInfo gpu_info;      /**< CUDA device information */
	MatrixInfo matrix_info;       /**< Matrix/graph information */
	BenchmarkInfo benchmark_info; /**< Benchmark parameters */
	Result results[IMPL_ALL];     /**< Array of algorithm results */
} Benchmark;

/* ------------------------------------------------------------------------- */
/*                          Public API Functions                             */
/* ------------------------------------------------------------------------- */

/**
 * @brief Returns current monotonic time in seconds.
 *
 * Uses CLOCK_MONOTONIC for reliable timing measurements that are not
 * affected by system clock adjustments.
 *
 * @return Current time in seconds (floating point).
 */
double now_sec(void);

/**
 * @brief Initializes a benchmark structure.
 *
 * Allocates and populates a new Benchmark instance for the specified
 * algorithm and dataset. Initializes all result structures and allocates
 * memory for timing results.
 *
 * @param[in] path    Path to the dataset file.
 * @param[in] trials  Number of trials to run.
 * @param[in] wtrials Number of warmup trials before measuring performance.
 * @param[in] imptype Implementation type to benchmark (or IMPL_ALL).
 * @param[in] mtx     Pointer to the Matrix used as input.
 *
 * @return Pointer to a newly allocated Benchmark structure, or NULL on failure.
 */
Benchmark* benchmark_init(const char *path, const unsigned int trials, 
                         const unsigned int wtrials, const unsigned int imptype, 
                         const Matrix *mtx);

/**
 * @brief Frees a Benchmark structure and all allocated resources.
 *
 * Releases all memory associated with the benchmark including timing
 * arrays for all implementations. Safe to call with NULL.
 *
 * @param[in,out] bench Pointer to the Benchmark structure to free.
 */
void benchmark_free(Benchmark *bench);

/**
 * @brief Runs the complete connected components benchmark.
 *
 * Executes the specified implementation(s) with the configured number
 * of warmup and measurement trials. Collects timing data, memory usage,
 * and validates result stability across trials.
 *
 * @param[in]     mtx   Input Matrix (adjacency matrix in CSC format).
 * @param[in,out] bench Benchmark object containing configuration and result storage.
 *
 * @return 0 on success, 1 on algorithm failure or invalid data.
 */
int benchmark_cc(const Matrix *mtx, Benchmark *bench);

/**
 * @brief Prints benchmark results in structured JSON format.
 *
 * Outputs benchmark metadata, timing statistics, system information,
 * GPU information, and matrix properties in valid JSON format to stdout.
 * Suitable for pipeline integration and automated result collection.
 *
 * @param[in] bench Pointer to the Benchmark structure with populated data.
 */
void benchmark_print(Benchmark *bench);

#endif /* BENCHMARK_H */
