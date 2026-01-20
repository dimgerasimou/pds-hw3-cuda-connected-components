/**
 * @file benchmark.h
 * @brief Benchmarking framework for parallel algorithms with CUDA support.
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <limits.h>

#include "matrix.h"

/**
 * @enum
 * @brief Defines all the possible implementation types.
 *
 * @note IMPL_ALL provides the count of all implementations.
 */
enum {
	IMPL_SERIAL = 0,
	IMPL_CUDA,
	IMPL_ALL
};

/**
 * @struct Statistics
 * @brief Statistical summary of benchmark timing results.
 *
 * Provides comprehensive timing statistics computed across multiple
 * benchmark trials.
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
 * @brief Complete benchmark result for a single algorithm.
 *
 * Contains all measured and computed metrics for one algorithm implementation.
 */
typedef struct {
	char name[32];      /**< Name of the implementation benchmarked */
	char timestamp[32];                /**< ISO 8601 timestamp of benchmark execution */
	unsigned int connected_components; /**< Number of connected components found */
	unsigned int stable_results;       /**< True if components match between all retries */
	Statistics stats;                  /**< Timing statistics */
	double throughput_edges_per_sec;   /**< Processing throughput in edges per second */
	double *times;                     /**< Array of trial execution times in seconds. */
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
	double ram_mb;      /**< Total RAM in megabytes */
	double swap_mb;     /**< Total swap space in megabytes */
} SystemInfo;

/**
 * @struct MatrixInfo
 * @brief Information about the input matrix/graph.
 *
 * Describes the sparse matrix used as input for the connected
 * components algorithm, including its dimensions and sparsity.
 */
typedef struct {
	char path[PATH_MAX]; /**< File path to the matrix */
	unsigned int rows;   /**< Number of rows in the matrix */
	unsigned int cols;   /**< Number of columns in the matrix */
	unsigned int nnz;    /**< Number of non-zero elements (edges in graph) */
	double load_time_s;  /**< Time it took to load the matrix in memory */
} MatrixInfo;

/**
 * @struct BenchmarkInfo
 * @brief Benchmark execution parameters.
 *
 * Contains the configuration parameters used for running the benchmark.
 */
typedef struct {
	unsigned int trials;  /**< Number of benchmark trials performed */
	unsigned int wtrials; /**< Number of warmup trials before mesuring performance */
	unsigned int imptype; /**< Implementation type index */
} BenchmarkInfo;

/**
 * @brief Holds benchmark results and metadata.
 */
typedef struct {
	SystemInfo sys_info;          /**< System information */
	MatrixInfo matrix_info;       /**< Matrix/graph information */
	BenchmarkInfo benchmark_info; /**< Benchmark parameters */
	Result results[IMPL_ALL];     /**< Algorithm results */
} Benchmark;

/**
 * @brief Returns current monotonic time in seconds.
 */
double nowsec(void);

/**
 * @brief Initializes a benchmark structure.
 *
 * Allocates and populates a new Benchmark instance for the specified
 * algorithm and dataset. Also allocates memory for timing results.
 *
 * @param[in] path     Path to the dataset file.
 * @param[in] trials   Number of trials to run.
 * @param[in] wtrials  Number of warmup trials before mesuring performance.
 * @param[in] imptype  Implementation type to benchmark.
 * @param[in] mat      Pointer to the CSCBinaryMatrix used as input.
 *
 * @return Pointer to a newly allocated Benchmark structure, or `NULL` on failure.
 */
Benchmark* benchmarkinit(const char *path, const unsigned int trials, const unsigned int wtrials, const unsigned int imptype, const Matrix *mtx);

/**
 * @brief Frees a Benchmark structure and all allocated resources.
 *
 * @param[in,out] b Pointer to the Benchmark structure to free. Safe to call with NULL.
 */
void benchmarkfree(Benchmark *bench);

/**
 * @brief Runs the complete connected components benchmark.
 *
 * @param[in]     m Input Matrix.
 * @param[in,out] b Benchmark object containing configuration and result storage.
 *
 * @return
 * - `0` on success,
 * - `1` on algorithm failure or invalid data,
 */
int benchmarkcc(const Matrix *mtx, Benchmark *bench);

/**
 * @brief Prints benchmark results in structured JSON format.
 *
 * Outputs benchmark metadata, timing statistics, system information,
 * and matrix properties in JSON form for easy parsing or logging.
 *
 * @param[in] b Pointer to the Benchmark structure with populated data.
 */
void benchmarkprint(Benchmark *bench);

#endif /* BENCHMARK_H */
