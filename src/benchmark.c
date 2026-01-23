/**
 * @file benchmark.c
 * @brief Implementation of benchmarking framework for parallel algorithms.
 *
 * Supports both single-process (OpenMP) and multi-process (MPI+OpenMP) benchmarking.
 */

#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <unistd.h>

#include "benchmark.h"
#include "cc.h"
#include "error.h"
#include "json.h"

/* JSON output indentation level */
#define JSON_INDENT 2

const char *implementation_names[] = {
	"CPU: Sequential",
	"CUDA: Thread-per-Vertex",
	"CUDA: Warp-per-Row",
	"CUDA: Block-per-Row",
	"CUDA: Afforest"
};

/* ------------------------------------------------------------------------- */
/*                            Static Helper Functions                        */
/* ------------------------------------------------------------------------- */

/**
 * @brief Comparison function for sorting doubles.
 */
static int
cmp_double(const void *a, const void *b)
{
	double da = *(const double *)a;
	double db = *(const double *)b;
	return (da > db) - (da < db);
}

/**
 * @brief Calculates the statistics of a particular implementation.
 *
 * Populates fields in the Statistics structure, including min, max,
 * mean, median, and standard deviation.
 *
 * @param[in,out] b  Pointer to benchmark structure.
 * @param[in]     im Implementation for which to populate.
 *
 * @return 0 on success, 1 on error.
 */
static int
calcstatistics(Benchmark *b, const unsigned int im)
{
	size_t trials;
	double *times;
	double sum = 0.0, sum_sq = 0.0, time_avg = 0.0;

	if (!b) {
		DERRF("benchmark structure not initialized");
		return 1;
	}

	if (im >= IMPL_ALL) {
		DERRF("invalid implementation index: %u", im);
		return 1;
	}

	if (!b->results[im].times) {
		DERRF("times array for implementation %u not defined", im);
		return 1;
	}

	trials = b->benchmark_info.trials;
	
	/* Sort in-place since we don't need original order after benchmarking */
	times = b->results[im].times;
	qsort(times, trials, sizeof(double), cmp_double);

	b->results[im].stats.min_time_s = times[0];
	b->results[im].stats.max_time_s = times[trials - 1];
	b->results[im].stats.median_time_s = (trials % 2)
		? times[trials / 2]
		: (times[trials / 2] + times[trials / 2 - 1]) / 2.0;

	for (unsigned int i = 0; i < trials; i++) {
		sum += times[i];
		sum_sq += times[i] * times[i];
	}

	time_avg = sum / trials;
	b->results[im].stats.mean_time_s = time_avg;
	b->results[im].stats.std_dev_s = (trials > 1)
		? sqrt((sum_sq - trials * time_avg * time_avg) / (trials - 1))
		: 0.0;

	return 0;
}

/**
 * @brief Retrieves system memory information in MB.
 */
static void
getmeminfo(Benchmark *b)
{
#ifdef __linux__
	struct sysinfo info;
	if (sysinfo(&info) == 0) {
		b->sys_info.ram_gb  = info.totalram  / 1024.0 / 1024.0 / 1024.0 * info.mem_unit;
		b->sys_info.swap_gb = info.totalswap / 1024.0 / 1024.0 / 1024.0 * info.mem_unit;
	} else {
		b->sys_info.ram_gb = b->sys_info.swap_gb = 0.0;
	}
#else
	b->sys_info.ram_gb = b->sys_info.swap_gb = 0.0;
#endif
}

/**
 * @brief Retrieves CPU model information from /proc/cpuinfo.
 */
static void
getcpuinfo(Benchmark *b)
{
#ifdef __linux__
	FILE *f = fopen("/proc/cpuinfo", "r");
	if (!f) {
		snprintf(b->sys_info.cpu_info, sizeof(b->sys_info.cpu_info), "unknown");
		return;
	}

	char line[256];
	while (fgets(line, sizeof(line), f)) {
		if (strncmp(line, "model name", 10) == 0) {
			char *p = strchr(line, ':');
			if (p) {
				snprintf(b->sys_info.cpu_info, sizeof(b->sys_info.cpu_info), "%s", p + 2); // skip ": "
				b->sys_info.cpu_info[strcspn(b->sys_info.cpu_info, "\n")] = 0;             // remove newline
			}
			break;
		}
	}
	fclose(f);
#else
	snprintf(b->sys_info.cpu_info, sizeof(b->sys_info.cpu_info), "unknown");
#endif
}

/**
 * @brief Generates an ISO-8601 formatted timestamp for the selected implementation.
 */
static void
gettimestamp(Benchmark *b, const unsigned int im)
{
	time_t t = time(NULL);
	struct tm tm;
	localtime_r(&t, &tm);
	strftime(b->results[im].timestamp, sizeof(b->results[im].timestamp), "%Y-%m-%dT%H:%M:%S", &tm);
}

/**
 * @brief Read current resident set size (RSS) in kilobytes.
 *
 * Linux-specific: parses /proc/self/statm.
 */
#ifdef __linux__
static unsigned long long
read_rss_kb(void)
{
	FILE *f = fopen("/proc/self/statm", "r");
	if (!f)
		return 0ULL;

	unsigned long long size_pages = 0ULL;
	unsigned long long rss_pages = 0ULL;
	if (fscanf(f, "%llu %llu", &size_pages, &rss_pages) != 2) {
		fclose(f);
		return 0ULL;
	}
	fclose(f);

	long pagesize = sysconf(_SC_PAGESIZE);
	if (pagesize <= 0)
		pagesize = 4096;

	unsigned long long rss_bytes = rss_pages * (unsigned long long)pagesize;
	return rss_bytes / 1024ULL;
}
#else
static unsigned long long
read_rss_kb(void)
{
	return 0ULL;
}
#endif

/**
 * @brief Runs a connected components benchmark on the selected implementation.
 *
 * @param[in]     m   Input Matrix (local partition).
 * @param[in,out] b   Benchmark object containing configuration and result storage.
 * @param[in]     im  Implementation type.
 *
 * @return `0` on success or`1` on algorithm failure or invalid data,
 */
static int
benchmark_impl(const Matrix *m, Benchmark *b, unsigned int im)
{
	double time_tot_start, time_tot_end;
	unsigned long long rss_peak_kb;

	/* allocate times array */
	b->results[im].times = malloc(b->benchmark_info.trials * sizeof(double));
	if (!b->results[im].times) {
		DERRNOF("malloc() failed");
		return 1;
	}
	
	/* set result timestamp */
	gettimestamp(b, im);

	/* set implementation name */
	strncpy(b->results[im].name, implementation_names[im], sizeof(b->results[im].name));
	b->results[im].name[sizeof(b->results[im].name) - 1] = '\0';

	/* initialize per-implementation memory peaks */
	rss_peak_kb = read_rss_kb();

	/* warmup runs */
	for (unsigned int i = 0; i < b->benchmark_info.wtrials; i++) {
		int result;

		result = connected_components(m, im);
		if (result < 0) {
			uerrf("implementation \"%s\" encountered an error", b->results[im].name);
			return 1;
		}

		/* Track per-implementation CPU RSS peak (warmups included). */
		{
			unsigned long long rss_now = read_rss_kb();
			if (rss_now > rss_peak_kb)
				rss_peak_kb = rss_now;
		}

		/* Track per-implementation GPU peak memory (warmups included). */
		if (im != IMPL_SEQUENTIAL)
			if (set_cuda_memory_metrics(&b->results[im]))
				DERRF("failed to set CUDA memory metrics");
	}

	/* normal runs */
	time_tot_start = now_sec();
	for (unsigned int i = 0; i < b->benchmark_info.trials; i++) {
		double time_start, time_end;
		int result;

		time_start = now_sec();
		result = connected_components(m, im);
		time_end = now_sec();

		if (result < 0) {
			uerrf("implementation \"%s\" encountered an error", b->results[im].name);
			return 1;
		}

		b->results[im].times[i] = time_end - time_start;

		if (i == 0)
			b->results[im].connected_components = result;

		if ((unsigned int) result != b->results[im].connected_components)
			b->results[im].stable_results = 0;

		/* Track per-implementation CPU RSS peak (warmups included). */
		{
			unsigned long long rss_now = read_rss_kb();
			if (rss_now > rss_peak_kb)
				rss_peak_kb = rss_now;
		}

		/* Track per-implementation GPU peak memory (warmups included). */
		if (im != IMPL_SEQUENTIAL)
			if (set_cuda_memory_metrics(&b->results[im]))
				DERRF("failed to set CUDA memory metrics");
	}
	time_tot_end = now_sec();

	b->results[im].cpu_peak_rss_gb = (double)rss_peak_kb / 1024.0 / 1024.0;
	b->results[im].stats.total_time_s = time_tot_end - time_tot_start;
	return 0;
}

/* ------------------------------------------------------------------------- */
/*                            Public API Implementation                      */
/* ------------------------------------------------------------------------- */

/**
 * @brief Returns current monotonic time in seconds.
 *
 * Uses CLOCK_MONOTONIC for reliable timing measurements that are not
 * affected by system clock adjustments.
 *
 * @return Current time in seconds (floating point).
 */
double
now_sec(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + t.tv_nsec / 1e9;
}

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
Benchmark*
benchmark_init(const char *path, const unsigned int trials, const unsigned int wtrials, const unsigned int imptype, const Matrix *m)
{
	if (!trials) {
		DERRF("invalid number of trials");
		return NULL;
	}

	Benchmark *b = malloc(sizeof(Benchmark));
	if (!b) {
		DERRNOF("malloc() failed");
		return NULL;
	}

	// Add matrix info
	b->matrix_info.cols = m->ncols;
	b->matrix_info.rows = m->nrows;
	b->matrix_info.nnz = m->nnz;
	strncpy(b->matrix_info.path, path, sizeof(b->matrix_info.path));
	b->matrix_info.path[sizeof(b->matrix_info.path) - 1] = '\0';

	b->benchmark_info.trials  = trials;
	b->benchmark_info.wtrials = wtrials;
	b->benchmark_info.imptype = imptype;
	b->gpu_info.available = 0;

	for (unsigned int i = 0; i < IMPL_ALL; i++) {
		b->results[i].times = NULL;
		b->results[i].name[0] = '\0';
		b->results[i].timestamp[0] = '\0';
		b->results[i].stable_results = 1;
	}

	return b;
}

/**
 * @brief Frees a Benchmark structure and all allocated resources.
 *
 * Releases all memory associated with the benchmark including timing
 * arrays for all implementations. Safe to call with NULL.
 *
 * @param[in,out] bench Pointer to the Benchmark structure to free.
 */
void
benchmark_free(Benchmark *b)
{
	if (!b)
		return;

	for (unsigned int i = 0; i < IMPL_ALL; i++) {
		if (b->results[i].times)
			free(b->results[i].times);
	}

	free(b);
}

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
int
benchmark_cc(const Matrix *m, Benchmark *b)
{
	if (b->benchmark_info.imptype == IMPL_ALL) {
		for (unsigned int i = 0; i < IMPL_ALL; i++) {
			if (benchmark_impl(m, b, i))
				return 1;
		}
		return 0;
	}
	
	return benchmark_impl(m, b, b->benchmark_info.imptype);
}

/**
 * @brief Prints benchmark results in structured JSON format.
 *
 * Outputs benchmark metadata, timing statistics, system information,
 * GPU information, and matrix properties in valid JSON format to stdout.
 * Suitable for pipeline integration and automated result collection.
 *
 * @param[in] bench Pointer to the Benchmark structure with populated data.
 */
void
benchmark_print(Benchmark *b)
{
	if (!b)
		return;

	if (b->benchmark_info.imptype == IMPL_ALL) {
		for (unsigned int i = 0; i < IMPL_ALL; i++) {
			if (calcstatistics(b, i))
				return;
				
			/* NOTE: graphs are treated as undirected; the .mtx contains one triangle,
			 * so nnz counts each undirected edge once. For throughput we report
			 * *undirected edges processed per second* (i.e., 2*nnz / time).
			 */
			double edges = 2.0 * (double)b->matrix_info.nnz;
			b->results[i].throughput_edges_per_sec =
			    edges / b->results[i].stats.mean_time_s;
		}
	} else {
		unsigned int im = b->benchmark_info.imptype;
		if (calcstatistics(b, im))
			return;

		/* NOTE: graphs are treated as undirected; the .mtx contains one triangle,
		 * so nnz counts each undirected edge once. For throughput we report
		 * *undirected edges processed per second* (i.e., 2*nnz / time).
		 */
		double edges = 2.0 * (double)b->matrix_info.nnz;
		b->results[im].throughput_edges_per_sec =
		    edges / b->results[im].stats.mean_time_s;
	}

	if (b->benchmark_info.imptype != IMPL_SEQUENTIAL) {
		if (get_cuda_device_info(&b->gpu_info))
			return;
	}

	getcpuinfo(b);
	getmeminfo(b);

	printf("{\n");
	print_sys_info(&(b->sys_info), JSON_INDENT);
	printf(",\n");
	print_gpu_info(&(b->gpu_info), JSON_INDENT);
	printf(",\n");
	print_matrix_info(&(b->matrix_info), JSON_INDENT);
	printf(",\n");
	print_benchmark_info(&(b->benchmark_info), JSON_INDENT);
	printf(",\n");
	print_results(b->results, JSON_INDENT, b->benchmark_info.imptype);
	printf("\n");
	printf("}\n");
}
