/**
 * @file main.c
 * @brief Entry point for the connected components benchmark program.
 *
 * This implementation was developed for the purposes of the class:
 * Parallel and Distributed Systems,
 * Department of Electrical and Computer Engineering,
 * Aristotle University of Thessaloniki.
 *
 * Benchmarks connected components algorithms on graphs stored in .mtx format.
 * Uses CUDA to run different implementations on the GPU.
 *
 * Usage: ./connected_components [-n trials] [-w wtrials] [-i imptype] <matrix_file>
 */

#include "args.h"
#include "benchmark.h"
#include "error.h"
#include "matrix.h"

/* ------------------------------------------------------------------------- */
/*                              Default Values                               */
/* ------------------------------------------------------------------------- */

#define DEFAULT_TRIALS  5
#define DEFAULT_WTRIALS 1
#define DEFAULT_IMPTYPE IMPL_ALL

/* ------------------------------------------------------------------------- */
/*                                Main Function                              */
/* ------------------------------------------------------------------------- */

/**
 * @brief Program entry point.
 *
 * Parses command-line arguments, loads the input matrix, initializes the
 * benchmark structure, runs the connected components algorithm(s), and
 * prints results in JSON format.
 *
 * @param[in] argc Argument count.
 * @param[in] argv Argument vector.
 *
 * @return 0 on success, 1 on error.
 */
int
main(int argc, char *argv[])
{
	Matrix *mtx = NULL;
	Benchmark *bench = NULL;
	double load_time;
	int ret = 1;

	/* Command-line arguments with defaults */
	char *path = NULL;
	unsigned int trials = DEFAULT_TRIALS;
	unsigned int wtrials = DEFAULT_WTRIALS;
	unsigned int imptype = DEFAULT_IMPTYPE;

	/* Initialize error reporting with program name */
	err_init(argv[0]);

	/* Parse command-line arguments */
	switch (parse_args(argc, argv, &trials, &wtrials, &imptype, &path)) {
	case 1:
		/* Parse error */
		return 1;

	case -1:
		/* Help requested */
		return 0;

	default:
		/* Success, continue */
		break;
	}

	/* Load matrix from file and measure loading time */
	{
		double time_start = now_sec();
		mtx = matrix_load(path);
		load_time = now_sec() - time_start;
	}

	if (!mtx)
		goto cleanup;

	/* Initialize benchmark structure */
	bench = benchmark_init(path, trials, wtrials, imptype, mtx);
	if (!bench)
		goto cleanup;

	/* Record matrix load time in benchmark metadata */
	bench->matrix_info.load_time_s = load_time;

	/* Run the benchmark */
	if (benchmark_cc(mtx, bench))
		goto cleanup;

	/* Print results in JSON format */
	benchmark_print(bench);

	/* Success */
	ret = 0;

cleanup:
	benchmark_free(bench);
	matrix_free(mtx);
	return ret;
}