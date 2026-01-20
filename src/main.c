/**
 * @file main.c
 * @brief Entry point for the connected components benchmark program.
 *
 * This implementation was developed for the purposes of the class:
 * Parallel and Distributed Systems,
 * Department of Electrical and Computer Engineering,
 * Aristotle University of Thessaloniki.
 *
 * Benchmarks connected components algorithms,on graphs store in .mtx format.
 * Uses cuda to run the different implementations on the gpu.
 * Usage: ./connected_components [-n trials] [-w wtrials] [-i imptype] ./data_filepath
 */

#include <stdlib.h>

#include "args.h"
#include "benchmark.h"
#include "error.h"
#include "matrix.h"

/* default values for cli args */
#define DEFAULT_TRIALS  5
#define DEFAILT_WTRIALS 1
#define DEFAULT_IMPTYPE IMPL_ALL

int
main(int argc, char *argv[])
{
	Matrix *mtx;
	Benchmark *bench;
	int result;
	double load_time;

	/* cmdline args */
	char         *path    = NULL;
	unsigned int trials   = DEFAULT_TRIALS;
	unsigned int wtrials  = DEFAILT_WTRIALS;
	unsigned int imptype  = DEFAULT_IMPTYPE;

	errinit(argv[0]);

	switch (parseargs(argc, argv, &trials, &wtrials, &imptype, &path)) {
		case 1:
			return 1;

		case -1:
			return 0;

		default:
			break;
	}

	{
		double time_start = nowsec();
		mtx = matrixload(path);
		load_time = nowsec() - time_start;
	}

	if (!mtx)
		return 1;

	bench = benchmarkinit(path, trials, wtrials, imptype, mtx);
	if (!bench) {
		matrixfree(mtx);
		return 1;
	}

	bench->matrix_info.load_time_s = load_time;

	if (benchmarkcc(mtx, bench)) {
		benchmarkfree(bench);
		matrixfree(mtx);
		return 1;
	}

	benchmarkprint(bench);

	benchmarkfree(bench);
	matrixfree(mtx);
	return 0;
}
