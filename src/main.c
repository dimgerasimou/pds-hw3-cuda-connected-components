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
#include <stdio.h>

#include "args.h"
#include "error.h"

#define DEFAULT_TRIALS  5
#define DEFAILT_WTRIALS 1
#define DEFAULT_IMPTYPE 0

int
main(int argc, char *argv[])
{
	/* cmdline args */
	char *path            = NULL;
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

	printf("n:%u w:%u i:%u p:%s\n", trials, wtrials, imptype, path);

	return 0;
}
