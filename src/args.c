/**
 * @file args.c
 * @brief Command-line argument parsing implementation.
 *
 * Provides functions to parse program arguments.
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "args.h"
#include "benchmark.h"
#include "error.h"

/* ------------------------------------------------------------------------- */
/*                            Static Helper Functions                        */
/* ------------------------------------------------------------------------- */

/**
 * @brief Prints program usage instructions.
 *
 * Displays the valid command-line options and their expected arguments.
 */
static void
usage(void)
{
	char *program_name = get_progname();

	if (!program_name)
		return;

	printf(
		"\n"
		"Usage: %s [OPTIONS] <matrix_file>\n\n"
		"Options:\n"
		"  -n <trials>        Number of benchmark trials (default: 5)\n"
		"  -w <wtrials>       Number of warmup trials    (default: 1)\n"
		"  -i <imptype>       Implementation type        (default: 0)\n"
		"  -h                 Show this help message and exit\n\n"
		"Arguments:\n"
		"  matrix_file Path to the input matrix file (.mtx file format)\n\n"
		"Example:\n"
		"  %s -n 10 -w 2 -i 1 ./data/graph.mtx\n",
		program_name, program_name
	);

	free(program_name);
}

/**
 * @brief Parse an unsigned int from a decimal string with full validation.
 *
 * Accepts only a base-10 non-negative integer (no leading sign, no trailing
 * garbage). Detects overflow and reports failure.
 *
 * @param[in]  s   Input string.
 * @param[out] out Output value.
 * @return 1 on success, 0 on failure.
 */
static int
parse_uint(const char *s, unsigned int *out)
{
	char *end = NULL;
	unsigned long v;

	if (!s || !*s)
		return 0;

	errno = 0;
	v = strtoul(s, &end, 10);

	if (errno != 0)
		return 0;
	if (end == s || *end != '\0')
		return 0;
	if (v > UINT_MAX)
		return 0;

	*out = (unsigned int)v;
	return 1;
}

/**
 * @brief Emit a consistent error for invalid/missing numeric argument.
 */
static int
badnum(char opt)
{
	uerrf("invalid or missing argument for -%c", opt);
	usage();
	return 1;
}

/**
 * @brief Emit a consistent error for missing/unknown option.
 */
static int
badopt(int opt, int is_missing_arg)
{
	if (is_missing_arg)
		uerrf("missing argument for -%c", opt);
	else
		uerrf("unknown option '-%c'", opt ? opt : '?');
	usage();
	return 1;
}

/* ------------------------------------------------------------------------- */
/*                            Public API Functions                           */
/* ------------------------------------------------------------------------- */

/**
 * @brief Parses command-line arguments.
 *
 * Supported options:
 *   -n <trials>   Number of trials (must be > 0)
 *   -w <wtrials>  Number of warmup trials (must be > 0)
 *   -i <imptype>  Implementation type (>= 0)
 *   -h            Show usage and exit
 *
 * Arguments:
 *   <matrix_file> Path to the input matrix file (Matrix Market format)
 *
 * @param[in]  argc     Argument count.
 * @param[in]  argv     Argument vector.
 * @param[out] trials   Output: number of trials.
 * @param[out] wtrials  Output: number of warmup trials.
 * @param[out] imptype  Output: implementation type.
 * @param[out] filepath Output: path to matrix file.
 *
 * @return 0 on success, -1 if help requested, 1 on error
 */
int
parseargs(int argc, char *argv[],
          unsigned int *trials, unsigned int *wtrials,
          unsigned int *imptype, char **filepath)
{
	int opt;

	opterr = 0;

	while ((opt = getopt(argc, argv, "+n:w:i:h")) != -1) {
		switch (opt) {
		case 'n': {
			unsigned int v;
			if (!parse_uint(optarg, &v))
				return badnum('n');
			if (v == 0) {
				uerrf("trials must be > 0");
				usage();
				return 1;
			}
			if (trials)
				*trials = v;
			break;
		}
		case 'w': {
			unsigned int v;
			if (!parse_uint(optarg, &v))
				return badnum('w');
			if (wtrials)
				*wtrials = v;
			break;
		}
		case 'i': {
			unsigned int v;
			if (!parse_uint(optarg, &v))
				return badnum('i');
			if (v > IMPL_ALL) {
				uerrf("implementation type is invalid. Must be <= %u", IMPL_ALL);
				usage();
				return 1;
			}
			if (imptype)
				*imptype = v;
			break;
		}
		case 'h':
			usage();
			return -1;

		case '?':
		default:
			if (optopt == 'n' || optopt == 'w' || optopt == 'i')
				return badopt(optopt, 1);
			return badopt(optopt ? optopt : '?', 0);
		}
	}

	/* Expect exactly one positional argument: the matrix file */
	if (optind >= argc) {
		uerrf("no input file specified");
		usage();
		return 1;
	}

	if (filepath)
		*filepath = argv[optind];
	else {
		DERRF("filepath is NULL");
		usage();
		return 1;
	}

	if (optind + 1 < argc) {
		uerrf("too many arguments");
		usage();
		return 1;
	}

	if (access(*filepath, R_OK) != 0) {
		uerrnof(errno, "cannot access file: \"%s\"", *filepath);
		usage();
		return 1;
	}

	return 0;
}

