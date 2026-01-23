/**
 * @file args.h
 * @brief Command-line argument parsing interface.
 *
 * This header declares the function used to parse command-line arguments
 * for configuring the program's execution parameters such as number of
 * trials, number of warmup trials, implementation type, and input file path.
 *
 * Supports standard POSIX-style option parsing with validation.
 */

#ifndef ARGS_H
#define ARGS_H

/**
 * @brief Parses command-line arguments.
 *
 * Validates all arguments and ensures the matrix file exists and is readable.
 * Provides helpful error messages and usage information on invalid input.
 *
 * Supported options:
 *   -n <trials>   Number of benchmark trials (must be > 0)
 *   -w <wtrials>  Number of warmup trials (>= 0)
 *   -i <imptype>  Implementation type (0 to IMPL_ALL)
 *   -h            Show usage and exit
 *
 * Required argument:
 *   <matrix_file> Path to the input matrix file (MatrixMarket .mtx format)
 *
 * @param[in]  argc     Argument count from main().
 * @param[in]  argv     Argument vector from main().
 * @param[out] trials   Output: number of benchmark trials.
 * @param[out] wtrials  Output: number of warmup trials.
 * @param[out] imptype  Output: implementation type index.
 * @param[out] filepath Output: path to matrix file.
 *
 * @return 0 on success, -1 if help requested, 1 on error.
 */
int parse_args(int argc, char *argv[], unsigned int *trials, 
              unsigned int *wtrials, unsigned int *imptype, char **filepath);

#endif /* ARGS_H */