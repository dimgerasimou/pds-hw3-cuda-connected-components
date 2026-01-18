/**
 * @file args.h
 * @brief Command-line argument parsing interface.
 *
 * This header declares the function used to parse command-line arguments
 * for configuring the program's execution parameters such as
 * number of trials, number of warmup trials, implementation type
 * and input file path.
 */

#ifndef ARGS_H
#define ARGS_H

/**
 * @brief Parses command-line arguments.
 *
 * Supported options:
 *   -n <trials>   Number of trials
 *   -w <wtrials>  Number of warmup trials
 *   -i <imptype>  Implementation type
 *   -h             Show usage and exit
 *
 * Arguments:
 * filepath Path to the input matrix file (Matrix Market format)
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
int parseargs(int argc, char *argv[], unsigned int *trials, unsigned int *wtrials, unsigned int *imptype, char **filepath);

#endif /* ARGS_H */
