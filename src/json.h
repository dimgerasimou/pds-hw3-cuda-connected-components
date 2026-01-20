/**
 * @file json.h
 * @brief Minimal JSON printer for benchmark output.
 */

#ifndef JSON_H
#define JSON_H

#include "benchmark.h"

/**
 * @brief Print system information as formatted JSON.
 *
 * @param[in] info         Pointer to SystemInfo structure to print.
 * @param[in] indent_level Number of spaces to indent the output.
 *
 * @note Output is written to stdout.
 */
void print_sys_info(const SystemInfo *info, const unsigned int indent_level);

/**
 * @brief Print matrix information as formatted JSON.
 *
 * @param[in] info         Pointer to MatrixInfo structure to print.
 * @param[in] indent_level Number of spaces to indent the output.
 *
 * @note Output is written to stdout.
 */
void print_matrix_info(const MatrixInfo *info, const unsigned int indent_level);

/**
 * @brief Print benchmark parameters as formatted JSON.
 *
 * @param[in] info         Pointer to BenchmarkInfo structure to print.
 * @param[in] indent_level Number of spaces to indent the output.
 *
 * @note Output is written to stdout.
 */
void print_benchmark_info(const BenchmarkInfo *info, const unsigned int indent_level);

/**
 * @brief Print results as formatted JSON.
 *
 * @param[in] result       Pointer to Result structure to print.
 * @param[in] indent_level Number of spaces to indent the output.
 * @param[in] imptype      Implementation type of the benchmark.
 *
 * @note Output is written to stdout.
 */
void print_results(const Result *results, const unsigned int indent_level, const unsigned int imptype);

#endif /* JSON_H */
