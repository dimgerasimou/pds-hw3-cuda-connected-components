/**
 * @file json.h
 * @brief Minimal JSON printer for benchmark output.
 *
 * Provides functions to print benchmark structures as valid, properly
 * formatted and escaped JSON to stdout. Designed for easy integration
 * with analysis pipelines and result collection systems.
 */

#ifndef JSON_H
#define JSON_H

#include "benchmark.h"

/**
 * @brief Print system information as formatted JSON.
 *
 * Outputs a JSON object containing CPU information, total RAM, and
 * swap space in gigabytes.
 *
 * @param[in] info         Pointer to SystemInfo structure to print.
 * @param[in] indent_level Number of spaces to indent the output.
 *
 * @note Output is written to stdout.
 */
void print_sys_info(const SystemInfo *info, const unsigned int indent_level);

/**
 * @brief Print CUDA device information as formatted JSON.
 *
 * Outputs a JSON object containing CUDA availability status and,
 * if available, device name, compute capability, and VRAM size.
 *
 * @param[in] gpu_info     Pointer to CudaDeviceInfo structure to print.
 * @param[in] indent_level Number of spaces to indent the output.
 *
 * @note Output is written to stdout.
 */
void print_gpu_info(const CudaDeviceInfo *gpu_info, const unsigned int indent_level);

/**
 * @brief Print matrix information as formatted JSON.
 *
 * Outputs a JSON object containing matrix file path, dimensions,
 * number of non-zero elements, and loading time.
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
 * Outputs a JSON object containing the number of trials, warmup trials,
 * and implementation type index.
 *
 * @param[in] info         Pointer to BenchmarkInfo structure to print.
 * @param[in] indent_level Number of spaces to indent the output.
 *
 * @note Output is written to stdout.
 */
void print_benchmark_info(const BenchmarkInfo *info, const unsigned int indent_level);

/**
 * @brief Print benchmark results array as formatted JSON.
 *
 * Outputs a JSON array containing either all implementation results
 * (if imptype == IMPL_ALL) or a single implementation's result.
 * Each result includes timing statistics, memory usage, and component count.
 *
 * @param[in] results      Pointer to Result array to print.
 * @param[in] indent_level Number of spaces to indent the output.
 * @param[in] imptype      Implementation type of the benchmark.
 *
 * @note Output is written to stdout.
 */
void print_results(const Result *results, const unsigned int indent_level, 
                   const unsigned int imptype);

#endif /* JSON_H */