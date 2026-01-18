/**
 * @file matrix.h
 * @brief CSC (Compressed Sparse Column) binary matrix structure and API.
 *
 * Provides functionality to read and free sparse binary matrices
 * stored in CSC format. Non-zero values are represented implicitly as 1.
 * Reads from .mtx files.
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#include <stdint.h>

/**
 * @struct Matrix
 * @brief Compressed Sparse Column (CSC) representation of a binary matrix.
 *
 * Non-zero entries are implicitly 1. Stores only row indices and column pointers.
 */
typedef struct {
	size_t nrows;     /**< Number of rows in the matrix */
	size_t ncols;     /**< Number of columns in the matrix */
	size_t nnz;       /**< Number of non-zero (1) entries */
	uint32_t *rowi;   /**< Row indices of non-zero elements (length nnz) */
	uint32_t *colptr; /**< Column pointers (length ncols + 1) */
} Matrix;

/** @brief Load a sparse binary matrix from a .mtx file.
 *
 * @note The returned matrix must be freed using matrixfree().
 *
 * @param[in] path Path to the matrix file.
 *
 * @return Newly allocated Matrix, or NULL on failure.
 */
Matrix* matrixload(const char *path);

/**
 * @brief Free a CSCBinaryMatrix and its associated memory.
 *
 * @note Safe to call with NULL.
 *
 * @param[in] m Matrix to free.
 */
void matrixfree(Matrix *matrix);

#endif /* MATRIX_H */
