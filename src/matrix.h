/**
 * @file matrix.h
 * @brief CSC (Compressed Sparse Column) binary matrix structure and API.
 *
 * Provides functionality to read and free sparse binary matrices
 * stored in CSC format. Non-zero values are represented implicitly as 1.
 * Reads from MatrixMarket .mtx files.
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#include <stdint.h>

/**
 * @struct Matrix
 * @brief Compressed Sparse Column (CSC) representation of a binary matrix.
 *
 * Non-zero entries are implicitly 1. Stores only row indices and column 
 * pointers. Used to represent adjacency matrices for graph algorithms.
 *
 * For a matrix with ncols columns:
 * - Column j's non-zero row indices are stored in rowi[colptr[j]..colptr[j+1])
 * - Total non-zeros (edges) is colptr[ncols]
 */
typedef struct {
	size_t nrows;     /**< Number of rows in the matrix */
	size_t ncols;     /**< Number of columns in the matrix */
	size_t nnz;       /**< Number of non-zero (1) entries */
	uint32_t *rowi;   /**< Row indices of non-zero elements (length nnz) */
	uint32_t *colptr; /**< Column pointers (length ncols + 1) */
} Matrix;

/**
 * @brief Load a sparse binary matrix from a MatrixMarket .mtx file.
 *
 * Reads coordinate format matrices and converts them to CSC format.
 * For undirected graphs (symmetric/skew-symmetric/Hermitian matrices or
 * when CC_UNDIRECTED=1 environment variable is set), edges are stored
 * only once in canonical form: (min -> max).
 *
 * Self-loops are removed in undirected mode.
 * Duplicate entries are removed.
 *
 * @note The returned matrix must be freed using matrixfree().
 *
 * @param[in] path Path to the .mtx file.
 *
 * @return Newly allocated Matrix, or NULL on failure.
 */
Matrix* matrix_load(const char *path);

/**
 * @brief Free a Matrix and its associated memory.
 *
 * Frees the row index array, column pointer array, and the Matrix structure
 * itself. Safe to call with NULL.
 *
 * @param[in] matrix Matrix to free.
 */
void matrix_free(Matrix *matrix);

#endif /* MATRIX_H */