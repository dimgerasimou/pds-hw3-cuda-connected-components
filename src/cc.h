/**
 * @file cc.h
 * @brief Connected components counting algorithms for sparse binary matrices.
 *
 * Provides implementation of label propagation algorithms to count connected
 * components in a sparse binary matrix (CSC format).
 */

#ifndef CC_H
#define CC_H

#include "benchmark.h"
#include "matrix.h"

/**
 * @brief Computes connected components sequentialy.
 *
 * Runs sequentialy, on the CPU, using label propagation.
 *
 * @param[in] mtx  Sparse binary matrix in CSC format
 *
 * @return  Number of connected components, or -1 on error
 */
int connected_components_sequential(const Matrix *mtx);

static inline int connected_components(const Matrix *mtx, const unsigned int im) {
	switch (im) {
	case IMPL_SEQUENTIAL:
		return connected_components_sequential(mtx);
	
	case IMPL_CUDA:
		return -1;
	
	default:
		break;
	}

	return -1;
}

#endif /* CC_H */
