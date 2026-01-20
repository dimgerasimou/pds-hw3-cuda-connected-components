/**
 * @file cc_sequential.c
 * @brief Optimized label propagation sequential algorithm
 *        for computing connected components.
 */

#include <stdlib.h>
#include <stdint.h>

#include "cc.h"
#include "error.h"

/**
 * @brief Computes connected components using optimized label propagation.
 *
 * Algorithm steps:
 * 1. Initialize each node with its own index as label
 * 2. Iterate over all edges, propagating minimum labels
 * 3. Use cached column label to reduce redundant reads
 * 4. Repeat until no labels change (convergence)
 * 5. Count unique components using bitmap with hardware popcount
 *
 * Optimization: Cache the column label in the inner loop to avoid
 * redundant memory reads when processing multiple edges in the same column.
 *
 * @param m  Sparse binary matrix in CSC format representing graph
 *
 * @return  Number of connected components, or -1 on error
 */
int
connected_components_sequential(const Matrix *m)
{
	if (!m || m->nrows != m->ncols) {
		DERRF("connected components expects a square adjacency matrix (rows=%zu, cols=%zu)", m ? m->nrows : 0, m ? m->ncols : 0);
		return -1;
	}

	uint32_t *label = malloc(sizeof(uint32_t) * m->nrows);
	if (!label) {
		DERRNOF("malloc() failed");
		return -1;
	}
	
	/* Initialize: each node labeled with its own index */
	for (size_t i = 0; i < m->nrows; i++) {
		label[i] = i;
	}
	
	/* Iterate until convergence */
	uint8_t finished;
	do {
		finished = 1;
		
		/* Process all edges, propagating minimum labels */
		for (size_t i = 0; i < m->ncols; i++) {
			uint32_t col_label = label[i];  /* Cache column label */
			
			for (uint32_t j = m->colptr[i]; j < m->colptr[i + 1]; j++) {
				uint32_t row = m->rowi[j];
				uint32_t row_label = label[row];
				
				if (col_label != row_label) {
					uint32_t min_label = col_label < row_label ? col_label : row_label;
					
					/* Update column label if needed (and cache it) */
					if (col_label > min_label) {
						label[i] = col_label = min_label;
						finished = 0;
					}
					
					/* Update row label if needed */
					if (row_label > min_label) {
						label[row] = min_label;
						finished = 0;
					}
				}
			}
		}
	} while (!finished);
	
	/* Count unique components using a bitmap */
	size_t bitmap_size = (m->nrows + 63) / 64;
	uint64_t *bitmap = calloc(bitmap_size, sizeof(uint64_t));
	if (!bitmap) {
		DERRNOF("calloc() failed");
		free(label);
		return -1;
	}
	
	/* Bitmap construction: set bit for each unique label */
	for (uint32_t i = 0; i < m->nrows; i++) {
		uint32_t val = label[i];
		bitmap[val >> 6] |= (1ULL << (val & 63));
	}
	
	/* Count set bits using hardware popcount */
	uint32_t count = 0;
	for (size_t i = 0; i < bitmap_size; i++) {
		count += __builtin_popcountll(bitmap[i]);
	}
	
	free(label);
	free(bitmap);
	return (int)count;
}
