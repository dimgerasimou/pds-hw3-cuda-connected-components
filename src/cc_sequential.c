/**
 * @file cc_sequential.c
 * @brief Optimized union-find sequential algorithm with better locality
 *        for computing connected components.
 */

#include <stdlib.h>
#include <stdint.h>

#include "cc.h"
#include "error.h"

/**
 * @brief Finds the root of a node with path halving optimization.
 *
 * Path halving is a one-pass variant of path compression that makes every
 * node point to its grandparent, effectively halving the path length on
 * each traversal. This provides nearly the same performance as full path
 * compression with less overhead.
 *
 * @param[in,out] label Array where label[i] is the parent of node i.
 * @param[in]     i     Node to find root for.
 *
 * @return Root node (where label[root] == root)
 */
static inline uint32_t
find_root_halving(uint32_t *label, uint32_t i)
{
	while (label[i] != i) {
		label[i] = label[label[i]]; /* Path halving: skip one level */
		i = label[i];
	}
	return i;
}

/**
 * @brief Unites two nodes by attaching their roots.
 *
 * This performs union-by-index, where the root with the larger index is
 * always attached to the root with the smaller index. This maintains a
 * canonical form where component representatives are always the minimum
 * node index in each component.
 *
 * @param[in,out] label Array of parent pointers.
 * @param[in]     i     First node.
 * @param[in]     j     Second node.
 *
 * @return 1 if union was performed, 0 if nodes already in same set
 */
static inline int
union_nodes_by_index(uint32_t *label, uint32_t i, uint32_t j)
{
	uint32_t root_i = find_root_halving(label, i);
	uint32_t root_j = find_root_halving(label, j);
	
	if (root_i == root_j)
		return 0;
	
	/* Attach larger index to smaller (maintains canonical form) */
	if (root_i < root_j) {
		label[root_j] = root_i;
	} else {
		label[root_i] = root_j;
	}
	return 1;
}

/**
 * @brief Computes connected components using union-find algorithm.
 *
 * Algorithm steps:
 * 1. Initialize each node as its own parent (singleton sets)
 * 2. For each edge (i,j), union the sets containing i and j
 * 3. Perform final path compression to flatten all trees
 * 4. Count nodes that are their own parent (roots = components)
 *
 * @param[in]  m          Sparse binary matrix in CSC format representing graph.
 * @param[out] iterations Iterations needed by the algorithm to converge
 *                        (always 1 in this union find)
 * @return Number of connected components, or -1 on error
 */
int
connected_components_sequential(const Matrix *m)
{
	if (!m || m->nrows != m->ncols) {
		DERRF("connected components expects a square adjacency matrix (rows=%zu, cols=%zu)", m ? m->nrows : 0, m ? m->ncols : 0);
		return -1;
	}

	uint32_t *label = malloc(m->nrows * sizeof(uint32_t));
	if (!label) {
		DERRNOF("malloc() failed");
		return -1;
	}
	
	/* Initialize: each node is its own parent */
	for (size_t i = 0; i < m->nrows; i++) {
		label[i] = i;
	}
	
	/* Process all edges: union connected nodes */
	for (size_t i = 0; i < m->ncols; i++) {
		uint32_t col_start = m->colptr[i];
		uint32_t col_end = m->colptr[i + 1];
		
		for (uint32_t j = col_start; j < col_end; j++) {
			union_nodes_by_index(label, i, m->rowi[j]);
		}
	}
	
	/* Final compression pass: flatten all paths for accurate counting */
	for (size_t i = 0; i < m->nrows; i++) {
		find_root_halving(label, i);
	}
	
	/* Count roots (each root represents one component) */
	uint32_t unique_count = 0;
	for (size_t i = 0; i < m->nrows; i++) {
		if (label[i] == i) {
			unique_count++;
		}
	}
	
	free(label);
	return (int)unique_count;
}