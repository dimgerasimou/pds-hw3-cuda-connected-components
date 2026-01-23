/**
 * @file matrix.c
 * @brief CSC (Compressed Sparse Column) binary matrix utilities.
 *
 * Reads MatrixMarket .mtx files into CSC. Values are treated structurally (nonzero => 1).
 *
 * Important optimization for undirected graphs:
 *  - If the input is undirected (MatrixMarket symmetry != "general") OR the env var
 *      CC_UNDIRECTED=1
 *    is set, then we store each undirected edge only once by canonicalizing (i,j) into:
 *      col = max(i,j), row = min(i,j)   (skip i==j)
 *
 * This reduces nnz ~2x for symmetric/general-undirected inputs and removes the need
 * for per-edge filtering in CUDA kernels.
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h> 
#include <limits.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"
#include "error.h"

/* ------------------------------------------------------------------------- */
/*                            Static Helper Functions                        */
/* ------------------------------------------------------------------------- */

static int
streqi(const char *a, const char *b)
{
	for (;; a++, b++) {
		unsigned char ca = (unsigned char)*a;
		unsigned char cb = (unsigned char)*b;
		if (ca >= 'A' && ca <= 'Z') ca = (unsigned char)(ca - 'A' + 'a');
		if (cb >= 'A' && cb <= 'Z') cb = (unsigned char)(cb - 'A' + 'a');
		if (ca != cb) return 0;
		if (ca == 0) return 1;
	}
}

static int
cmp_u32(const void *a, const void *b)
{
	uint32_t x = *(const uint32_t *)a;
	uint32_t y = *(const uint32_t *)b;
	return (x > y) - (x < y);
}

/* Parse a MatrixMarket coordinate line: "i j [val...]"
 * Returns:
 *   1 on success (and sets *nonzero)
 *   0 if line is empty/comment
 *  -1 on parse error
 */
static int
parse_coord_line(const char *line, size_t nrows, size_t ncols,
                 int is_pattern, uint32_t *oi, uint32_t *oj, int *nonzero)
{
	/* skip leading spaces */
	while (*line == ' ' || *line == '\t' || *line == '\r')
		line++;

	if (*line == '\0' || *line == '\n' || *line == '%')
		return 0;

	char *end = NULL;

	errno = 0;
	unsigned long long i = strtoull(line, &end, 10);
	if (errno || end == line) return -1;
	if (i > ULLONG_MAX) return -1;  /* Overflow check */
	line = end;

	errno = 0;
	unsigned long long j = strtoull(line, &end, 10);
	if (errno || end == line) return -1;
	if (j > ULLONG_MAX) return -1;  /* Overflow check */
	line = end;

	if (i == 0 || j == 0) {
		DERRF("matrix coordinate entry has zero index (i=%llu, j=%llu)", i, j);
		return -1;
	}
	
	if (i > nrows || j > ncols) {
		DERRF("matrix coordinate entry out of bounds (i=%llu, j=%llu, nrows=%zu, ncols=%zu)", i, j, nrows, ncols);
		return -1;
	}

	/* MatrixMarket is 1-based */
	*oi = (uint32_t)(i - 1);
	*oj = (uint32_t)(j - 1);

	if (is_pattern) {
		*nonzero = 1;
		return 1;
	}

	/* For non-pattern, parse the next token as double and check != 0. */
	while (*line == ' ' || *line == '\t' || *line == '\r')
		line++;

	if (*line == '\0' || *line == '\n')
		return -1;

	errno = 0;
	double v = strtod(line, &end);
	if (errno || end == line) return -1;

	*nonzero = (v != 0.0);
	return 1;
}

/* Unique in-place for sorted slice, returns new length */
static size_t
unique_u32(uint32_t *a, size_t n)
{
	if (n == 0) return 0;
	size_t w = 1;
	for (size_t r = 1; r < n; r++) {
		if (a[r] != a[w - 1])
			a[w++] = a[r];
	}
	return w;
}

static int
env_truthy(const char *name)
{
	const char *v = getenv(name);
	if (!v || !*v) return 0;
	/* accept: 1, true, yes, on (case-insensitive-ish) */
	if (v[0] == '1') return 1;
	if (v[0] == 't' || v[0] == 'T') return 1;
	if (v[0] == 'y' || v[0] == 'Y') return 1;
	if (v[0] == 'o' || v[0] == 'O') return 1; /* "on" */
	return 0;
}

/* ------------------------------------------------------------------------- */
/*                           Public API Functions                            */
/* ------------------------------------------------------------------------- */

Matrix*
matrixload(const char *path)
{
	FILE *f = fopen(path, "r");
	if (!f) {
		uerrnof(errno, "cannot open \"%s\"", path);
		return NULL;
	}

	/* --- Read and parse header line ----------------------------------- */
	char *line = NULL;
	size_t cap = 0;
	ssize_t nread;

	nread = getline(&line, &cap, f);
	if (nread < 0) {
		uerrf("failed to read matrix header");
		fclose(f);
		return NULL;
	}

	/* %%MatrixMarket matrix <format> <field> <symmetry> */
	char banner[64], object[64], format[64], field[64], sym[64];
	if (sscanf(line, "%63s %63s %63s %63s %63s", banner, object, format, field, sym) != 5 ||
	    strcmp(banner, "%%MatrixMarket") != 0 || !streqi(object, "matrix")) {
		uerrf("invalid MatrixMarket header");
		free(line);
		fclose(f);
		return NULL;
	}

	if (!streqi(format, "coordinate")) {
		uerrf("only coordinate .mtx supported for graphs");
		free(line);
		fclose(f);
		return NULL;
	}

	int is_pattern = streqi(field, "pattern");
	/* Accept real/integer/complex too, but we treat nonzero as 1. */
	if (!is_pattern && !streqi(field, "real") && !streqi(field, "integer") && !streqi(field, "complex")) {
		uerrf("unsupported matrix field: %s", field);
		free(line);
		fclose(f);
		return NULL;
	}

	int general   = streqi(sym, "general");
	int symmetric = streqi(sym, "symmetric");
	int skew      = streqi(sym, "skew-symmetric");
	int herm      = streqi(sym, "hermitian");

	if (!general && !symmetric && !skew && !herm) {
		uerrf("unsupported matrix symmetry: %s", sym);
		free(line);
		fclose(f);
		return NULL;
	}

	/* Decide undirected canonicalization mode:
	   - symmetric/skew/hermitian imply undirected structure
	   - or user forces with env var CC_UNDIRECTED=1
	*/
	int undirected = (!general) || env_truthy("CC_UNDIRECTED");

	/* In undirected mode, we DO NOT expand. We store each edge once as (min -> max). */
	int expand = 0;

	/* --- Skip comments and read size line ------------------------------ */
	size_t nrows = 0, ncols = 0;
	size_t nnz_decl = 0;

	for (;;) {
		nread = getline(&line, &cap, f);
		if (nread < 0) {
			uerrf("missing matrix size line");
			free(line);
			fclose(f);
			return NULL;
		}
		if (line[0] == '%')
			continue;

		if (sscanf(line, "%zu %zu %zu", &nrows, &ncols, &nnz_decl) != 3 ||
		    nrows == 0 || ncols == 0) {
			uerrf("invalid matrix size line");
			free(line);
			fclose(f);
			return NULL;
		}
		break;
	}

	if (nrows > UINT32_MAX || ncols > UINT32_MAX) {
		uerrf("matrix dimensions exceed uint32_t");
		free(line);
		fclose(f);
		return NULL;
	}

	/* --- Pass 1: count entries per column ------------------------------ */
	size_t *colcnt = (size_t *)calloc(ncols, sizeof(size_t));
	if (!colcnt) {
		DERRNOF("calloc failed");
		free(line);
		fclose(f);
		return NULL;
	}

	for (;;) {
		nread = getline(&line, &cap, f);
		if (nread < 0)
			break;

		uint32_t i, j;
		int nonzero;
		int r = parse_coord_line(line, nrows, ncols, is_pattern, &i, &j, &nonzero);
		if (r == 0) continue;
		if (r < 0) {
			uerrf("malformed matrix coordinate entry");
			goto fail;
		}
		if (!nonzero) continue;

		if (undirected) {
			if (i == j) continue; /* drop self loops */
			uint32_t hi = (i < j) ? j : i;
			colcnt[hi]++;         /* store in column hi, row lo */
		} else {
			colcnt[j]++;
			if (expand && i != j)
				colcnt[i]++;
		}
	}

	/* Build colptr via prefix sum */
	uint32_t *colptr = (uint32_t *)malloc((ncols + 1) * sizeof(uint32_t));
	if (!colptr) {
		DERRNOF("malloc failed");
		goto fail;
	}

	colptr[0] = 0;
	size_t run = 0;
	for (size_t c = 0; c < ncols; c++) {
		run += colcnt[c];
		if (run > (size_t)UINT32_MAX) {
			uerrf("nnz exceeds uint32_t");
			free(colptr);
			goto fail;
		}
		colptr[c + 1] = (uint32_t)run;
	}

	size_t nnz = (size_t)colptr[ncols];
	uint32_t *rowi = (uint32_t *)malloc(nnz * sizeof(uint32_t));
	if (!rowi) {
		DERRNOF("malloc failed");
		free(colptr);
		goto fail;
	}

	/* --- Pass 2: fill row indices ------------------------------------- */
	if (fseek(f, 0, SEEK_SET) != 0) {
		DERRF("fseek failed");
		free(rowi);
		free(colptr);
		goto fail;
	}

	/* skip header */
	nread = getline(&line, &cap, f);
	if (nread < 0) {
		free(rowi);
		free(colptr);
		goto fail;
	}

	/* skip comments to size line again */
	for (;;) {
		nread = getline(&line, &cap, f);
		if (nread < 0) {
			free(rowi);
			free(colptr);
			goto fail;
		}
		if (line[0] == '%')
			continue;
		break; /* size line */
	}

	/* next[c] starts at colptr[c] */
	uint32_t *next = (uint32_t *)malloc(ncols * sizeof(uint32_t));
	if (!next) {
		DERRNOF("malloc failed");
		free(rowi);
		free(colptr);
		goto fail;
	}
	for (size_t c = 0; c < ncols; c++)
		next[c] = colptr[c];

	for (;;) {
		nread = getline(&line, &cap, f);
		if (nread < 0)
			break;

		uint32_t i, j;
		int nonzero;
		int r = parse_coord_line(line, nrows, ncols, is_pattern, &i, &j, &nonzero);
		if (r == 0) continue;
		if (r < 0) {
			uerrf("malformed matrix coordinate entry");
			free(next);
			free(rowi);
			free(colptr);
			goto fail;
		}
		if (!nonzero) continue;

		if (undirected) {
			if (i == j) continue;
			uint32_t lo = (i < j) ? i : j;
			uint32_t hi = (i < j) ? j : i;
			rowi[next[hi]++] = lo;
		} else {
			rowi[next[j]++] = i;
			if (expand && i != j)
				rowi[next[i]++] = j;
		}
	}

	free(next);
	free(colcnt);
	free(line);
	fclose(f);

	/* --- Sort + dedup per column -------------------------------------- */
	int T = 1;
#ifdef _OPENMP
	T = omp_get_max_threads();
	if (T < 1) T = 1;
#endif

	#pragma omp parallel for schedule(dynamic, 512) if(T > 1)
	for (size_t c = 0; c < ncols; c++) {
		uint32_t a = colptr[c];
		uint32_t b = colptr[c + 1];
		if (b > a)
			qsort(rowi + a, (size_t)(b - a), sizeof(uint32_t), cmp_u32);
	}

	/* Dedup requires compaction + new colptr (simple 2-pass) */
	size_t *uniqcnt = (size_t *)calloc(ncols, sizeof(size_t));
	if (!uniqcnt) {
		Matrix *m = (Matrix *)calloc(1, sizeof(Matrix));
		if (!m) { free(rowi); free(colptr); return NULL; }
		m->nrows = nrows; m->ncols = ncols; m->nnz = nnz;
		m->rowi = rowi; m->colptr = colptr;
		return m;
	}

	#pragma omp parallel for schedule(dynamic, 512) if(T > 1)
	for (size_t c = 0; c < ncols; c++) {
		uint32_t a = colptr[c];
		uint32_t b = colptr[c + 1];
		uniqcnt[c] = unique_u32(rowi + a, (size_t)(b - a));
	}

	uint32_t *newcol = (uint32_t *)malloc((ncols + 1) * sizeof(uint32_t));
	if (!newcol) {
		free(uniqcnt);
		Matrix *m = (Matrix *)calloc(1, sizeof(Matrix));
		if (!m) { free(rowi); free(colptr); return NULL; }
		m->nrows = nrows; m->ncols = ncols; m->nnz = nnz;
		m->rowi = rowi; m->colptr = colptr;
		return m;
	}

	newcol[0] = 0;
	size_t runu = 0;
	for (size_t c = 0; c < ncols; c++) {
		runu += uniqcnt[c];
		newcol[c + 1] = (uint32_t)runu;
	}

	uint32_t *newrow = (uint32_t *)malloc((size_t)newcol[ncols] * sizeof(uint32_t));
	if (!newrow) {
		free(uniqcnt);
		free(newcol);
		Matrix *m = (Matrix *)calloc(1, sizeof(Matrix));
		if (!m) { free(rowi); free(colptr); return NULL; }
		m->nrows = nrows; m->ncols = ncols; m->nnz = nnz;
		m->rowi = rowi; m->colptr = colptr;
		return m;
	}

	#pragma omp parallel for schedule(dynamic, 512) if(T > 1)
	for (size_t c = 0; c < ncols; c++) {
		uint32_t olda = colptr[c];
		size_t   ulen = uniqcnt[c];

		uint32_t out = newcol[c];
		memcpy(newrow + out, rowi + olda, ulen * sizeof(uint32_t));
	}

	free(uniqcnt);
	free(rowi);
	free(colptr);

	Matrix *m = (Matrix *)calloc(1, sizeof(Matrix));
	if (!m) {
		free(newrow);
		free(newcol);
		return NULL;
	}

	m->nrows = nrows;
	m->ncols = ncols;
	m->nnz   = (size_t)newcol[ncols];
	m->rowi  = newrow;
	m->colptr = newcol;

	return m;

fail:
	free(colcnt);
	free(line);
	fclose(f);
	return NULL;
}

/**
 * @brief Free a CSCBinaryMatrix and its associated memory.
 *
 * @note Safe to call with NULL.
 *
 * @param[in] m Matrix to free.
 */
void
matrixfree(Matrix *matrix)
{
	if (!matrix)
		return;

	if (matrix->rowi) {
		free(matrix->rowi);
		matrix->rowi = NULL;
	}

	if (matrix->colptr) {
		free(matrix->colptr);
		matrix->colptr = NULL;
	}

	free(matrix);
}