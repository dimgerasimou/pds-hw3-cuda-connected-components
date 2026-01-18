/**
 * @file error.c
 * @brief Implementation of error handling utilities.
 */

#define _POSIX_C_SOURCE 200809L

#include "error.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static const char *g_prog = "connected_components_cuda";

/* ------------------------------------------------------------------------- */
/*                              Static Helpers                               */
/* ------------------------------------------------------------------------- */

/**
 * @brief Extract basename from name and set as program name.
 *
 * @param[in] name Program path or name (typically argv[0]).
 */
static void
set_progname(const char *name)
{
	if (!name || !*name)
		return;

	const char *slash = strrchr(name, '/');
	g_prog = slash ? slash + 1 : name;
}

/**
 * @brief Format current local timestamp.
 *
 * Timestamp format: YYYY-MM-DD HH:MM:SS
 *
 * @param[out] buf Output buffer.
 * @param[in]  n   Size of output buffer.
 */
static void
timestamp_now(char *buf, size_t n)
{
	time_t t = time(NULL);
	struct tm tm;

	if (n == 0)
		return;

	if (!localtime_r(&t, &tm)) {
		buf[0] = '\0';
		return;
	}

	strftime(buf, n, "%Y-%m-%d %H:%M:%S", &tm);
}

/* ------------------------------------------------------------------------- */
/*                              Public API                                   */
/* ------------------------------------------------------------------------- */

/**
 * @brief Initalize error reporting.
 *
 * @param[in] name Program name (typically argv[0]).
 */
void
errinit(const char *name)
{
	set_progname(name);
}

/* ------------------------------------------------------------------------- */
/*                            User-facing errors                             */
/* ------------------------------------------------------------------------- */

void
uerrf(const char *fmt, ...)
{
	va_list ap;

	if (!fmt)
		fmt = "?";

	fprintf(stderr, "%s: ", g_prog);

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	fputc('\n', stderr);
}

void
uerrnof(int err, const char *fmt, ...)
{
	va_list ap;

	if (!fmt)
		fmt = "?";

	fprintf(stderr, "%s: ", g_prog);

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	if (err)
		fprintf(stderr, ": %s\n", strerror(err));
	else
		fputc('\n', stderr);
}

/* ------------------------------------------------------------------------- */
/*                          Developer-facing errors                          */
/* ------------------------------------------------------------------------- */

void
vderrf_at(const char *file, int line, const char *func,
          int err, const char *fmt, va_list ap)
{
	char tbuf[32];

	if (!file) file = "?";
	if (!func) func = "?";
	if (!fmt)  fmt  = "?";

	timestamp_now(tbuf, sizeof(tbuf));

	fprintf(stderr, "[%s] %s %s:%d %s: ", tbuf, g_prog, file, line, func);
	vfprintf(stderr, fmt, ap);

	if (err)
		fprintf(stderr, " (errno=%d: %s)\n", err, strerror(err));
	else
		fputc('\n', stderr);
}

void
derrf_at(const char *file, int line, const char *func,
         int err, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vderrf_at(file, line, func, err, fmt, ap);
	va_end(ap);
}

