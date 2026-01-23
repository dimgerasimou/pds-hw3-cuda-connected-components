/**
 * @file error.c
 * @brief Implementation of error handling utilities.
 */

#define _POSIX_C_SOURCE 200809L

#include "error.h"

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
	struct tm *tm;

	if (n == 0)
		return;

	tm = localtime(&t);

	if (tm)
		strftime(buf, n, "%Y-%m-%d %H:%M:%S", tm);
	else
		buf[0] = '\0';
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
err_init(const char *name)
{
	set_progname(name);
}

/**
 * @brief Get program name.
 *
 * @note Caller must free.
 *
 * @return Pointer to program name.
 */
char*
get_progname(void)
{
	char *ret = strdup(g_prog);
	if (!ret) {
		DERRNOF("strdup failed");
		return NULL;
	}
	return ret;
}

/* ------------------------------------------------------------------------- */
/*                            User-facing errors                             */
/* ------------------------------------------------------------------------- */

/**
 * @brief Prints a user-facing error message to `stderr`.
 *
 * Message format:
 *   program_name: <formatted message>\n
 *
 * Use this for expected runtime/CLI errors (bad arguments, missing input,
 * invalid values, etc.).
 *
 * @param[in] fmt printf-style format string.
 * @param[in] ... Format arguments.
 */
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

/**
 * @brief Prints a user-facing error message to `stderr` with an errno string.
 *
 * Message format:
 *   program_name: <formatted message>: strerror(err)\n   (if err != 0)
 *   program_name: <formatted message>\n                  (if err == 0)
 *
 * Use this for expected failures that have a meaningful errno (file open,
 * read/write, permission errors, etc.).
 *
 * @param[in] err errno value to display (0 to omit strerror()).
 * @param[in] fmt printf-style format string.
 * @param[in] ... Format arguments.
 */
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

/**
 * @brief Prints a developer-facing error message to `stderr`.
 *
 * Message format:
 *   [timestamp] program_name file:line function: <formatted message>\n
 *
 * If @p err is non-zero, appends:
 *   (errno=<err>: strerror(err))
 *
 * @note Use the macros provided so file:line:func are automatic.
 *
 * @param[in] file Source file the error happened.
 * @param[in] line Line number the error happened.
 * @param[in] func Function name the error happened.
 * @param[in] err  errno value to display (0 to omit strerror()).
 * @param[in] fmt  printf-style format string.
 * @param[in] ap   Format argument list.
 */
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

/**
 * @brief Prints a developer-facing error message to `stderr` (printf-style).
 *
 * Same output as vderrf_at(), but takes variadic arguments.
 *
 * @note Use the macros provided so file:line:func are automatic.
 *
 * @param[in] file Source file the error happened.
 * @param[in] line Line number the error happened.
 * @param[in] func Function name the error happened.
 * @param[in] err  errno value to display (0 to omit strerror()).
 * @param[in] fmt  printf-style format string.
 * @param[in] ...  Format arguments.
 */
void
derrf_at(const char *file, int line, const char *func,
         int err, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vderrf_at(file, line, func, err, fmt, ap);
	va_end(ap);
}