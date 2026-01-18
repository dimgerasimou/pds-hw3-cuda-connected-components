
/**
 * @file error.h
 * @brief Error handling and reporting utilities.
 *
 * Provides simple helper functions for standardized error reporting.
 * This includes setting the program name (for message prefixes) and
 * printing formatted error messages to `stderr`.
 */

#ifndef ERROR_H
#define ERROR_H

#include <stdarg.h>
#include <errno.h>

/**
 * @brief Initalize error reporting.
 *
 * @param[in] name Program name (typically argv[0]).
 */
void errinit(const char *name);

/**
 * @brief Get program name.
 *
 * @note Caller must free.
 *
 * @return Pointer to program name.
 */
char* get_progname(void);

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
void uerrf(const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
	__attribute__((format(printf, 1, 2)))
#endif
;

/**
 * @brief Prints a user-facing error message to `stderr` with an errno string.
 *
 * Message format:
 *   program_name: <formatted message>: strerror(err)\n   (if err != 0)
 *   program_name: <formatted message>\n                 (if err == 0)
 *
 * Use this for expected failures that have a meaningful errno (file open,
 * read/write, permission errors, etc.).
 *
 * @param[in] err errno value to display (0 to omit strerror()).
 * @param[in] fmt printf-style format string.
 * @param[in] ... Format arguments.
 */
void uerrnof(int err, const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
	__attribute__((format(printf, 2, 3)))
#endif
;

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
void vderrf_at(const char *file, int line, const char *func,
               int err, const char *fmt, va_list ap);

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
void derrf_at(const char *file, int line, const char *func,
              int err, const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
	__attribute__((format(printf, 5, 6)))
#endif
;

/**
 * @brief Developer-facing convenience macros.
 *
 * Automatically injects file/line/function and optionally errno.
 */
#define DERRF(fmt, ...)    derrf_at(__FILE__, __LINE__, __func__, 0, (fmt), ##__VA_ARGS__)
#define DERRNOF(fmt, ...)  derrf_at(__FILE__, __LINE__, __func__, errno, (fmt), ##__VA_ARGS__)

#endif /* ERROR_H */

