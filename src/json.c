/**
 * @file json.c
 * @brief Minimal JSON printer for benchmark output (valid JSON).
 */

#include <stdio.h>
#include "json.h"

static void
json_print_escaped(const char *s)
{
	const unsigned char *p = (const unsigned char *)(s ? s : "");
	putchar('"');
	for (; *p; p++) {
		switch (*p) {
		case '\"': fputs("\\\"", stdout); break;
		case '\\': fputs("\\\\", stdout); break;
		case '\b': fputs("\\b",  stdout); break;
		case '\f': fputs("\\f",  stdout); break;
		case '\n': fputs("\\n",  stdout); break;
		case '\r': fputs("\\r",  stdout); break;
		case '\t': fputs("\\t",  stdout); break;
		default:
			if (*p < 0x20)
				printf("\\u%04x", (unsigned)*p);
			else
				putchar(*p);
		}
	}
	putchar('"');
}

void
print_sys_info(const SystemInfo *info, const unsigned int indent_level)
{
	printf("%*s\"sys_info\": {\n", indent_level, "");
	printf("%*s\"cpu_info\": ", indent_level + 2, "");
	json_print_escaped(info->cpu_info);
	printf(",\n");
	printf("%*s\"ram_gb\": %.2f,\n", indent_level + 2, "", info->ram_gb);
	printf("%*s\"swap_gb\": %.2f\n", indent_level + 2, "", info->swap_gb);
	printf("%*s}", indent_level, "");
}

void
print_gpu_info(const CudaDeviceInfo *gpu_info, const unsigned int indent_level)
{
	printf("%*s\"gpu_info\": {\n", indent_level, "");
	printf("%*s\"available\": %s", indent_level + 2, "", gpu_info->available ? "true" : "false");
		
	if (gpu_info->available) {
		printf(",\n%*s\"name\": ", indent_level + 2, "");
		json_print_escaped(gpu_info->name);
		printf(",\n%*s\"cc\": \"%d.%d\",\n", indent_level + 2, "", gpu_info->cc_major, gpu_info->cc_minor);
		printf("%*s\"vram_gb\": %.2f\n", indent_level + 2, "", gpu_info->vram_gb);
	} else {
		printf("\n");
	}
	printf("%*s}", indent_level, "");
}

void
print_matrix_info(const MatrixInfo *info, const unsigned int indent_level)
{
	printf("%*s\"matrix_info\": {\n", indent_level, "");
	printf("%*s\"path\": ", indent_level + 2, "");
	json_print_escaped(info->path);
	printf(",\n");
	printf("%*s\"rows\": %u,\n", indent_level + 2, "", info->rows);
	printf("%*s\"cols\": %u,\n", indent_level + 2, "", info->cols);
	printf("%*s\"nnz\": %u,\n", indent_level + 2, "", info->nnz);
	printf("%*s\"load_time_s\": %.6f\n", indent_level + 2, "", info->load_time_s);
	printf("%*s}", indent_level, "");
}

void
print_benchmark_info(const BenchmarkInfo *info, const unsigned int indent_level)
{
	printf("%*s\"benchmark_info\": {\n", indent_level, "");
	printf("%*s\"trials\": %u,\n",  indent_level + 2, "", info->trials);
	printf("%*s\"wtrials\": %u,\n", indent_level + 2, "", info->wtrials);
	printf("%*s\"imptype\": %u\n",  indent_level + 2, "", info->imptype);
	printf("%*s}", indent_level, "");
}

static void
print_one_result(const Result *r, const unsigned int indent_level)
{
	printf("%*s{\n", indent_level, "");

	printf("%*s\"implementation_name\": ", indent_level + 2, "");
	json_print_escaped(r->name);
	printf(",\n");

	printf("%*s\"timestamp\": ", indent_level + 2, "");
	json_print_escaped(r->timestamp);
	printf(",\n");

	printf("%*s\"connected_components\": %u,\n", indent_level + 2, "", r->connected_components);
	printf("%*s\"stable_results\": %s,\n", indent_level + 2, "", r->stable_results ? "true" : "false");

	printf("%*s\"statistics\": {\n", indent_level + 2, "");
	printf("%*s\"mean_time_s\": %.6f,\n",   indent_level + 4, "", r->stats.mean_time_s);
	printf("%*s\"std_dev_s\": %.6f,\n",     indent_level + 4, "", r->stats.std_dev_s);
	printf("%*s\"median_time_s\": %.6f,\n", indent_level + 4, "", r->stats.median_time_s);
	printf("%*s\"min_time_s\": %.6f,\n",    indent_level + 4, "", r->stats.min_time_s);
	printf("%*s\"max_time_s\": %.6f,\n",    indent_level + 4, "", r->stats.max_time_s);
	printf("%*s\"total_time_s\": %.6f\n",   indent_level + 4, "", r->stats.total_time_s);
	printf("%*s},\n", indent_level + 2, "");

	printf("%*s\"throughput_edges_per_sec\": %.2f,\n", indent_level + 2, "", r->throughput_edges_per_sec);
	printf("%*s\"cpu_peak_rss_gb\": %.6f,\n", indent_level + 2, "", r->cpu_peak_rss_gb);
	printf("%*s\"gpu_peak_used_gb\": %.6f\n", indent_level + 2, "", r->gpu_peak_used_gb);

	printf("%*s}", indent_level, "");
}

void
print_results(const Result *results, const unsigned int indent_level, const unsigned int imptype)
{
	printf("%*s\"results\": [\n", indent_level, "");

	if (imptype == IMPL_ALL) {
		for (unsigned int i = 0; i < IMPL_ALL; i++) {
			print_one_result(&results[i], indent_level + 2);
			if (i + 1 < IMPL_ALL)
				printf(",\n");
			else
				printf("\n");
		}
	} else {
		print_one_result(&results[imptype], indent_level + 2);
		printf("\n");
	}

	printf("%*s]", indent_level, "");
}

