// SPDX-License-Identifier: BSD-3-Clause
// Copyright 2020, Intel Corporation

#include <libpmem.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "memcpy.h"

#define MEGA (1ULL << 20)
#define GIGA (1ULL << 30)
#define NSEC_IN_SEC (1000ULL * 1000 * 1000)

#define LOOPS 10
#define PMEM_SIZE (4 * GIGA)
#define DRAM_SIZE (100 * MEGA)

static void
measure(char *pmem, size_t pmemlen, size_t cpy_len, char *dram, bool csv)
{
	char *dest = pmem;
	char *dest_end = dest + pmemlen - cpy_len;
	char *src = dram;
	char *src_end = src + DRAM_SIZE - cpy_len;

	if (csv) {
		printf("%s,%0.3f,%zu,", mode, 1.0 * pmemlen / GIGA, cpy_len);
	} else {
		printf("mode:      %s\n", mode);
		printf("file size: %0.3f GiB\n", 1.0 * pmemlen / GIGA);
		printf("copy len:  %zu B\n", cpy_len);
	}

	struct timespec start, end;
	if (clock_gettime(CLOCK_REALTIME, &start)) {
		perror("clock_gettime");
		exit(1);
	}

	size_t copied = 0;
	for (unsigned i = 0; i < LOOPS; ++i) {
		while (dest < dest_end) {
			wc_memcpy(dest, src, cpy_len);

			dest += cpy_len;
			src += cpy_len;
			copied += cpy_len;

			if (src >= src_end)
				src = dram;
		}
		dest = pmem;
	}

	if (clock_gettime(CLOCK_REALTIME, &end)) {
		perror("clock_gettime");
		exit(1);
	}

	unsigned long long tm = (end.tv_sec - start.tv_sec) * NSEC_IN_SEC +
			end.tv_nsec - start.tv_nsec;

	if (csv) {
		printf("%.3f,", 1.0 * tm / NSEC_IN_SEC);
		printf("%.3f,", 1.0 * copied / GIGA);
		printf("%0.3f\n", 1.0 * NSEC_IN_SEC * copied / GIGA / tm);
	} else {
		printf("time:      %.3f s\n", 1.0 * tm / NSEC_IN_SEC);
		printf("copied:    %.3f GiB\n", 1.0 * copied / GIGA);
		printf("bw:        %0.3f GiB/s\n", 1.0 * NSEC_IN_SEC * copied / GIGA / tm);
	}
}

int
main(int argc, char *argv[])
{
	char *pmem;
	size_t pmemlen;
	int is_pmem;

	if (argc < 3) {
		fprintf(stderr,
			"Usage: %s path memcpy_size_in_cls [max_memcpy_size_in_cls]\n",
			argv[0]);
		exit(1);
	}

	pmem = pmem_map_file(argv[1], PMEM_SIZE, PMEM_FILE_CREATE, 0644,
			&pmemlen, &is_pmem);
	if (pmem == NULL) {
		fprintf(stderr, "pmem_map_file: %s\n", pmem_errormsg());
		exit(1);
	}
	if (!is_pmem) {
		fprintf(stderr, "%s is not pmem\n", argv[1]);
		exit(1);
	}

	char *dram = malloc(DRAM_SIZE);
	if (!dram) {
		perror("malloc");
		exit(1);
	}

	for (size_t i = 0; i < DRAM_SIZE; ++i)
		dram[i] = i & 0xff;

	/* prefault */
	memset(pmem, 0, pmemlen);

	size_t cpy_len = atoi(argv[2]) * 64;
	size_t max_cpy_len = cpy_len;

	if (argc > 3)
		max_cpy_len = atoi(argv[3]) * 64;

	bool csv = cpy_len != max_cpy_len;

	if (csv)
		printf("mode,file_size[GiB],copy_len[B],time[s],copied[GiB],bw[GiB/s]\n");

	while (cpy_len <= max_cpy_len) {
		measure(pmem, pmemlen, cpy_len, dram, csv);
		cpy_len += 64;
		if (!csv)
			printf("--\n");
	}

//	pmem_msync(pmem, pmemlen);
	pmem_unmap(pmem, pmemlen);
	free(dram);

	return 0;
}
