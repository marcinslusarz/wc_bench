// SPDX-License-Identifier: BSD-3-Clause
// Copyright 2020, Intel Corporation

#include <immintrin.h>
#include <libpmem.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define MEGA (1ULL << 20)
#define GIGA (1ULL << 30)
#define NSEC_IN_SEC (1000ULL * 1000 * 1000)

#define LOOPS 10
#define PMEM_SIZE (4 * GIGA)
#define DRAM_SIZE (100 * MEGA)

#if defined(USE_SSE2)

static const char mode[] = "SSE2";

static inline void
memmove_movnt1x64b(char *dest, const char *src)
{
	__m128i xmm0 = _mm_loadu_si128((__m128i *)src + 0);
	__m128i xmm1 = _mm_loadu_si128((__m128i *)src + 1);
	__m128i xmm2 = _mm_loadu_si128((__m128i *)src + 2);
	__m128i xmm3 = _mm_loadu_si128((__m128i *)src + 3);

	_mm_stream_si128((__m128i *)dest + 0, xmm0);
	_mm_stream_si128((__m128i *)dest + 1, xmm1);
	_mm_stream_si128((__m128i *)dest + 2, xmm2);
	_mm_stream_si128((__m128i *)dest + 3, xmm3);
}

static inline void
memmove_movnt4x64b(char *dest, const char *src)
{
	__m128i xmm0 = _mm_loadu_si128((__m128i *)src + 0);
	__m128i xmm1 = _mm_loadu_si128((__m128i *)src + 1);
	__m128i xmm2 = _mm_loadu_si128((__m128i *)src + 2);
	__m128i xmm3 = _mm_loadu_si128((__m128i *)src + 3);
	__m128i xmm4 = _mm_loadu_si128((__m128i *)src + 4);
	__m128i xmm5 = _mm_loadu_si128((__m128i *)src + 5);
	__m128i xmm6 = _mm_loadu_si128((__m128i *)src + 6);
	__m128i xmm7 = _mm_loadu_si128((__m128i *)src + 7);
	__m128i xmm8 = _mm_loadu_si128((__m128i *)src + 8);
	__m128i xmm9 = _mm_loadu_si128((__m128i *)src + 9);
	__m128i xmm10 = _mm_loadu_si128((__m128i *)src + 10);
	__m128i xmm11 = _mm_loadu_si128((__m128i *)src + 11);
	__m128i xmm12 = _mm_loadu_si128((__m128i *)src + 12);
	__m128i xmm13 = _mm_loadu_si128((__m128i *)src + 13);
	__m128i xmm14 = _mm_loadu_si128((__m128i *)src + 14);
	__m128i xmm15 = _mm_loadu_si128((__m128i *)src + 15);

	_mm_stream_si128((__m128i *)dest + 0, xmm0);
	_mm_stream_si128((__m128i *)dest + 1, xmm1);
	_mm_stream_si128((__m128i *)dest + 2, xmm2);
	_mm_stream_si128((__m128i *)dest + 3, xmm3);
	_mm_stream_si128((__m128i *)dest + 4, xmm4);
	_mm_stream_si128((__m128i *)dest + 5, xmm5);
	_mm_stream_si128((__m128i *)dest + 6, xmm6);
	_mm_stream_si128((__m128i *)dest + 7, xmm7);
	_mm_stream_si128((__m128i *)dest + 8, xmm8);
	_mm_stream_si128((__m128i *)dest + 9, xmm9);
	_mm_stream_si128((__m128i *)dest + 10, xmm10);
	_mm_stream_si128((__m128i *)dest + 11, xmm11);
	_mm_stream_si128((__m128i *)dest + 12, xmm12);
	_mm_stream_si128((__m128i *)dest + 13, xmm13);
	_mm_stream_si128((__m128i *)dest + 14, xmm14);
	_mm_stream_si128((__m128i *)dest + 15, xmm15);
}

#elif defined(USE_AVX)

static const char mode[] = "AVX";

static inline void
memmove_movnt4x64b(char *dest, const char *src)
{
	__m256i ymm0 = _mm256_loadu_si256((__m256i *)src + 0);
	__m256i ymm1 = _mm256_loadu_si256((__m256i *)src + 1);
	__m256i ymm2 = _mm256_loadu_si256((__m256i *)src + 2);
	__m256i ymm3 = _mm256_loadu_si256((__m256i *)src + 3);
	__m256i ymm4 = _mm256_loadu_si256((__m256i *)src + 4);
	__m256i ymm5 = _mm256_loadu_si256((__m256i *)src + 5);
	__m256i ymm6 = _mm256_loadu_si256((__m256i *)src + 6);
	__m256i ymm7 = _mm256_loadu_si256((__m256i *)src + 7);

	_mm256_stream_si256((__m256i *)dest + 0, ymm0);
	_mm256_stream_si256((__m256i *)dest + 1, ymm1);
	_mm256_stream_si256((__m256i *)dest + 2, ymm2);
	_mm256_stream_si256((__m256i *)dest + 3, ymm3);
	_mm256_stream_si256((__m256i *)dest + 4, ymm4);
	_mm256_stream_si256((__m256i *)dest + 5, ymm5);
	_mm256_stream_si256((__m256i *)dest + 6, ymm6);
	_mm256_stream_si256((__m256i *)dest + 7, ymm7);
}

static inline void
memmove_movnt1x64b(char *dest, const char *src)
{
	__m256i ymm0 = _mm256_loadu_si256((__m256i *)src + 0);
	__m256i ymm1 = _mm256_loadu_si256((__m256i *)src + 1);

	_mm256_stream_si256((__m256i *)dest + 0, ymm0);
	_mm256_stream_si256((__m256i *)dest + 1, ymm1);
}


#elif defined(USE_AVX512F)

static const char mode[] = "AVX512F";

static inline void
memmove_movnt4x64b(char *dest, const char *src)
{
	__m512i zmm0 = _mm512_loadu_si512((__m512i *)src + 0);
	__m512i zmm1 = _mm512_loadu_si512((__m512i *)src + 1);
	__m512i zmm2 = _mm512_loadu_si512((__m512i *)src + 2);
	__m512i zmm3 = _mm512_loadu_si512((__m512i *)src + 3);

	_mm512_stream_si512((__m512i *)dest + 0, zmm0);
	_mm512_stream_si512((__m512i *)dest + 1, zmm1);
	_mm512_stream_si512((__m512i *)dest + 2, zmm2);
	_mm512_stream_si512((__m512i *)dest + 3, zmm3);
}

static inline void
memmove_movnt1x64b(char *dest, const char *src)
{
	__m512i zmm0 = _mm512_loadu_si512((__m512i *)src + 0);

	_mm512_stream_si512((__m512i *)dest + 0, zmm0);
}

#else
#error set USE_SSE2 or USE_AVX or USE_AVX512F
static const char mode[] = "?";
void memmove_movnt4x64b(char *dest, const char *src);
void memmove_movnt1x64b(char *dest, const char *src);
#endif

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
			size_t sz = cpy_len;
			while (sz >= 256) {
				memmove_movnt4x64b(dest, src);
				dest += 256;
				src += 256;
				sz -= 256;
			}

			while (sz >= 64) {
				memmove_movnt1x64b(dest, src);
				dest += 64;
				src += 64;
				sz -= 64;
			}

			_mm_sfence();
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
		fprintf(stderr, "not enough args\n");
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
