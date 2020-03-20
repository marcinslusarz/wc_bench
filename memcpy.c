// SPDX-License-Identifier: BSD-3-Clause
// Copyright 2020, Intel Corporation

#include <immintrin.h>

int max_batch_size = MAX_BATCH_SIZE;

#if defined(USE_SSE2)

const char *level = "SSE2";

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

static inline void
memmove_movnt2x64b(char *dest, const char *src)
{
	__m128i xmm0 = _mm_loadu_si128((__m128i *)src + 0);
	__m128i xmm1 = _mm_loadu_si128((__m128i *)src + 1);
	__m128i xmm2 = _mm_loadu_si128((__m128i *)src + 2);
	__m128i xmm3 = _mm_loadu_si128((__m128i *)src + 3);
	__m128i xmm4 = _mm_loadu_si128((__m128i *)src + 4);
	__m128i xmm5 = _mm_loadu_si128((__m128i *)src + 5);
	__m128i xmm6 = _mm_loadu_si128((__m128i *)src + 6);
	__m128i xmm7 = _mm_loadu_si128((__m128i *)src + 7);

	_mm_stream_si128((__m128i *)dest + 0, xmm0);
	_mm_stream_si128((__m128i *)dest + 1, xmm1);
	_mm_stream_si128((__m128i *)dest + 2, xmm2);
	_mm_stream_si128((__m128i *)dest + 3, xmm3);
	_mm_stream_si128((__m128i *)dest + 4, xmm4);
	_mm_stream_si128((__m128i *)dest + 5, xmm5);
	_mm_stream_si128((__m128i *)dest + 6, xmm6);
	_mm_stream_si128((__m128i *)dest + 7, xmm7);
}

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

#elif defined(USE_AVX)

const char *level = "AVX";

static inline void
memmove_movnt8x64b(char *dest, const char *src)
{
	__m256i ymm0 = _mm256_loadu_si256((__m256i *)src + 0);
	__m256i ymm1 = _mm256_loadu_si256((__m256i *)src + 1);
	__m256i ymm2 = _mm256_loadu_si256((__m256i *)src + 2);
	__m256i ymm3 = _mm256_loadu_si256((__m256i *)src + 3);
	__m256i ymm4 = _mm256_loadu_si256((__m256i *)src + 4);
	__m256i ymm5 = _mm256_loadu_si256((__m256i *)src + 5);
	__m256i ymm6 = _mm256_loadu_si256((__m256i *)src + 6);
	__m256i ymm7 = _mm256_loadu_si256((__m256i *)src + 7);
	__m256i ymm8 = _mm256_loadu_si256((__m256i *)src + 8);
	__m256i ymm9 = _mm256_loadu_si256((__m256i *)src + 9);
	__m256i ymm10 = _mm256_loadu_si256((__m256i *)src + 10);
	__m256i ymm11 = _mm256_loadu_si256((__m256i *)src + 11);
	__m256i ymm12 = _mm256_loadu_si256((__m256i *)src + 12);
	__m256i ymm13 = _mm256_loadu_si256((__m256i *)src + 13);
	__m256i ymm14 = _mm256_loadu_si256((__m256i *)src + 14);
	__m256i ymm15 = _mm256_loadu_si256((__m256i *)src + 15);

	_mm256_stream_si256((__m256i *)dest + 0, ymm0);
	_mm256_stream_si256((__m256i *)dest + 1, ymm1);
	_mm256_stream_si256((__m256i *)dest + 2, ymm2);
	_mm256_stream_si256((__m256i *)dest + 3, ymm3);
	_mm256_stream_si256((__m256i *)dest + 4, ymm4);
	_mm256_stream_si256((__m256i *)dest + 5, ymm5);
	_mm256_stream_si256((__m256i *)dest + 6, ymm6);
	_mm256_stream_si256((__m256i *)dest + 7, ymm7);
	_mm256_stream_si256((__m256i *)dest + 8, ymm8);
	_mm256_stream_si256((__m256i *)dest + 9, ymm9);
	_mm256_stream_si256((__m256i *)dest + 10, ymm10);
	_mm256_stream_si256((__m256i *)dest + 11, ymm11);
	_mm256_stream_si256((__m256i *)dest + 12, ymm12);
	_mm256_stream_si256((__m256i *)dest + 13, ymm13);
	_mm256_stream_si256((__m256i *)dest + 14, ymm14);
	_mm256_stream_si256((__m256i *)dest + 15, ymm15);
}

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
memmove_movnt2x64b(char *dest, const char *src)
{
	__m256i ymm0 = _mm256_loadu_si256((__m256i *)src + 0);
	__m256i ymm1 = _mm256_loadu_si256((__m256i *)src + 1);
	__m256i ymm2 = _mm256_loadu_si256((__m256i *)src + 2);
	__m256i ymm3 = _mm256_loadu_si256((__m256i *)src + 3);

	_mm256_stream_si256((__m256i *)dest + 0, ymm0);
	_mm256_stream_si256((__m256i *)dest + 1, ymm1);
	_mm256_stream_si256((__m256i *)dest + 2, ymm2);
	_mm256_stream_si256((__m256i *)dest + 3, ymm3);
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

const char *level = "AVX512F";

static inline void
memmove_movnt32x64b(char *dest, const char *src)
{
	__m512i zmm0 = _mm512_loadu_si512((__m512i *)src + 0);
	__m512i zmm1 = _mm512_loadu_si512((__m512i *)src + 1);
	__m512i zmm2 = _mm512_loadu_si512((__m512i *)src + 2);
	__m512i zmm3 = _mm512_loadu_si512((__m512i *)src + 3);
	__m512i zmm4 = _mm512_loadu_si512((__m512i *)src + 4);
	__m512i zmm5 = _mm512_loadu_si512((__m512i *)src + 5);
	__m512i zmm6 = _mm512_loadu_si512((__m512i *)src + 6);
	__m512i zmm7 = _mm512_loadu_si512((__m512i *)src + 7);
	__m512i zmm8 = _mm512_loadu_si512((__m512i *)src + 8);
	__m512i zmm9 = _mm512_loadu_si512((__m512i *)src + 9);
	__m512i zmm10 = _mm512_loadu_si512((__m512i *)src + 10);
	__m512i zmm11 = _mm512_loadu_si512((__m512i *)src + 11);
	__m512i zmm12 = _mm512_loadu_si512((__m512i *)src + 12);
	__m512i zmm13 = _mm512_loadu_si512((__m512i *)src + 13);
	__m512i zmm14 = _mm512_loadu_si512((__m512i *)src + 14);
	__m512i zmm15 = _mm512_loadu_si512((__m512i *)src + 15);
	__m512i zmm16 = _mm512_loadu_si512((__m512i *)src + 16);
	__m512i zmm17 = _mm512_loadu_si512((__m512i *)src + 17);
	__m512i zmm18 = _mm512_loadu_si512((__m512i *)src + 18);
	__m512i zmm19 = _mm512_loadu_si512((__m512i *)src + 19);
	__m512i zmm20 = _mm512_loadu_si512((__m512i *)src + 20);
	__m512i zmm21 = _mm512_loadu_si512((__m512i *)src + 21);
	__m512i zmm22 = _mm512_loadu_si512((__m512i *)src + 22);
	__m512i zmm23 = _mm512_loadu_si512((__m512i *)src + 23);
	__m512i zmm24 = _mm512_loadu_si512((__m512i *)src + 24);
	__m512i zmm25 = _mm512_loadu_si512((__m512i *)src + 25);
	__m512i zmm26 = _mm512_loadu_si512((__m512i *)src + 26);
	__m512i zmm27 = _mm512_loadu_si512((__m512i *)src + 27);
	__m512i zmm28 = _mm512_loadu_si512((__m512i *)src + 28);
	__m512i zmm29 = _mm512_loadu_si512((__m512i *)src + 29);
	__m512i zmm30 = _mm512_loadu_si512((__m512i *)src + 30);
	__m512i zmm31 = _mm512_loadu_si512((__m512i *)src + 31);

	_mm512_stream_si512((__m512i *)dest + 0, zmm0);
	_mm512_stream_si512((__m512i *)dest + 1, zmm1);
	_mm512_stream_si512((__m512i *)dest + 2, zmm2);
	_mm512_stream_si512((__m512i *)dest + 3, zmm3);
	_mm512_stream_si512((__m512i *)dest + 4, zmm4);
	_mm512_stream_si512((__m512i *)dest + 5, zmm5);
	_mm512_stream_si512((__m512i *)dest + 6, zmm6);
	_mm512_stream_si512((__m512i *)dest + 7, zmm7);
	_mm512_stream_si512((__m512i *)dest + 8, zmm8);
	_mm512_stream_si512((__m512i *)dest + 9, zmm9);
	_mm512_stream_si512((__m512i *)dest + 10, zmm10);
	_mm512_stream_si512((__m512i *)dest + 11, zmm11);
	_mm512_stream_si512((__m512i *)dest + 12, zmm12);
	_mm512_stream_si512((__m512i *)dest + 13, zmm13);
	_mm512_stream_si512((__m512i *)dest + 14, zmm14);
	_mm512_stream_si512((__m512i *)dest + 15, zmm15);
	_mm512_stream_si512((__m512i *)dest + 16, zmm16);
	_mm512_stream_si512((__m512i *)dest + 17, zmm17);
	_mm512_stream_si512((__m512i *)dest + 18, zmm18);
	_mm512_stream_si512((__m512i *)dest + 19, zmm19);
	_mm512_stream_si512((__m512i *)dest + 20, zmm20);
	_mm512_stream_si512((__m512i *)dest + 21, zmm21);
	_mm512_stream_si512((__m512i *)dest + 22, zmm22);
	_mm512_stream_si512((__m512i *)dest + 23, zmm23);
	_mm512_stream_si512((__m512i *)dest + 24, zmm24);
	_mm512_stream_si512((__m512i *)dest + 25, zmm25);
	_mm512_stream_si512((__m512i *)dest + 26, zmm26);
	_mm512_stream_si512((__m512i *)dest + 27, zmm27);
	_mm512_stream_si512((__m512i *)dest + 28, zmm28);
	_mm512_stream_si512((__m512i *)dest + 29, zmm29);
	_mm512_stream_si512((__m512i *)dest + 30, zmm30);
	_mm512_stream_si512((__m512i *)dest + 31, zmm31);
}

static inline void
memmove_movnt16x64b(char *dest, const char *src)
{
	__m512i zmm0 = _mm512_loadu_si512((__m512i *)src + 0);
	__m512i zmm1 = _mm512_loadu_si512((__m512i *)src + 1);
	__m512i zmm2 = _mm512_loadu_si512((__m512i *)src + 2);
	__m512i zmm3 = _mm512_loadu_si512((__m512i *)src + 3);
	__m512i zmm4 = _mm512_loadu_si512((__m512i *)src + 4);
	__m512i zmm5 = _mm512_loadu_si512((__m512i *)src + 5);
	__m512i zmm6 = _mm512_loadu_si512((__m512i *)src + 6);
	__m512i zmm7 = _mm512_loadu_si512((__m512i *)src + 7);
	__m512i zmm8 = _mm512_loadu_si512((__m512i *)src + 8);
	__m512i zmm9 = _mm512_loadu_si512((__m512i *)src + 9);
	__m512i zmm10 = _mm512_loadu_si512((__m512i *)src + 10);
	__m512i zmm11 = _mm512_loadu_si512((__m512i *)src + 11);
	__m512i zmm12 = _mm512_loadu_si512((__m512i *)src + 12);
	__m512i zmm13 = _mm512_loadu_si512((__m512i *)src + 13);
	__m512i zmm14 = _mm512_loadu_si512((__m512i *)src + 14);
	__m512i zmm15 = _mm512_loadu_si512((__m512i *)src + 15);

	_mm512_stream_si512((__m512i *)dest + 0, zmm0);
	_mm512_stream_si512((__m512i *)dest + 1, zmm1);
	_mm512_stream_si512((__m512i *)dest + 2, zmm2);
	_mm512_stream_si512((__m512i *)dest + 3, zmm3);
	_mm512_stream_si512((__m512i *)dest + 4, zmm4);
	_mm512_stream_si512((__m512i *)dest + 5, zmm5);
	_mm512_stream_si512((__m512i *)dest + 6, zmm6);
	_mm512_stream_si512((__m512i *)dest + 7, zmm7);
	_mm512_stream_si512((__m512i *)dest + 8, zmm8);
	_mm512_stream_si512((__m512i *)dest + 9, zmm9);
	_mm512_stream_si512((__m512i *)dest + 10, zmm10);
	_mm512_stream_si512((__m512i *)dest + 11, zmm11);
	_mm512_stream_si512((__m512i *)dest + 12, zmm12);
	_mm512_stream_si512((__m512i *)dest + 13, zmm13);
	_mm512_stream_si512((__m512i *)dest + 14, zmm14);
	_mm512_stream_si512((__m512i *)dest + 15, zmm15);
}

static inline void
memmove_movnt8x64b(char *dest, const char *src)
{
	__m512i zmm0 = _mm512_loadu_si512((__m512i *)src + 0);
	__m512i zmm1 = _mm512_loadu_si512((__m512i *)src + 1);
	__m512i zmm2 = _mm512_loadu_si512((__m512i *)src + 2);
	__m512i zmm3 = _mm512_loadu_si512((__m512i *)src + 3);
	__m512i zmm4 = _mm512_loadu_si512((__m512i *)src + 4);
	__m512i zmm5 = _mm512_loadu_si512((__m512i *)src + 5);
	__m512i zmm6 = _mm512_loadu_si512((__m512i *)src + 6);
	__m512i zmm7 = _mm512_loadu_si512((__m512i *)src + 7);

	_mm512_stream_si512((__m512i *)dest + 0, zmm0);
	_mm512_stream_si512((__m512i *)dest + 1, zmm1);
	_mm512_stream_si512((__m512i *)dest + 2, zmm2);
	_mm512_stream_si512((__m512i *)dest + 3, zmm3);
	_mm512_stream_si512((__m512i *)dest + 4, zmm4);
	_mm512_stream_si512((__m512i *)dest + 5, zmm5);
	_mm512_stream_si512((__m512i *)dest + 6, zmm6);
	_mm512_stream_si512((__m512i *)dest + 7, zmm7);
}

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
memmove_movnt2x64b(char *dest, const char *src)
{
	__m512i zmm0 = _mm512_loadu_si512((__m512i *)src + 0);
	__m512i zmm1 = _mm512_loadu_si512((__m512i *)src + 1);

	_mm512_stream_si512((__m512i *)dest + 0, zmm0);
	_mm512_stream_si512((__m512i *)dest + 1, zmm1);
}

static inline void
memmove_movnt1x64b(char *dest, const char *src)
{
	__m512i zmm0 = _mm512_loadu_si512((__m512i *)src + 0);

	_mm512_stream_si512((__m512i *)dest + 0, zmm0);
}

#else
#error set USE_SSE2 or USE_AVX or USE_AVX512F
const char *level = "?";
void memmove_movnt1x64b(char *dest, const char *src);
#endif

void
wc_memcpy(char *dest, const char *src, size_t sz)
{
#if defined(USE_AVX512F) && MAX_BATCH_SIZE >= 2048
	while (sz >= 2048) {
		memmove_movnt32x64b(dest, src);
		dest += 2048;
		src += 2048;
		sz -= 2048;
	}
#endif

#if defined(USE_AVX512F) && MAX_BATCH_SIZE >= 1024
	while (sz >= 1024) {
		memmove_movnt16x64b(dest, src);
		dest += 1024;
		src += 1024;
		sz -= 1024;
	}
#endif

#if (defined(USE_AVX) || defined(USE_AVX512F)) && MAX_BATCH_SIZE >= 512
	while (sz >= 512) {
		memmove_movnt8x64b(dest, src);
		dest += 512;
		src += 512;
		sz -= 512;
	}
#endif

#if MAX_BATCH_SIZE >= 256
	while (sz >= 256) {
		memmove_movnt4x64b(dest, src);
		dest += 256;
		src += 256;
		sz -= 256;
	}
#endif

#if MAX_BATCH_SIZE >= 128
	while (sz >= 128) {
		memmove_movnt2x64b(dest, src);
		dest += 128;
		src += 128;
		sz -= 128;
	}
#endif

	while (sz >= 64) {
		memmove_movnt1x64b(dest, src);
		dest += 64;
		src += 64;
		sz -= 64;
	}
#if defined(USE_AVX) || defined(USE_AVX512F)
	_mm256_zeroupper();
#endif

	_mm_sfence();
}
