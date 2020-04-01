// SPDX-License-Identifier: BSD-3-Clause
// Copyright 2020, Intel Corporation

#include <immintrin.h>

int max_batch_size = MAX_BATCH_SIZE;

#if defined(USE_SSE2)

const char *level = "SSE2";

static inline void
memmove_movnt4x64b(char *dest, const char *src)
{
#ifdef USE_ASM
	asm(
		"movdqu 0x00(%1), %%xmm0\n"
		"movdqu 0x10(%1), %%xmm1\n"
		"movdqu 0x20(%1), %%xmm2\n"
		"movdqu 0x30(%1), %%xmm3\n"
		"movdqu 0x40(%1), %%xmm4\n"
		"movdqu 0x50(%1), %%xmm5\n"
		"movdqu 0x60(%1), %%xmm6\n"
		"movdqu 0x70(%1), %%xmm7\n"
		"movdqu 0x80(%1), %%xmm8\n"
		"movdqu 0x90(%1), %%xmm9\n"
		"movdqu 0xa0(%1), %%xmm10\n"
		"movdqu 0xb0(%1), %%xmm11\n"
		"movdqu 0xc0(%1), %%xmm12\n"
		"movdqu 0xd0(%1), %%xmm13\n"
		"movdqu 0xe0(%1), %%xmm14\n"
		"movdqu 0xf0(%1), %%xmm15\n"
		"movntdq %%xmm0, 0x00(%0)\n"
		"movntdq %%xmm1, 0x10(%0)\n"
		"movntdq %%xmm2, 0x20(%0)\n"
		"movntdq %%xmm3, 0x30(%0)\n"
		"movntdq %%xmm4, 0x40(%0)\n"
		"movntdq %%xmm5, 0x50(%0)\n"
		"movntdq %%xmm6, 0x60(%0)\n"
		"movntdq %%xmm7, 0x70(%0)\n"
		"movntdq %%xmm8, 0x80(%0)\n"
		"movntdq %%xmm9, 0x90(%0)\n"
		"movntdq %%xmm10, 0xa0(%0)\n"
		"movntdq %%xmm11, 0xb0(%0)\n"
		"movntdq %%xmm12, 0xc0(%0)\n"
		"movntdq %%xmm13, 0xd0(%0)\n"
		"movntdq %%xmm14, 0xe0(%0)\n"
		"movntdq %%xmm15, 0xf0(%0)\n"
		:
		: "r"(dest), "r"(src)
		: "memory", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5",
		  "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12",
		  "xmm13", "xmm14", "xmm15"
	);
#else
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
#endif
}

static inline void
memmove_movnt2x64b(char *dest, const char *src)
{
#ifdef USE_ASM
	asm(
		"movdqu 0x00(%1), %%xmm0\n"
		"movdqu 0x10(%1), %%xmm1\n"
		"movdqu 0x20(%1), %%xmm2\n"
		"movdqu 0x30(%1), %%xmm3\n"
		"movdqu 0x40(%1), %%xmm4\n"
		"movdqu 0x50(%1), %%xmm5\n"
		"movdqu 0x60(%1), %%xmm6\n"
		"movdqu 0x70(%1), %%xmm7\n"
		"movntdq %%xmm0, 0x00(%0)\n"
		"movntdq %%xmm1, 0x10(%0)\n"
		"movntdq %%xmm2, 0x20(%0)\n"
		"movntdq %%xmm3, 0x30(%0)\n"
		"movntdq %%xmm4, 0x40(%0)\n"
		"movntdq %%xmm5, 0x50(%0)\n"
		"movntdq %%xmm6, 0x60(%0)\n"
		"movntdq %%xmm7, 0x70(%0)\n"
		:
		: "r"(dest), "r"(src)
		: "memory", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5",
		  "xmm6", "xmm7"
	);
#else
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
#endif
}

static inline void
memmove_movnt1x64b(char *dest, const char *src)
{
#ifdef USE_ASM
	asm(
		"movdqu 0x00(%1), %%xmm0\n"
		"movdqu 0x10(%1), %%xmm1\n"
		"movdqu 0x20(%1), %%xmm2\n"
		"movdqu 0x30(%1), %%xmm3\n"
		"movntdq %%xmm0, 0x00(%0)\n"
		"movntdq %%xmm1, 0x10(%0)\n"
		"movntdq %%xmm2, 0x20(%0)\n"
		"movntdq %%xmm3, 0x30(%0)\n"
		:
		: "r"(dest), "r"(src)
		: "memory", "xmm0", "xmm1", "xmm2", "xmm3"
	);
#else
	__m128i xmm0 = _mm_loadu_si128((__m128i *)src + 0);
	__m128i xmm1 = _mm_loadu_si128((__m128i *)src + 1);
	__m128i xmm2 = _mm_loadu_si128((__m128i *)src + 2);
	__m128i xmm3 = _mm_loadu_si128((__m128i *)src + 3);

	_mm_stream_si128((__m128i *)dest + 0, xmm0);
	_mm_stream_si128((__m128i *)dest + 1, xmm1);
	_mm_stream_si128((__m128i *)dest + 2, xmm2);
	_mm_stream_si128((__m128i *)dest + 3, xmm3);
#endif
}

#elif defined(USE_AVX)

const char *level = "AVX";

static inline void
memmove_movnt8x64b(char *dest, const char *src)
{
#ifdef USE_ASM
	asm(
		"vmovdqu 0x00(%1), %%ymm0\n"
		"vmovdqu 0x20(%1), %%ymm1\n"
		"vmovdqu 0x40(%1), %%ymm2\n"
		"vmovdqu 0x60(%1), %%ymm3\n"
		"vmovdqu 0x80(%1), %%ymm4\n"
		"vmovdqu 0xa0(%1), %%ymm5\n"
		"vmovdqu 0xc0(%1), %%ymm6\n"
		"vmovdqu 0xe0(%1), %%ymm7\n"
		"vmovdqu 0x100(%1), %%ymm8\n"
		"vmovdqu 0x120(%1), %%ymm9\n"
		"vmovdqu 0x140(%1), %%ymm10\n"
		"vmovdqu 0x160(%1), %%ymm11\n"
		"vmovdqu 0x180(%1), %%ymm12\n"
		"vmovdqu 0x1a0(%1), %%ymm13\n"
		"vmovdqu 0x1c0(%1), %%ymm14\n"
		"vmovdqu 0x1e0(%1), %%ymm15\n"
		"vmovntdq %%ymm0, 0x00(%0)\n"
		"vmovntdq %%ymm1, 0x20(%0)\n"
		"vmovntdq %%ymm2, 0x40(%0)\n"
		"vmovntdq %%ymm3, 0x60(%0)\n"
		"vmovntdq %%ymm4, 0x80(%0)\n"
		"vmovntdq %%ymm5, 0xa0(%0)\n"
		"vmovntdq %%ymm6, 0xc0(%0)\n"
		"vmovntdq %%ymm7, 0xe0(%0)\n"
		"vmovntdq %%ymm8, 0x100(%0)\n"
		"vmovntdq %%ymm9, 0x120(%0)\n"
		"vmovntdq %%ymm10, 0x140(%0)\n"
		"vmovntdq %%ymm11, 0x160(%0)\n"
		"vmovntdq %%ymm12, 0x180(%0)\n"
		"vmovntdq %%ymm13, 0x1a0(%0)\n"
		"vmovntdq %%ymm14, 0x1c0(%0)\n"
		"vmovntdq %%ymm15, 0x1e0(%0)\n"
		:
		: "r"(dest), "r"(src)
		: "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
		  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12",
		  "ymm13", "ymm14", "ymm15"
	);
#else
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
#endif
}

static inline void
memmove_movnt4x64b(char *dest, const char *src)
{
#ifdef USE_ASM
	asm(
		"vmovdqu 0x00(%1), %%ymm0\n"
		"vmovdqu 0x20(%1), %%ymm1\n"
		"vmovdqu 0x40(%1), %%ymm2\n"
		"vmovdqu 0x60(%1), %%ymm3\n"
		"vmovdqu 0x80(%1), %%ymm4\n"
		"vmovdqu 0xa0(%1), %%ymm5\n"
		"vmovdqu 0xc0(%1), %%ymm6\n"
		"vmovdqu 0xe0(%1), %%ymm7\n"
		"vmovntdq %%ymm0, 0x00(%0)\n"
		"vmovntdq %%ymm1, 0x20(%0)\n"
		"vmovntdq %%ymm2, 0x40(%0)\n"
		"vmovntdq %%ymm3, 0x60(%0)\n"
		"vmovntdq %%ymm4, 0x80(%0)\n"
		"vmovntdq %%ymm5, 0xa0(%0)\n"
		"vmovntdq %%ymm6, 0xc0(%0)\n"
		"vmovntdq %%ymm7, 0xe0(%0)\n"
		:
		: "r"(dest), "r"(src)
		: "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
		  "ymm6", "ymm7"
	);
#else
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
#endif
}

static inline void
memmove_movnt2x64b(char *dest, const char *src)
{
#ifdef USE_ASM
	asm(
		"vmovdqu 0x00(%1), %%ymm0\n"
		"vmovdqu 0x20(%1), %%ymm1\n"
		"vmovdqu 0x40(%1), %%ymm2\n"
		"vmovdqu 0x60(%1), %%ymm3\n"
		"vmovntdq %%ymm0, 0x00(%0)\n"
		"vmovntdq %%ymm1, 0x20(%0)\n"
		"vmovntdq %%ymm2, 0x40(%0)\n"
		"vmovntdq %%ymm3, 0x60(%0)\n"
		:
		: "r"(dest), "r"(src)
		: "memory", "ymm0", "ymm1", "ymm2", "ymm3"
	);
#else
	__m256i ymm0 = _mm256_loadu_si256((__m256i *)src + 0);
	__m256i ymm1 = _mm256_loadu_si256((__m256i *)src + 1);
	__m256i ymm2 = _mm256_loadu_si256((__m256i *)src + 2);
	__m256i ymm3 = _mm256_loadu_si256((__m256i *)src + 3);

	_mm256_stream_si256((__m256i *)dest + 0, ymm0);
	_mm256_stream_si256((__m256i *)dest + 1, ymm1);
	_mm256_stream_si256((__m256i *)dest + 2, ymm2);
	_mm256_stream_si256((__m256i *)dest + 3, ymm3);
#endif
}

static inline void
memmove_movnt1x64b(char *dest, const char *src)
{
#ifdef USE_ASM
	asm(
		"vmovdqu 0x00(%1), %%ymm0\n"
		"vmovdqu 0x20(%1), %%ymm1\n"
		"vmovntdq %%ymm0, 0x00(%0)\n"
		"vmovntdq %%ymm1, 0x20(%0)\n"
		:
		: "r"(dest), "r"(src)
		: "memory", "ymm0", "ymm1"
	);
#else
	__m256i ymm0 = _mm256_loadu_si256((__m256i *)src + 0);
	__m256i ymm1 = _mm256_loadu_si256((__m256i *)src + 1);

	_mm256_stream_si256((__m256i *)dest + 0, ymm0);
	_mm256_stream_si256((__m256i *)dest + 1, ymm1);
#endif
}


#elif defined(USE_AVX512F)

const char *level = "AVX512F";

static inline void
memmove_movnt32x64b(char *dest, const char *src)
{
#ifdef USE_ASM
	asm(
		"vmovdqu64 0x000(%1), %%zmm0\n"
		"vmovdqu64 0x040(%1), %%zmm1\n"
		"vmovdqu64 0x080(%1), %%zmm2\n"
		"vmovdqu64 0x0c0(%1), %%zmm3\n"
		"vmovdqu64 0x100(%1), %%zmm4\n"
		"vmovdqu64 0x140(%1), %%zmm5\n"
		"vmovdqu64 0x180(%1), %%zmm6\n"
		"vmovdqu64 0x1c0(%1), %%zmm7\n"
		"vmovdqu64 0x200(%1), %%zmm8\n"
		"vmovdqu64 0x240(%1), %%zmm9\n"
		"vmovdqu64 0x280(%1), %%zmm10\n"
		"vmovdqu64 0x2c0(%1), %%zmm11\n"
		"vmovdqu64 0x300(%1), %%zmm12\n"
		"vmovdqu64 0x340(%1), %%zmm13\n"
		"vmovdqu64 0x380(%1), %%zmm14\n"
		"vmovdqu64 0x3c0(%1), %%zmm15\n"
		"vmovdqu64 0x400(%1), %%zmm16\n"
		"vmovdqu64 0x440(%1), %%zmm17\n"
		"vmovdqu64 0x480(%1), %%zmm18\n"
		"vmovdqu64 0x4c0(%1), %%zmm19\n"
		"vmovdqu64 0x500(%1), %%zmm20\n"
		"vmovdqu64 0x540(%1), %%zmm21\n"
		"vmovdqu64 0x580(%1), %%zmm22\n"
		"vmovdqu64 0x5c0(%1), %%zmm23\n"
		"vmovdqu64 0x600(%1), %%zmm24\n"
		"vmovdqu64 0x640(%1), %%zmm25\n"
		"vmovdqu64 0x680(%1), %%zmm26\n"
		"vmovdqu64 0x6c0(%1), %%zmm27\n"
		"vmovdqu64 0x700(%1), %%zmm28\n"
		"vmovdqu64 0x740(%1), %%zmm29\n"
		"vmovdqu64 0x780(%1), %%zmm30\n"
		"vmovdqu64 0x7c0(%1), %%zmm31\n"
		"vmovntdq %%zmm0,  0x000(%0)\n"
		"vmovntdq %%zmm1,  0x040(%0)\n"
		"vmovntdq %%zmm2,  0x080(%0)\n"
		"vmovntdq %%zmm3,  0x0c0(%0)\n"
		"vmovntdq %%zmm4,  0x100(%0)\n"
		"vmovntdq %%zmm5,  0x140(%0)\n"
		"vmovntdq %%zmm6,  0x180(%0)\n"
		"vmovntdq %%zmm7,  0x1c0(%0)\n"
		"vmovntdq %%zmm8,  0x200(%0)\n"
		"vmovntdq %%zmm9,  0x240(%0)\n"
		"vmovntdq %%zmm10, 0x280(%0)\n"
		"vmovntdq %%zmm11, 0x2c0(%0)\n"
		"vmovntdq %%zmm12, 0x300(%0)\n"
		"vmovntdq %%zmm13, 0x340(%0)\n"
		"vmovntdq %%zmm14, 0x380(%0)\n"
		"vmovntdq %%zmm15, 0x3c0(%0)\n"
		"vmovntdq %%zmm16, 0x400(%0)\n"
		"vmovntdq %%zmm17, 0x440(%0)\n"
		"vmovntdq %%zmm18, 0x480(%0)\n"
		"vmovntdq %%zmm19, 0x4c0(%0)\n"
		"vmovntdq %%zmm20, 0x500(%0)\n"
		"vmovntdq %%zmm21, 0x540(%0)\n"
		"vmovntdq %%zmm22, 0x580(%0)\n"
		"vmovntdq %%zmm23, 0x5c0(%0)\n"
		"vmovntdq %%zmm24, 0x600(%0)\n"
		"vmovntdq %%zmm25, 0x640(%0)\n"
		"vmovntdq %%zmm26, 0x680(%0)\n"
		"vmovntdq %%zmm27, 0x6c0(%0)\n"
		"vmovntdq %%zmm28, 0x700(%0)\n"
		"vmovntdq %%zmm29, 0x740(%0)\n"
		"vmovntdq %%zmm30, 0x780(%0)\n"
		"vmovntdq %%zmm31, 0x7c0(%0)\n"
		:
		: "r"(dest), "r"(src)
		: "memory", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
		  "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12",
		  "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19",
		  "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
		  "zmm27", "zmm28", "zmm29", "zmm30", "zmm31"
	);
#else
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
#endif
}

static inline void
memmove_movnt16x64b(char *dest, const char *src)
{
#ifdef USE_ASM
	asm(
		"vmovdqu64 0x000(%1), %%zmm0\n"
		"vmovdqu64 0x040(%1), %%zmm1\n"
		"vmovdqu64 0x080(%1), %%zmm2\n"
		"vmovdqu64 0x0c0(%1), %%zmm3\n"
		"vmovdqu64 0x100(%1), %%zmm4\n"
		"vmovdqu64 0x140(%1), %%zmm5\n"
		"vmovdqu64 0x180(%1), %%zmm6\n"
		"vmovdqu64 0x1c0(%1), %%zmm7\n"
		"vmovdqu64 0x200(%1), %%zmm8\n"
		"vmovdqu64 0x240(%1), %%zmm9\n"
		"vmovdqu64 0x280(%1), %%zmm10\n"
		"vmovdqu64 0x2c0(%1), %%zmm11\n"
		"vmovdqu64 0x300(%1), %%zmm12\n"
		"vmovdqu64 0x340(%1), %%zmm13\n"
		"vmovdqu64 0x380(%1), %%zmm14\n"
		"vmovdqu64 0x3c0(%1), %%zmm15\n"
		"vmovntdq %%zmm0,  0x000(%0)\n"
		"vmovntdq %%zmm1,  0x040(%0)\n"
		"vmovntdq %%zmm2,  0x080(%0)\n"
		"vmovntdq %%zmm3,  0x0c0(%0)\n"
		"vmovntdq %%zmm4,  0x100(%0)\n"
		"vmovntdq %%zmm5,  0x140(%0)\n"
		"vmovntdq %%zmm6,  0x180(%0)\n"
		"vmovntdq %%zmm7,  0x1c0(%0)\n"
		"vmovntdq %%zmm8,  0x200(%0)\n"
		"vmovntdq %%zmm9,  0x240(%0)\n"
		"vmovntdq %%zmm10, 0x280(%0)\n"
		"vmovntdq %%zmm11, 0x2c0(%0)\n"
		"vmovntdq %%zmm12, 0x300(%0)\n"
		"vmovntdq %%zmm13, 0x340(%0)\n"
		"vmovntdq %%zmm14, 0x380(%0)\n"
		"vmovntdq %%zmm15, 0x3c0(%0)\n"
		:
		: "r"(dest), "r"(src)
		: "memory", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
		  "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12",
		  "zmm13", "zmm14", "zmm15"
	);
#else
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
#endif
}

static inline void
memmove_movnt8x64b(char *dest, const char *src)
{
#ifdef USE_ASM
	asm(
		"vmovdqu64 0x000(%1), %%zmm0\n"
		"vmovdqu64 0x040(%1), %%zmm1\n"
		"vmovdqu64 0x080(%1), %%zmm2\n"
		"vmovdqu64 0x0c0(%1), %%zmm3\n"
		"vmovdqu64 0x100(%1), %%zmm4\n"
		"vmovdqu64 0x140(%1), %%zmm5\n"
		"vmovdqu64 0x180(%1), %%zmm6\n"
		"vmovdqu64 0x1c0(%1), %%zmm7\n"
		"vmovntdq %%zmm0, 0x000(%0)\n"
		"vmovntdq %%zmm1, 0x040(%0)\n"
		"vmovntdq %%zmm2, 0x080(%0)\n"
		"vmovntdq %%zmm3, 0x0c0(%0)\n"
		"vmovntdq %%zmm4, 0x100(%0)\n"
		"vmovntdq %%zmm5, 0x140(%0)\n"
		"vmovntdq %%zmm6, 0x180(%0)\n"
		"vmovntdq %%zmm7, 0x1c0(%0)\n"
		:
		: "r"(dest), "r"(src)
		: "memory", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
		  "zmm6", "zmm7"
	);
#else
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
#endif
}

static inline void
memmove_movnt4x64b(char *dest, const char *src)
{
#ifdef USE_ASM
	asm(
		"vmovdqu64 0x00(%1), %%zmm0\n"
		"vmovdqu64 0x40(%1), %%zmm1\n"
		"vmovdqu64 0x80(%1), %%zmm2\n"
		"vmovdqu64 0xc0(%1), %%zmm3\n"
		"vmovntdq %%zmm0, 0x00(%0)\n"
		"vmovntdq %%zmm1, 0x40(%0)\n"
		"vmovntdq %%zmm2, 0x80(%0)\n"
		"vmovntdq %%zmm3, 0xc0(%0)\n"
		:
		: "r"(dest), "r"(src)
		: "memory", "zmm0", "zmm1", "zmm2", "zmm3"
	);
#else
	__m512i zmm0 = _mm512_loadu_si512((__m512i *)src + 0);
	__m512i zmm1 = _mm512_loadu_si512((__m512i *)src + 1);
	__m512i zmm2 = _mm512_loadu_si512((__m512i *)src + 2);
	__m512i zmm3 = _mm512_loadu_si512((__m512i *)src + 3);

	_mm512_stream_si512((__m512i *)dest + 0, zmm0);
	_mm512_stream_si512((__m512i *)dest + 1, zmm1);
	_mm512_stream_si512((__m512i *)dest + 2, zmm2);
	_mm512_stream_si512((__m512i *)dest + 3, zmm3);
#endif
}

static inline void
memmove_movnt2x64b(char *dest, const char *src)
{
#ifdef USE_ASM
	asm(
		"vmovdqu64 0x00(%1), %%zmm0\n"
		"vmovdqu64 0x40(%1), %%zmm1\n"
		"vmovntdq %%zmm0, 0x00(%0)\n"
		"vmovntdq %%zmm1, 0x40(%0)\n"
		:
		: "r"(dest), "r"(src)
		: "memory", "zmm0", "zmm1"
	);
#else
	__m512i zmm0 = _mm512_loadu_si512((__m512i *)src + 0);
	__m512i zmm1 = _mm512_loadu_si512((__m512i *)src + 1);

	_mm512_stream_si512((__m512i *)dest + 0, zmm0);
	_mm512_stream_si512((__m512i *)dest + 1, zmm1);
#endif
}

static inline void
memmove_movnt1x64b(char *dest, const char *src)
{
#ifdef USE_ASM
	asm(
		"vmovdqu64 0x00(%1), %%zmm0\n"
		"vmovntdq %%zmm0, 0x00(%0)\n"
		:
		: "r"(dest), "r"(src)
		: "memory", "zmm0"
	);
#else
	__m512i zmm0 = _mm512_loadu_si512((__m512i *)src + 0);

	_mm512_stream_si512((__m512i *)dest + 0, zmm0);
#endif
}

#else
#error set USE_SSE2 or USE_AVX or USE_AVX512F
const char *level = "?";
void memmove_movnt1x64b(char *dest, const char *src);
#endif

void
wc_memcpy(char *dest, const char *src, size_t sz)
{
#if defined(USE_AVX)
	while (sz >= 11 * 64) {
		memmove_movnt8x64b(dest, src);
		dest += 8 * 64;
		src += 8 * 64;
		sz -= 8 * 64;

		memmove_movnt2x64b(dest, src);
		dest += 2 * 64;
		src += 2 * 64;
		sz -= 2 * 64;

		memmove_movnt1x64b(dest, src);
		dest += 1 * 64;
		src += 1 * 64;
		sz -= 1 * 64;

		_mm_sfence();
	}

	if (sz >= 8 * 64) {
		memmove_movnt8x64b(dest, src);
		dest += 8 * 64;
		src += 8 * 64;
		sz -= 8 * 64;
	}

	if (sz >= 4 * 64) {
		memmove_movnt4x64b(dest, src);
		dest += 4 * 64;
		src += 4 * 64;
		sz -= 4 * 64;
	}

	if (sz >= 2 * 64) {
		memmove_movnt2x64b(dest, src);
		dest += 2 * 64;
		src += 2 * 64;
		sz -= 2 * 64;
	}

	if (sz >= 64) {
		memmove_movnt1x64b(dest, src);
		dest += 64;
		src += 64;
		sz -= 64;
	}

	_mm256_zeroupper();
#else
	while (sz >= 11 * 64) {
		memmove_movnt4x64b(dest, src);
		dest += 4 * 64;
		src += 4 * 64;
		sz -= 4 * 64;

		memmove_movnt4x64b(dest, src);
		dest += 4 * 64;
		src += 4 * 64;
		sz -= 4 * 64;

		memmove_movnt2x64b(dest, src);
		dest += 2 * 64;
		src += 2 * 64;
		sz -= 2 * 64;

		memmove_movnt1x64b(dest, src);
		dest += 64;
		src += 64;
		sz -= 64;

		_mm_sfence();
	}

	if (sz >= 4 * 64) {
		memmove_movnt4x64b(dest, src);
		dest += 4 * 64;
		src += 4 * 64;
		sz -= 4 * 64;
	}

	if (sz >= 2 * 64) {
		memmove_movnt2x64b(dest, src);
		dest += 2 * 64;
		src += 2 * 64;
		sz -= 2 * 64;
	}

	if (sz >= 64) {
		memmove_movnt1x64b(dest, src);
		dest += 64;
		src += 64;
		sz -= 64;
	}
#endif

	_mm_sfence();
}
