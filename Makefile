# SPDX-License-Identifier: BSD-3-Clause
# Copyright 2020, Intel Corporation

PMEM?=/dev/shm/aaa
NUMANODE?=0
CLS?=1
CLSMAX?=16

all: wc_sse2 wc_avx wc_avx512f

run_all: run_sse2 run_avx run_avx512f

clean:
	rm -f wc_sse2 wc_avx wc_avx512f

wc_sse2: wc.c Makefile
	$(CC) wc.c -DUSE_SSE2 $(shell pkg-config --cflags libpmem) -o wc_sse2 $(shell pkg-config --libs libpmem) -O2

wc_avx: wc.c Makefile
	$(CC) wc.c -DUSE_AVX $(shell pkg-config --cflags libpmem) -o wc_avx $(shell pkg-config --libs libpmem) -mavx -O2

wc_avx512f: wc.c Makefile
	$(CC) wc.c -DUSE_AVX512F $(shell pkg-config --cflags libpmem) -o wc_avx512f $(shell pkg-config --libs libpmem) -mavx512f -O2

run_sse2: wc_sse2
	LD_LIBRARY_PATH=$(shell pkg-config --variable libdir libpmem):$LD_LIBRARY_PATH numactl -N $(NUMANODE) ./wc_sse2 $(PMEM) $(CLS) $(CLSMAX)

run_avx: wc_avx
	LD_LIBRARY_PATH=$(shell pkg-config --variable libdir libpmem):$LD_LIBRARY_PATH numactl -N $(NUMANODE) ./wc_avx $(PMEM) $(CLS) $(CLSMAX)
	
run_avx512f: wc_avx512f
	LD_LIBRARY_PATH=$(shell pkg-config --variable libdir libpmem):$LD_LIBRARY_PATH numactl -N $(NUMANODE) ./wc_avx512f $(PMEM) $(CLS) $(CLSMAX)
