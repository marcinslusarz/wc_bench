# SPDX-License-Identifier: BSD-3-Clause
# Copyright 2020, Intel Corporation

PMEM?=/dev/shm/aaa
NUMANODE?=0
CLS?=1
CLSMAX?=16
MEMCPY_CFLAGS=-DUSE_ALL_REGS -O2

MAIN_CFLAGS=$(shell pkg-config --cflags libpmem)
LDFLAGS=$(shell pkg-config --libs libpmem)
LIBDIR=$(shell pkg-config --variable libdir libpmem)

all: wc_sse2 wc_avx wc_avx512f

run_all: run_sse2 run_avx run_avx512f

clean:
	rm -f wc_sse2 wc_avx wc_avx512f *.o


memcpy_sse2.o: memcpy.c Makefile
	$(CC) -c memcpy.c -DUSE_SSE2    $(MEMCPY_CFLAGS)           -o memcpy_sse2.o

memcpy_avx.o: memcpy.c Makefile
	$(CC) -c memcpy.c -DUSE_AVX     $(MEMCPY_CFLAGS) -mavx     -o memcpy_avx.o

memcpy_avx512f.o: memcpy.c Makefile
	$(CC) -c memcpy.c -DUSE_AVX512F $(MEMCPY_CFLAGS) -mavx512f -o memcpy_avx512f.o


main.o: main.c Makefile
	$(CC) -c main.c $(MAIN_CFLAGS) -o main.o


wc_sse2:    main.o Makefile memcpy_sse2.o
	$(CC) main.o memcpy_sse2.o    -o wc_sse2    $(LDFLAGS)

wc_avx:     main.o Makefile memcpy_avx.o
	$(CC) main.o memcpy_avx.o     -o wc_avx     $(LDFLAGS)

wc_avx512f: main.o Makefile memcpy_avx512f.o
	$(CC) main.o memcpy_avx512f.o -o wc_avx512f $(LDFLAGS)


run_sse2: wc_sse2
	LD_LIBRARY_PATH=$(LIBDIR):$LD_LIBRARY_PATH numactl -N $(NUMANODE) ./wc_sse2    $(PMEM) $(CLS) $(CLSMAX)

run_avx: wc_avx
	LD_LIBRARY_PATH=$(LIBDIR):$LD_LIBRARY_PATH numactl -N $(NUMANODE) ./wc_avx     $(PMEM) $(CLS) $(CLSMAX)
	
run_avx512f: wc_avx512f
	LD_LIBRARY_PATH=$(LIBDIR):$LD_LIBRARY_PATH numactl -N $(NUMANODE) ./wc_avx512f $(PMEM) $(CLS) $(CLSMAX)
