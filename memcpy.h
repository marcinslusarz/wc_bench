// SPDX-License-Identifier: BSD-3-Clause
// Copyright 2020, Intel Corporation

#ifndef WC_MEMCPY_H
#define WC_MEMCPY_H

extern int max_batch_size;
extern const char *level;
void wc_memcpy(char *dest, const char *src, size_t sz);

#endif
