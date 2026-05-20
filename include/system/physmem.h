/*
 * QEMU physical memory interfaces (target independent).
 *
 *  Copyright (c) 2003 Fabrice Bellard
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef QEMU_SYSTEM_PHYSMEM_H
#define QEMU_SYSTEM_PHYSMEM_H

#include "exec/hwaddr.h"
#include "exec/ramlist.h"

#define DIRTY_CLIENTS_ALL     ((1 << DIRTY_MEMORY_NUM) - 1)
#define DIRTY_CLIENTS_NOCODE  (DIRTY_CLIENTS_ALL & ~(1 << DIRTY_MEMORY_CODE))

bool physical_memory_get_dirty_flag(ram_addr_t addr, unsigned client);

bool physical_memory_is_clean(ram_addr_t addr);

/*
 * Like physical_memory_is_clean() but only considers the clients in
 * @check_mask plus DIRTY_MEMORY_CODE (which is special-cased — code
 * dirty bit is always relevant when TCG is on).
 *
 * Returns true if the page is "clean" relative to those clients (i.e.
 * any expected bit is unset, meaning some consumer hasn't yet seen the
 * write). Returns false only when all expected bits are set, which is
 * the signal to drop the NOTDIRTY/slow-path flag on the TLB entry.
 *
 * Use this when calling code knows the actual dirty_log_mask of the
 * backing MR (e.g. from CPUTLBEntryFull.dirty_log_mask) and doesn't
 * want to require bits from clients that have no consumer on this
 * platform — the old physical_memory_is_clean() requires ALL five
 * bits, which forces the slow path to repeat on every write whenever
 * any client has no listener.
 */
bool physical_memory_is_clean_for(ram_addr_t addr, uint8_t check_mask);

uint8_t physical_memory_range_includes_clean(ram_addr_t start,
                                             ram_addr_t length,
                                             uint8_t mask);

void physical_memory_set_dirty_flag(ram_addr_t addr, unsigned client);

void physical_memory_set_dirty_range(ram_addr_t start, ram_addr_t length,
                                     uint8_t mask);

/*
 * Contrary to physical_memory_sync_dirty_bitmap() this function returns
 * the number of dirty pages in @bitmap passed as argument. On the other hand,
 * physical_memory_sync_dirty_bitmap() returns newly dirtied pages that
 * weren't set in the global migration bitmap.
 */
uint64_t physical_memory_set_dirty_lebitmap(unsigned long *bitmap,
                                            ram_addr_t start,
                                            ram_addr_t pages);

void physical_memory_dirty_bits_cleared(ram_addr_t start, ram_addr_t length);

bool physical_memory_test_and_clear_dirty(ram_addr_t start,
                                          ram_addr_t length,
                                          unsigned client);

DirtyBitmapSnapshot *
physical_memory_snapshot_and_clear_dirty(MemoryRegion *mr, hwaddr offset,
                                         hwaddr length, unsigned client);

bool physical_memory_snapshot_get_dirty(DirtyBitmapSnapshot *snap,
                                        ram_addr_t start,
                                        ram_addr_t length);

#endif
