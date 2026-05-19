/*
 * Geforce NV2A PGRAPH Vulkan Renderer
 *
 * Copyright (c) 2024 Matt Borgerson
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 */

#ifndef HW_XBOX_NV2A_PGRAPH_VK_DEBUG_H
#define HW_XBOX_NV2A_PGRAPH_VK_DEBUG_H

#define VK_LOG_VERBOSE 0

#ifdef __ANDROID__
#include <android/log.h>
#include <sched.h>
#include <unistd.h>      /* _exit */
#include "system/runstate.h"
#define VK_GRACEFUL_SHUTDOWN_CAUSE SHUTDOWN_CAUSE_HOST_UI
#if VK_LOG_VERBOSE
#define VK_LOG(fmt, ...) \
    __android_log_print(ANDROID_LOG_INFO, "hakuX-vk", fmt, ##__VA_ARGS__)
#else
#define VK_LOG(fmt, ...) do { } while (0)
#endif
#define VK_LOG_ERROR(fmt, ...) \
    __android_log_print(ANDROID_LOG_ERROR, "hakuX-vk", fmt, ##__VA_ARGS__)
#else
#define VK_LOG(fmt, ...) do { } while (0)
#define VK_LOG_ERROR(fmt, ...) do { fprintf(stderr, "xemu-vk: " fmt "\n", ##__VA_ARGS__); } while (0)
#endif

#define DEBUG_VK 0

extern int nv2a_vk_dgroup_indent;

#define NV2A_VK_XDPRINTF(x, fmt, ...)                                  \
    do {                                                               \
        if (x) {                                                       \
            fprintf(stderr, "%*s" fmt "\n", nv2a_vk_dgroup_indent, "", \
                    ##__VA_ARGS__);                                    \
        }                                                              \
    } while (0)

#define NV2A_VK_DPRINTF(fmt, ...) NV2A_VK_XDPRINTF(DEBUG_VK, fmt, ##__VA_ARGS__)

#define NV2A_VK_DGROUP_BEGIN(fmt, ...)                  \
    do {                                                \
        NV2A_VK_XDPRINTF(DEBUG_VK, fmt, ##__VA_ARGS__); \
        nv2a_vk_dgroup_indent++;                        \
    } while (0)

#define NV2A_VK_DGROUP_END(...)             \
    do {                                    \
        nv2a_vk_dgroup_indent--;            \
        assert(nv2a_vk_dgroup_indent >= 0); \
    } while (0)

#ifdef __ANDROID__
/*
 * Android: on VK_CHECK failure we want the VkResult in logcat (it would
 * otherwise be lost to fprintf(stderr) which Android drops). We also
 * want a graceful exit instead of libc abort — abort produces an ANR
 * dialog and tombstone, which is terrible UX whenever Mali's stock
 * driver trips a DEVICE_LOST on Halo 2 etc.
 *
 * Flow:
 *   1. Log the failure at FATAL severity via __android_log_print so the
 *      VkResult and source location reach logcat.
 *   2. Request a graceful QEMU shutdown via qemu_system_shutdown_request.
 *   3. Park this thread forever with sched_yield. The main loop will
 *      exit the process cleanly.
 *
 * We deliberately do NOT call abort/assert here. The previous version
 * routed through __android_log_assert which calls libc abort.
 */
#define VK_CHECK(x)                                                            \
    do {                                                                       \
        VkResult vk_result = (x);                                              \
        if (vk_result != VK_SUCCESS) {                                         \
            __android_log_print(ANDROID_LOG_FATAL, "hakuX-vk",                 \
                "VK_CHECK FAILED: %s = %d at %s:%d — requesting emulator "    \
                "quit (returns to launcher activity, user sees what "          \
                "happened)",                                                   \
                #x, vk_result, __FILE__, __LINE__);                            \
            extern void nv2a_dbg_request_emulator_quit(void);                  \
            nv2a_dbg_request_emulator_quit();                                  \
            /*                                                                 \
             * Park this thread forever so it doesn't drive more Vulkan       \
             * calls into the dead device. The main loop wakes on our         \
             * SDL_QUIT push, processes the shutdown request, tears down     \
             * the activity, and the user returns to the launcher. The       \
             * sleep(60) loop is just to make this thread go to sleep        \
             * cheaply while the shutdown unfolds.                            \
             */                                                                \
            for (;;) { sleep(60); }                                            \
        }                                                                      \
    } while (0)
#else
#define VK_CHECK(x)                                                            \
    do {                                                                       \
        VkResult vk_result = (x);                                              \
        if (vk_result != VK_SUCCESS) {                                         \
            fprintf(stderr, "VK_CHECK FAILED: %s = %d at %s:%d\n",             \
                    #x, vk_result, __FILE__, __LINE__);                        \
        }                                                                      \
        assert(vk_result == VK_SUCCESS && "vk check failed");                  \
    } while (0)
#endif

void pgraph_vk_debug_frame_terminator(void);

#endif
