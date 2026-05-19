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

#include "renderer.h"
#include "debug.h"

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include "system/runstate.h"
#include <SDL.h>

int nv2a_vk_dgroup_indent = 0;

void pgraph_vk_debug_init(void)
{
}

void pgraph_vk_debug_frame_terminator(void)
{
}

/* Request a clean emulator shutdown. Called from VK_CHECK and the Mali
 * DEVICE_LOST handler in render_thread.c so the user lands back at the
 * launcher activity instead of seeing an ANR / abort tombstone. Thread-
 * safe: SDL_PushEvent is documented as safe from any thread, and
 * qemu_system_shutdown_request only flips an atomic flag the main loop
 * polls. */
void nv2a_dbg_request_emulator_quit(void)
{
    SDL_Event quit = { .type = SDL_QUIT };
    SDL_PushEvent(&quit);
    qemu_system_shutdown_request(SHUTDOWN_CAUSE_HOST_UI);
}

void pgraph_vk_insert_debug_marker(PGRAPHVkState *r, VkCommandBuffer cmd,
                                   float color[4], const char *format, ...)
{
    if (!r->debug_utils_extension_enabled) {
        return;
    }

    char *buf = NULL;

    va_list args;
    va_start(args, format);
    int err = vasprintf(&buf, format, args);
    assert(err >= 0);
    va_end(args);

    VkDebugUtilsLabelEXT label_info = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
        .pLabelName = buf,
    };
    memcpy(label_info.color, color, 4 * sizeof(float));
    vkCmdInsertDebugUtilsLabelEXT(cmd, &label_info);
    free(buf);
}

void pgraph_vk_begin_debug_marker(PGRAPHVkState *r, VkCommandBuffer cmd,
                                  float color[4], const char *format, ...)
{
    if (!r->debug_utils_extension_enabled) {
        return;
    }

    char *buf = NULL;

    va_list args;
    va_start(args, format);
    int err = vasprintf(&buf, format, args);
    assert(err >= 0);
    va_end(args);

    VkDebugUtilsLabelEXT label_info = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
        .pLabelName = buf,
    };
    memcpy(label_info.color, color, 4 * sizeof(float));
    vkCmdBeginDebugUtilsLabelEXT(cmd, &label_info);
    free(buf);

    r->debug_depth += 1;
    assert(r->debug_depth < 10 && "Missing pgraph_vk_debug_marker_end?");
}

void pgraph_vk_end_debug_marker(PGRAPHVkState *r, VkCommandBuffer cmd)
{
    if (!r->debug_utils_extension_enabled) {
        return;
    }

    vkCmdEndDebugUtilsLabelEXT(cmd);
    assert(r->debug_depth > 0);
    r->debug_depth -= 1;
}
