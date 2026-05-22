/*
 * Runtime resolution of the per-package external files directory.
 *
 * Multiple xemu subsystems (sched-android, adpf-android, the pgraph_vk
 * probe runner, the shader dumper, the NaN probe sentinel) need to
 * read/write under the app's external files dir. Hardcoding
 * "com.izzy2lost.x1box" breaks the moment you run the `perftest`
 * build variant (applicationId becomes "com.izzy2lost.x1box.perftest")
 * — config files / sentinels live under the wrong path. This helper
 * resolves the correct path at runtime from /proc/self/cmdline.
 */

#ifndef QEMU_ANDROID_PATHS_H
#define QEMU_ANDROID_PATHS_H

#include <stddef.h>

/* Returns the running package name (e.g. "com.izzy2lost.x1box" or
 * "com.izzy2lost.x1box.perftest"), stripped of any ":<subprocess>"
 * suffix that Android appends when the running process isn't the
 * package's main one. Computed once from /proc/self/cmdline and
 * cached. Returns NULL on non-Android or if cmdline read fails. */
const char *android_package_name(void);

/* Returns "/storage/emulated/0/Android/data/<pkg>/files/x1box" for
 * the currently-running package. Computed once and cached. Returns
 * NULL on non-Android. */
const char *android_x1box_ext_dir(void);

#endif /* QEMU_ANDROID_PATHS_H */
