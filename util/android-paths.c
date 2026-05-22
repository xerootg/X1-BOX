/*
 * util/android-paths.c — runtime per-package external files dir.
 * See android-paths.h for rationale.
 *
 * Implementation note: /proc/self/cmdline holds the package name as
 * its first NUL-terminated arg. For app's main process, this is the
 * applicationId verbatim. For Android `android:process=":xemu"`
 * processes (our case for the emulation half), cmdline reads
 * "com.izzy2lost.x1box.perftest:xemu" — strip the colon-suffix to
 * recover the package proper.
 */

#include "qemu/osdep.h"
#include "qemu/android-paths.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>

#define FALLBACK_PKG "com.izzy2lost.x1box"

static char g_pkg[128];
static char g_dir[256];
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static void compute_paths(void)
{
#ifdef __ANDROID__
    char buf[128] = {0};
    FILE *f = fopen("/proc/self/cmdline", "rb");
    if (f) {
        size_t n = fread(buf, 1, sizeof(buf) - 1, f);
        fclose(f);
        if (n > 0) buf[n] = '\0';
    }
    if (buf[0] == '\0') {
        /* No cmdline available — use the canonical debug-build path. */
        snprintf(g_pkg, sizeof(g_pkg), "%s", FALLBACK_PKG);
    } else {
        /* Strip ":<subprocess>" suffix that Android tacks on for
         * processes started under android:process=":name". */
        char *colon = strchr(buf, ':');
        if (colon) *colon = '\0';
        snprintf(g_pkg, sizeof(g_pkg), "%s", buf);
    }
    snprintf(g_dir, sizeof(g_dir),
             "/storage/emulated/0/Android/data/%s/files/x1box", g_pkg);
#else
    g_pkg[0] = '\0';
    g_dir[0] = '\0';
#endif
}

const char *android_package_name(void)
{
    pthread_once(&g_once, compute_paths);
    return g_pkg[0] ? g_pkg : NULL;
}

const char *android_x1box_ext_dir(void)
{
    pthread_once(&g_once, compute_paths);
    return g_dir[0] ? g_dir : NULL;
}
