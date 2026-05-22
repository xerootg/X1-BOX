/*
 * xemu xpacks — runtime patch-pack system implementation.
 *
 * Pack layout on disk:
 *   <base>/xpacks/
 *     <title_id_hex>/                       e.g. "4D530064"
 *       <pack_dir>/
 *         pack.toml                         INI-style manifest (see parser)
 *         patches/...                       optional binary blobs (cave/tag)
 *         shaders/<hash>.spv                optional SPIR-V replacement
 *
 *   <base>/xpacks/enabled.txt
 *       Newline-delimited list of "<title_id>/<pack_dir>" entries that are
 *       enabled. Written by the Android UI; read here at apply time.
 *
 * Manifest format (subset of TOML — parsed line-by-line):
 *
 *   schema = 1
 *   name   = "Friendly name"
 *
 *   [match]
 *   title_id = "4D530064"
 *
 *   [[patch]]
 *   kind        = "bytes"
 *   description = "..."
 *   address     = 0x0045DC6C
 *   expected    = "89 88 08 3D"
 *   replace     = "89 88 88 3C"
 *
 *   [[patch]]
 *   kind        = "shader"
 *   glsl_hash   = "abcd1234ef567890"
 *   file        = "shaders/abcd1234ef567890.spv"
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "system/dma.h"
#include "system/hw_accel.h"
#include "exec/cpu-common.h"
#include "exec/target_page.h"
#include "cpu.h"

#include "hw/xbox/xpacks.h"
#include "xemu-xbe.h"
#include "ui/xemu-settings.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <glib.h>
#include <glib/gstdio.h>

#define XPACKS_TAG "xpacks"

typedef struct BytesPatch {
    char *description;
    uint32_t address;       /* guest virtual */
    uint8_t *expected;      /* may be NULL */
    size_t expected_len;
    uint8_t *replace;
    size_t replace_len;
} BytesPatch;

/* Pattern-anchored patch: scan a guest-virtual range for a pattern (bytes
 * with '??' wildcards), then write `replace` at <match_addr + search_offset>.
 * Used for tag/map overlays where data lives at non-deterministic addresses
 * but has a stable internal signature. */
typedef struct PatternPatch {
    char *description;
    uint8_t *pattern;       /* bytes */
    uint8_t *pattern_mask;  /* 1 = compare, 0 = wildcard */
    size_t pattern_len;
    uint32_t search_start;  /* guest virtual; default 0x00010000 */
    uint32_t search_end;    /* guest virtual; default 0x10000000 */
    int32_t  search_offset; /* byte offset from match start to patch start */
    uint8_t *expected;      /* may be NULL */
    size_t expected_len;
    uint8_t *replace;
    size_t replace_len;
    bool applied;
} PatternPatch;

typedef struct ShaderOverride {
    uint64_t glsl_hash;
    char *file_abs;         /* absolute path to .spv */
} ShaderOverride;

typedef struct XPack {
    char *id;               /* "<title_id>/<pack_dir>" */
    char *dir_abs;          /* absolute pack directory */
    char *name;
    uint32_t match_title_id;
    /* Optional list of accepted SHA1 hex strings (40 char) for the xbe code
     * section — if non-empty, the pack is only applied to a matching build. */
    GPtrArray *match_xbe_sha1;
    GArray *bytes_patches;  /* BytesPatch */
    GArray *pattern_patches;/* PatternPatch */
    GArray *shader_ovr;     /* ShaderOverride */
} XPack;

static GArray *g_packs = NULL;          /* XPack */
static uint32_t g_applied_title_id = 0; /* set after a successful apply */
static int g_pending_patterns = 0;      /* >0 means xpacks_tick() should keep scanning */
static bool g_inited = false;

/* ---------- utilities ---------- */

static char *xpacks_root_dir(void)
{
    return g_strdup_printf("%sxpacks", xemu_settings_get_base_path());
}

static bool parse_hex_byte_string(const char *src, uint8_t **out_buf, size_t *out_len)
{
    /* Accept "DE AD BE EF", "deadbeef", "0xDE,0xAD" — anything with hex pairs. */
    GByteArray *a = g_byte_array_new();
    const char *p = src;
    while (*p) {
        while (*p && !g_ascii_isxdigit((unsigned char)*p)) p++;
        if (!*p) break;
        if (!g_ascii_isxdigit((unsigned char)p[1])) {
            g_byte_array_free(a, TRUE);
            return false;
        }
        char pair[3] = { p[0], p[1], 0 };
        unsigned v = (unsigned)strtoul(pair, NULL, 16);
        uint8_t byte = (uint8_t)v;
        g_byte_array_append(a, &byte, 1);
        p += 2;
    }
    if (a->len == 0) {
        g_byte_array_free(a, TRUE);
        return false;
    }
    *out_len = a->len;
    *out_buf = g_byte_array_free(a, FALSE);
    return true;
}

/* Parse a pattern string into bytes + mask. '??' (or any non-hex pair starting
 * with '?') becomes a wildcard byte (mask=0, value=0). Other tokens follow
 * parse_hex_byte_string rules. Returns false on empty/parse error. */
static bool parse_pattern_string(const char *src,
                                 uint8_t **out_bytes,
                                 uint8_t **out_mask,
                                 size_t *out_len)
{
    GByteArray *bytes = g_byte_array_new();
    GByteArray *mask  = g_byte_array_new();
    const char *p = src;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == ',' || *p == '\n' ||
               *p == '\r' || *p == '"') p++;
        if (!*p) break;
        if (p[0] == '?' && p[1] == '?') {
            uint8_t z = 0;
            g_byte_array_append(bytes, &z, 1);
            g_byte_array_append(mask,  &z, 1);
            p += 2;
            continue;
        }
        if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) p += 2;
        if (!g_ascii_isxdigit((unsigned char)p[0]) ||
            !g_ascii_isxdigit((unsigned char)p[1])) {
            g_byte_array_free(bytes, TRUE);
            g_byte_array_free(mask, TRUE);
            return false;
        }
        char pair[3] = { p[0], p[1], 0 };
        uint8_t b = (uint8_t)strtoul(pair, NULL, 16);
        uint8_t m = 0xFF;
        g_byte_array_append(bytes, &b, 1);
        g_byte_array_append(mask,  &m, 1);
        p += 2;
    }
    if (bytes->len == 0) {
        g_byte_array_free(bytes, TRUE);
        g_byte_array_free(mask, TRUE);
        return false;
    }
    *out_len = bytes->len;
    *out_bytes = g_byte_array_free(bytes, FALSE);
    *out_mask  = g_byte_array_free(mask, FALSE);
    return true;
}

static bool parse_i32(const char *s, int32_t *out)
{
    if (!s || !*s) return false;
    const char *p = s;
    while (*p == ' ' || *p == '\t' || *p == '"') p++;
    int sign = 1;
    if (*p == '-') { sign = -1; p++; }
    else if (*p == '+') p++;
    char *end = NULL;
    errno = 0;
    long v;
    if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
        v = strtol(p + 2, &end, 16);
    } else {
        v = strtol(p, &end, 0);
    }
    if (errno || end == p) return false;
    *out = (int32_t)(sign * v);
    return true;
}

static bool parse_u32(const char *s, uint32_t *out)
{
    if (!s || !*s) return false;
    const char *p = s;
    while (*p == ' ' || *p == '\t' || *p == '"') p++;
    char *end = NULL;
    errno = 0;
    unsigned long v;
    if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
        v = strtoul(p + 2, &end, 16);
    } else {
        v = strtoul(p, &end, 0);
    }
    if (errno || end == p) return false;
    *out = (uint32_t)v;
    return true;
}

static bool parse_u64_hex(const char *s, uint64_t *out)
{
    if (!s || !*s) return false;
    const char *p = s;
    while (*p == ' ' || *p == '\t' || *p == '"') p++;
    if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) p += 2;
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(p, &end, 16);
    if (errno || end == p) return false;
    *out = (uint64_t)v;
    return true;
}

static char *strip_quotes_and_whitespace(const char *s)
{
    if (!s) return NULL;
    while (*s == ' ' || *s == '\t') s++;
    size_t n = strlen(s);
    while (n > 0 && (s[n-1] == ' ' || s[n-1] == '\t' || s[n-1] == '\n' ||
                     s[n-1] == '\r')) n--;
    if (n >= 2 && s[0] == '"' && s[n-1] == '"') { s++; n -= 2; }
    return g_strndup(s, n);
}

/* ---------- guest memory write ---------- */

/* Translate a guest virtual address to physical via the running vCPU.
 * Returns -1 if the page is unmapped. Caller is responsible for ensuring
 * the cpu is in a state safe to query (we synchronize). */
static int virt_to_phys_guest(uint32_t vaddr, hwaddr *out_phys)
{
    CPUState *cs = qemu_get_cpu(0);
    if (!cs) return -1;
    cpu_synchronize_state(cs);
    MemTxAttrs attrs;
    hwaddr gpa = cpu_get_phys_page_attrs_debug(cs,
                    vaddr & TARGET_PAGE_MASK, &attrs);
    if (gpa == (hwaddr)-1) return -1;
    *out_phys = gpa + (vaddr & ~TARGET_PAGE_MASK);
    return 0;
}

static bool guest_read(uint32_t vaddr, void *buf, size_t len)
{
    size_t done = 0;
    while (done < len) {
        hwaddr phys;
        if (virt_to_phys_guest(vaddr + done, &phys) != 0) return false;
        size_t in_page = TARGET_PAGE_SIZE - (phys & ~TARGET_PAGE_MASK);
        size_t n = MIN(len - done, in_page);
        if (dma_memory_read(&address_space_memory, phys,
                            (uint8_t *)buf + done, n,
                            MEMTXATTRS_UNSPECIFIED) != MEMTX_OK) {
            return false;
        }
        done += n;
    }
    return true;
}

static bool guest_write(uint32_t vaddr, const void *buf, size_t len)
{
    size_t done = 0;
    while (done < len) {
        hwaddr phys;
        if (virt_to_phys_guest(vaddr + done, &phys) != 0) return false;
        size_t in_page = TARGET_PAGE_SIZE - (phys & ~TARGET_PAGE_MASK);
        size_t n = MIN(len - done, in_page);
        if (dma_memory_write(&address_space_memory, phys,
                             (const uint8_t *)buf + done, n,
                             MEMTXATTRS_UNSPECIFIED) != MEMTX_OK) {
            return false;
        }
        done += n;
    }
    return true;
}

/* ---------- enabled-set ---------- */

static GHashTable *load_enabled_set(void)
{
    GHashTable *set = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
    char *root = xpacks_root_dir();
    char *path = g_build_filename(root, "enabled.txt", NULL);
    g_free(root);
    gchar *contents = NULL;
    gsize len = 0;
    if (g_file_get_contents(path, &contents, &len, NULL)) {
        gchar **lines = g_strsplit(contents, "\n", -1);
        for (gchar **p = lines; *p; p++) {
            gchar *line = g_strstrip(*p);
            if (line[0] == '\0' || line[0] == '#') continue;
            g_hash_table_add(set, g_strdup(line));
        }
        g_strfreev(lines);
        g_free(contents);
    }
    g_free(path);
    return set;
}

/* ---------- manifest parser ---------- */

typedef enum {
    SEC_ROOT,
    SEC_META,
    SEC_MATCH,
    SEC_PATCH_BYTES,
    SEC_PATCH_PATTERN,
    SEC_PATCH_SHADER,
    SEC_PATCH_UNKNOWN,
} ManifestSection;

typedef struct {
    XPack *pack;
    ManifestSection section;
    /* current-patch staging */
    BytesPatch cur_bytes;
    bool cur_bytes_have_replace;
    PatternPatch cur_pattern;
    bool cur_pattern_have_replace;
    bool cur_pattern_have_signature;
    ShaderOverride cur_shader;
    bool cur_shader_have;
} ParseState;

static void reset_cur_bytes(ParseState *st)
{
    g_free(st->cur_bytes.description);
    g_free(st->cur_bytes.expected);
    g_free(st->cur_bytes.replace);
    memset(&st->cur_bytes, 0, sizeof(st->cur_bytes));
    st->cur_bytes_have_replace = false;
}

static void reset_cur_pattern(ParseState *st)
{
    g_free(st->cur_pattern.description);
    g_free(st->cur_pattern.pattern);
    g_free(st->cur_pattern.pattern_mask);
    g_free(st->cur_pattern.expected);
    g_free(st->cur_pattern.replace);
    memset(&st->cur_pattern, 0, sizeof(st->cur_pattern));
    st->cur_pattern_have_replace = false;
    st->cur_pattern_have_signature = false;
}

static void reset_cur_shader(ParseState *st)
{
    g_free(st->cur_shader.file_abs);
    memset(&st->cur_shader, 0, sizeof(st->cur_shader));
    st->cur_shader_have = false;
}

static void flush_current_patch(ParseState *st)
{
    if (st->section == SEC_PATCH_BYTES) {
        if (st->cur_bytes_have_replace) {
            g_array_append_val(st->pack->bytes_patches, st->cur_bytes);
            memset(&st->cur_bytes, 0, sizeof(st->cur_bytes));
            st->cur_bytes_have_replace = false;
        } else {
            reset_cur_bytes(st);
        }
    } else if (st->section == SEC_PATCH_PATTERN) {
        if (st->cur_pattern_have_replace && st->cur_pattern_have_signature) {
            /* Defaults for optional search range. */
            if (st->cur_pattern.search_start == 0 &&
                st->cur_pattern.search_end == 0) {
                st->cur_pattern.search_start = 0x00010000;
                st->cur_pattern.search_end   = 0x10000000;
            }
            g_array_append_val(st->pack->pattern_patches, st->cur_pattern);
            memset(&st->cur_pattern, 0, sizeof(st->cur_pattern));
            st->cur_pattern_have_replace = false;
            st->cur_pattern_have_signature = false;
        } else {
            reset_cur_pattern(st);
        }
    } else if (st->section == SEC_PATCH_SHADER) {
        if (st->cur_shader_have && st->cur_shader.file_abs) {
            g_array_append_val(st->pack->shader_ovr, st->cur_shader);
            memset(&st->cur_shader, 0, sizeof(st->cur_shader));
            st->cur_shader_have = false;
        } else {
            reset_cur_shader(st);
        }
    }
}

static void parse_manifest_line(ParseState *st, char *line)
{
    /* strip comments */
    char *hash = strchr(line, '#');
    if (hash) *hash = '\0';
    g_strstrip(line);
    if (line[0] == '\0') return;

    /* section header */
    if (line[0] == '[') {
        flush_current_patch(st);
        if (!strcmp(line, "[meta]")) {
            st->section = SEC_META;
        } else if (!strcmp(line, "[match]")) {
            st->section = SEC_MATCH;
        } else if (!strcmp(line, "[[patch]]")) {
            /* Default to "bytes" until kind= sets it. */
            st->section = SEC_PATCH_BYTES;
        } else {
            st->section = SEC_ROOT;
        }
        return;
    }

    char *eq = strchr(line, '=');
    if (!eq) return;
    *eq = '\0';
    char *key = g_strstrip(line);
    char *val_raw = eq + 1;
    char *val = strip_quotes_and_whitespace(val_raw);

    switch (st->section) {
    case SEC_META:
    case SEC_ROOT:
        if (!strcmp(key, "name")) {
            g_free(st->pack->name);
            st->pack->name = g_strdup(val);
        }
        break;
    case SEC_MATCH:
        if (!strcmp(key, "title_id")) {
            uint32_t tid = 0;
            if (parse_u32(val, &tid)) st->pack->match_title_id = tid;
        } else if (!strcmp(key, "xbe_code_sha1")) {
            /* Single value or "a, b, c" list of 40-char hex strings. */
            if (!st->pack->match_xbe_sha1) {
                st->pack->match_xbe_sha1 =
                    g_ptr_array_new_with_free_func(g_free);
            }
            gchar **toks = g_strsplit(val, ",", -1);
            for (gchar **t = toks; *t; t++) {
                gchar *s = g_strstrip(*t);
                /* allow ["a","b"] TOML-array style by stripping brackets/quotes */
                while (*s == '[' || *s == '"') s++;
                size_t n = strlen(s);
                while (n > 0 && (s[n-1] == ']' || s[n-1] == '"' ||
                                 s[n-1] == ' ')) n--;
                if (n == 40) {
                    g_ptr_array_add(st->pack->match_xbe_sha1,
                                    g_ascii_strdown(g_strndup(s, n), 40));
                }
            }
            g_strfreev(toks);
        }
        break;
    case SEC_PATCH_BYTES:
        if (!strcmp(key, "kind")) {
            if (!strcmp(val, "shader")) {
                reset_cur_bytes(st);
                memset(&st->cur_shader, 0, sizeof(st->cur_shader));
                st->cur_shader_have = false;
                st->section = SEC_PATCH_SHADER;
            } else if (!strcmp(val, "pattern_bytes")) {
                reset_cur_bytes(st);
                reset_cur_pattern(st);
                st->section = SEC_PATCH_PATTERN;
            } else if (!strcmp(val, "bytes")) {
                /* already bytes */
            } else {
                st->section = SEC_PATCH_UNKNOWN;
            }
        } else if (!strcmp(key, "description")) {
            g_free(st->cur_bytes.description);
            st->cur_bytes.description = g_strdup(val);
        } else if (!strcmp(key, "address")) {
            parse_u32(val, &st->cur_bytes.address);
        } else if (!strcmp(key, "expected")) {
            g_free(st->cur_bytes.expected);
            st->cur_bytes.expected = NULL;
            st->cur_bytes.expected_len = 0;
            parse_hex_byte_string(val, &st->cur_bytes.expected,
                                  &st->cur_bytes.expected_len);
        } else if (!strcmp(key, "replace")) {
            g_free(st->cur_bytes.replace);
            st->cur_bytes.replace = NULL;
            st->cur_bytes.replace_len = 0;
            if (parse_hex_byte_string(val, &st->cur_bytes.replace,
                                      &st->cur_bytes.replace_len)) {
                st->cur_bytes_have_replace = true;
            }
        }
        break;
    case SEC_PATCH_PATTERN:
        if (!strcmp(key, "kind")) {
            /* allow re-classification away from pattern */
            if (!strcmp(val, "shader")) {
                reset_cur_pattern(st);
                st->section = SEC_PATCH_SHADER;
            } else if (!strcmp(val, "bytes")) {
                reset_cur_pattern(st);
                st->section = SEC_PATCH_BYTES;
            }
        } else if (!strcmp(key, "description")) {
            g_free(st->cur_pattern.description);
            st->cur_pattern.description = g_strdup(val);
        } else if (!strcmp(key, "search_pattern")) {
            g_free(st->cur_pattern.pattern);
            g_free(st->cur_pattern.pattern_mask);
            st->cur_pattern.pattern = NULL;
            st->cur_pattern.pattern_mask = NULL;
            st->cur_pattern.pattern_len = 0;
            if (parse_pattern_string(val,
                                     &st->cur_pattern.pattern,
                                     &st->cur_pattern.pattern_mask,
                                     &st->cur_pattern.pattern_len)) {
                st->cur_pattern_have_signature = true;
            }
        } else if (!strcmp(key, "search_start")) {
            parse_u32(val, &st->cur_pattern.search_start);
        } else if (!strcmp(key, "search_end")) {
            parse_u32(val, &st->cur_pattern.search_end);
        } else if (!strcmp(key, "search_offset")) {
            parse_i32(val, &st->cur_pattern.search_offset);
        } else if (!strcmp(key, "expected")) {
            g_free(st->cur_pattern.expected);
            st->cur_pattern.expected = NULL;
            st->cur_pattern.expected_len = 0;
            parse_hex_byte_string(val, &st->cur_pattern.expected,
                                  &st->cur_pattern.expected_len);
        } else if (!strcmp(key, "replace")) {
            g_free(st->cur_pattern.replace);
            st->cur_pattern.replace = NULL;
            st->cur_pattern.replace_len = 0;
            if (parse_hex_byte_string(val, &st->cur_pattern.replace,
                                      &st->cur_pattern.replace_len)) {
                st->cur_pattern_have_replace = true;
            }
        }
        break;
    case SEC_PATCH_SHADER:
        if (!strcmp(key, "glsl_hash")) {
            parse_u64_hex(val, &st->cur_shader.glsl_hash);
            st->cur_shader_have = true;
        } else if (!strcmp(key, "file")) {
            g_free(st->cur_shader.file_abs);
            st->cur_shader.file_abs =
                g_build_filename(st->pack->dir_abs, val, NULL);
        }
        break;
    default:
        break;
    }

    g_free(val);
}

static XPack *load_pack_from_dir(const char *title_id_dir,
                                 const char *pack_dir_name)
{
    char *manifest_path = g_build_filename(title_id_dir, pack_dir_name,
                                           "pack.toml", NULL);
    gchar *contents = NULL;
    gsize len = 0;
    if (!g_file_get_contents(manifest_path, &contents, &len, NULL)) {
        g_free(manifest_path);
        return NULL;
    }

    XPack *pack = g_new0(XPack, 1);
    const char *title_id_basename = g_path_get_basename(title_id_dir);
    pack->id = g_strdup_printf("%s/%s", title_id_basename, pack_dir_name);
    g_free((char *)title_id_basename);
    pack->dir_abs = g_build_filename(title_id_dir, pack_dir_name, NULL);
    pack->bytes_patches   = g_array_new(FALSE, FALSE, sizeof(BytesPatch));
    pack->pattern_patches = g_array_new(FALSE, FALSE, sizeof(PatternPatch));
    pack->shader_ovr      = g_array_new(FALSE, FALSE, sizeof(ShaderOverride));

    ParseState st = { 0 };
    st.pack = pack;
    st.section = SEC_ROOT;

    gchar **lines = g_strsplit(contents, "\n", -1);
    for (gchar **p = lines; *p; p++) {
        parse_manifest_line(&st, *p);
    }
    flush_current_patch(&st);
    g_strfreev(lines);

    g_free(contents);
    g_free(manifest_path);
    return pack;
}

static void free_pack(XPack *pack)
{
    if (!pack) return;
    for (guint i = 0; i < pack->bytes_patches->len; i++) {
        BytesPatch *bp = &g_array_index(pack->bytes_patches, BytesPatch, i);
        g_free(bp->description);
        g_free(bp->expected);
        g_free(bp->replace);
    }
    g_array_free(pack->bytes_patches, TRUE);
    for (guint i = 0; i < pack->pattern_patches->len; i++) {
        PatternPatch *pp =
            &g_array_index(pack->pattern_patches, PatternPatch, i);
        g_free(pp->description);
        g_free(pp->pattern);
        g_free(pp->pattern_mask);
        g_free(pp->expected);
        g_free(pp->replace);
    }
    g_array_free(pack->pattern_patches, TRUE);
    for (guint i = 0; i < pack->shader_ovr->len; i++) {
        ShaderOverride *so = &g_array_index(pack->shader_ovr, ShaderOverride, i);
        g_free(so->file_abs);
    }
    g_array_free(pack->shader_ovr, TRUE);
    if (pack->match_xbe_sha1) {
        g_ptr_array_free(pack->match_xbe_sha1, TRUE);
    }
    g_free(pack->id);
    g_free(pack->dir_abs);
    g_free(pack->name);
    g_free(pack);
}

/* ---------- discovery ---------- */

static void discover_packs(uint32_t title_id)
{
    if (g_packs) {
        for (guint i = 0; i < g_packs->len; i++) {
            free_pack(g_array_index(g_packs, XPack *, i));
        }
        g_array_free(g_packs, TRUE);
    }
    g_packs = g_array_new(FALSE, FALSE, sizeof(XPack *));

    char *root = xpacks_root_dir();
    char title_id_str[16];
    snprintf(title_id_str, sizeof(title_id_str), "%08X", title_id);
    char *title_dir = g_build_filename(root, title_id_str, NULL);
    g_free(root);

    GDir *d = g_dir_open(title_dir, 0, NULL);
    if (!d) {
        g_free(title_dir);
        return;
    }

    GHashTable *enabled = load_enabled_set();

    const char *entry;
    while ((entry = g_dir_read_name(d))) {
        if (entry[0] == '.') continue;
        char *full = g_build_filename(title_dir, entry, NULL);
        if (g_file_test(full, G_FILE_TEST_IS_DIR)) {
            XPack *pack = load_pack_from_dir(title_dir, entry);
            if (pack) {
                /* enabled-set key matches "<title_id>/<pack_dir>" */
                if (g_hash_table_contains(enabled, pack->id)) {
                    /* must also match title id */
                    if (pack->match_title_id == 0 ||
                        pack->match_title_id == title_id) {
                        g_array_append_val(g_packs, pack);
                        qemu_log_mask(LOG_GUEST_ERROR,
                            XPACKS_TAG ": loaded pack '%s' (%s) "
                            "[%u bytes, %u pattern, %u shader]\n",
                            pack->id, pack->name ? pack->name : "",
                            pack->bytes_patches->len,
                            pack->pattern_patches->len,
                            pack->shader_ovr->len);
                    } else {
                        qemu_log_mask(LOG_GUEST_ERROR,
                            XPACKS_TAG ": skipping '%s' — match.title_id "
                            "0x%08X != game 0x%08X\n",
                            pack->id, pack->match_title_id, title_id);
                        free_pack(pack);
                    }
                } else {
                    free_pack(pack);
                }
            }
        }
        g_free(full);
    }

    g_dir_close(d);
    g_hash_table_destroy(enabled);
    g_free(title_dir);
}

/* ---------- SHA1 build matching ----------
 *
 * Hash the 256-byte digital-signature field of the xbe header. Each retail
 * build is signed independently, so this fingerprint is uniquely deterministic
 * per build without us needing to know section-table internals. */
static bool sha1_matches(const XPack *pack, const struct xbe *xbe)
{
    if (!pack->match_xbe_sha1 || pack->match_xbe_sha1->len == 0) {
        return true; /* no constraint */
    }
    GChecksum *cs = g_checksum_new(G_CHECKSUM_SHA1);
    g_checksum_update(cs, xbe->header->m_digsig, sizeof(xbe->header->m_digsig));
    const char *digest = g_checksum_get_string(cs); /* lowercase hex */
    bool ok = false;
    for (guint i = 0; i < pack->match_xbe_sha1->len; i++) {
        const char *want = g_ptr_array_index(pack->match_xbe_sha1, i);
        if (!g_ascii_strcasecmp(digest, want)) { ok = true; break; }
    }
    if (!ok) {
        qemu_log_mask(LOG_GUEST_ERROR,
            XPACKS_TAG ": [%s] xbe_code_sha1 mismatch — digsig=%s\n",
            pack->id, digest);
    }
    g_checksum_free(cs);
    return ok;
}

/* ---------- pattern scanning ---------- */

static bool pattern_match_at(const uint8_t *haystack, const PatternPatch *pp)
{
    for (size_t i = 0; i < pp->pattern_len; i++) {
        if (pp->pattern_mask[i] && haystack[i] != pp->pattern[i]) {
            return false;
        }
    }
    return true;
}

/* Scan one pattern over its [search_start, search_end) range. Returns 1 if
 * applied this call, 0 otherwise. Page-fault tolerant: unmapped pages are
 * skipped without aborting the scan. */
static int pattern_apply_one(const XPack *pack, PatternPatch *pp)
{
    if (pp->applied) return 0;
    if (pp->pattern_len == 0 || pp->replace_len == 0) return 0;
    if (pp->search_end <= pp->search_start) return 0;

    const size_t CHUNK = 64 * 1024;
    size_t overlap = pp->pattern_len - 1;
    uint8_t *buf = g_malloc(CHUNK + overlap);

    uint32_t addr = pp->search_start;
    while (addr + pp->pattern_len <= pp->search_end) {
        uint32_t remaining = pp->search_end - addr;
        size_t want = CHUNK + overlap;
        if ((size_t)remaining < want) want = remaining;

        if (!guest_read(addr, buf, want)) {
            /* Unmapped — advance to next page and retry. */
            uint32_t next_page =
                (addr + (uint32_t)TARGET_PAGE_SIZE) & ~((uint32_t)TARGET_PAGE_SIZE - 1);
            if (next_page <= addr) break;  /* overflow */
            addr = next_page;
            continue;
        }

        size_t scan_max = (want >= pp->pattern_len) ? (want - pp->pattern_len) : 0;
        for (size_t i = 0; i <= scan_max; i++) {
            if (!pattern_match_at(buf + i, pp)) continue;
            uint32_t hit = addr + (uint32_t)i;
            uint32_t patch_addr = hit + (uint32_t)pp->search_offset;

            if (pp->expected_len) {
                uint8_t *cur = g_malloc(pp->expected_len);
                if (!guest_read(patch_addr, cur, pp->expected_len) ||
                    memcmp(cur, pp->expected, pp->expected_len) != 0) {
                    g_free(cur);
                    continue;
                }
                g_free(cur);
            }
            if (!guest_write(patch_addr, pp->replace, pp->replace_len)) {
                continue;
            }
            pp->applied = true;
            qemu_log_mask(LOG_GUEST_ERROR,
                XPACKS_TAG ": [%s] pattern matched @ 0x%08X — wrote %zu @ 0x%08X "
                "(%s)\n",
                pack->id, hit, pp->replace_len, patch_addr,
                pp->description ? pp->description : "(no description)");
            g_free(buf);
            return 1;
        }

        if (addr > UINT32_MAX - (uint32_t)CHUNK) break;
        addr += (uint32_t)CHUNK;
    }
    g_free(buf);
    return 0;
}

/* ---------- apply ---------- */

static int apply_bytes_patch(const XPack *pack, const BytesPatch *bp)
{
    if (!bp->replace_len) return 0;

    if (bp->expected_len) {
        uint8_t *buf = g_malloc(bp->expected_len);
        if (!guest_read(bp->address, buf, bp->expected_len)) {
            qemu_log_mask(LOG_GUEST_ERROR,
                XPACKS_TAG ": [%s] read failed at 0x%08X\n",
                pack->id, bp->address);
            g_free(buf);
            return 0;
        }
        if (memcmp(buf, bp->expected, bp->expected_len) != 0) {
            qemu_log_mask(LOG_GUEST_ERROR,
                XPACKS_TAG ": [%s] expected-mismatch at 0x%08X — "
                "refusing patch '%s'\n",
                pack->id, bp->address,
                bp->description ? bp->description : "(no description)");
            g_free(buf);
            return 0;
        }
        g_free(buf);
    }

    if (!guest_write(bp->address, bp->replace, bp->replace_len)) {
        qemu_log_mask(LOG_GUEST_ERROR,
            XPACKS_TAG ": [%s] write failed at 0x%08X\n",
            pack->id, bp->address);
        return 0;
    }

    qemu_log_mask(LOG_GUEST_ERROR,
        XPACKS_TAG ": [%s] applied %zu bytes at 0x%08X — %s\n",
        pack->id, bp->replace_len, bp->address,
        bp->description ? bp->description : "(no description)");
    return 1;
}

int xpacks_apply_for_xbe(const struct xbe *xbe)
{
    if (!xbe || !xbe->cert) return 0;
    uint32_t title_id = xbe->cert->m_titleid;

    if (g_applied_title_id == title_id) {
        return 0;
    }

    discover_packs(title_id);
    if (!g_packs || g_packs->len == 0) {
        g_applied_title_id = title_id;
        return 0;
    }

    int total = 0;
    int pending_pattern = 0;
    for (guint i = 0; i < g_packs->len; i++) {
        XPack *pack = g_array_index(g_packs, XPack *, i);

        if (!sha1_matches(pack, xbe)) {
            /* Build mismatch — disqualify all of this pack's patches. */
            for (guint j = 0; j < pack->pattern_patches->len; j++) {
                PatternPatch *pp =
                    &g_array_index(pack->pattern_patches, PatternPatch, j);
                pp->applied = true; /* skip */
            }
            continue;
        }

        for (guint j = 0; j < pack->bytes_patches->len; j++) {
            BytesPatch *bp = &g_array_index(pack->bytes_patches, BytesPatch, j);
            total += apply_bytes_patch(pack, bp);
        }
        pending_pattern += pack->pattern_patches->len;
    }

    g_applied_title_id = title_id;
    g_pending_patterns = pending_pattern;
    g_inited = true;
    return total;
}

int xpacks_tick(void)
{
    if (g_pending_patterns <= 0 || !g_packs) return 0;

    /* Throttle the scan: only re-attempt every N calls. Renderer is calling
     * us at ~30 Hz; once per second is plenty for tag-data overlays which
     * load on level boundaries. */
    static int countdown = 0;
    if (countdown > 0) { countdown--; return 0; }
    countdown = 30;

    int applied = 0;
    int still_pending = 0;
    for (guint i = 0; i < g_packs->len; i++) {
        XPack *pack = g_array_index(g_packs, XPack *, i);
        for (guint j = 0; j < pack->pattern_patches->len; j++) {
            PatternPatch *pp =
                &g_array_index(pack->pattern_patches, PatternPatch, j);
            if (pp->applied) continue;
            int r = pattern_apply_one(pack, pp);
            applied += r;
            if (!pp->applied) still_pending++;
        }
    }
    g_pending_patterns = still_pending;
    return applied;
}

/* ---------- shader override ---------- */

void *xpacks_lookup_spirv(uint64_t glsl_hash, size_t *out_len)
{
    if (!g_packs) return NULL;
    for (guint i = 0; i < g_packs->len; i++) {
        XPack *pack = g_array_index(g_packs, XPack *, i);
        for (guint j = 0; j < pack->shader_ovr->len; j++) {
            ShaderOverride *so =
                &g_array_index(pack->shader_ovr, ShaderOverride, j);
            if (so->glsl_hash != glsl_hash) continue;
            gchar *data = NULL;
            gsize len = 0;
            if (!g_file_get_contents(so->file_abs, &data, &len, NULL)) {
                qemu_log_mask(LOG_GUEST_ERROR,
                    XPACKS_TAG ": [%s] shader override read failed: %s\n",
                    pack->id, so->file_abs);
                continue;
            }
            if (len < 4) {
                g_free(data);
                continue;
            }
            void *buf = g_malloc(len);
            memcpy(buf, data, len);
            g_free(data);
            *out_len = (size_t)len;
            qemu_log_mask(LOG_GUEST_ERROR,
                XPACKS_TAG ": [%s] shader override hit hash %016" PRIx64
                " (%zu bytes)\n", pack->id, glsl_hash, *out_len);
            return buf;
        }
    }
    return NULL;
}

void xpacks_reset(void)
{
    if (g_packs) {
        for (guint i = 0; i < g_packs->len; i++) {
            free_pack(g_array_index(g_packs, XPack *, i));
        }
        g_array_free(g_packs, TRUE);
        g_packs = NULL;
    }
    g_applied_title_id = 0;
    g_pending_patterns = 0;
    g_inited = false;
}
