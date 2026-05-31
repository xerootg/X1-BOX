package com.izzy2lost.x1box

import android.content.Context
import java.io.File
import java.util.Locale

/**
 * Discovery and enable-state management for xpacks (runtime patch bundles).
 *
 * On-disk layout under <externalFiles>/xpacks/:
 *
 *   xpacks/
 *     enabled.txt                       # newline list of "<title_id>/<pack_dir>"
 *     <TITLE_ID_HEX>/                   # e.g. "4D530064"
 *       <pack_dir>/
 *         pack.toml                     # manifest (parsed below)
 *         patches/...                   # optional payloads
 *         shaders/<hash>.spv            # optional SPIR-V overrides
 *
 * The native side (hw/xbox/xpacks.c) reads enabled.txt at xbe-detect time
 * and applies all enabled packs whose match.title_id equals the running game.
 */
object XPackManager {

    data class Pack(
        val id: String,             // "<title_id>/<pack_dir>"
        val titleIdHex: String,     // "4D530064"
        val dir: File,
        val name: String,
        val description: String,
        val bytesPatchCount: Int,
        val patternPatchCount: Int,
        val cavePatchCount: Int,
        val shaderOverrideCount: Int,
    )

    fun rootDir(context: Context): File {
        val base = context.getExternalFilesDir(null) ?: context.filesDir
        val root = File(base, "xpacks")
        if (!root.exists()) root.mkdirs()
        return root
    }

    /**
     * Copy bundled xpacks from APK assets/xpacks/ into <external>/xpacks/.
     * Only copies files that don't already exist, so user edits are preserved.
     * Safe to call on every launch — cheap and idempotent.
     */
    fun installBundledPacks(context: Context) {
        val am = context.assets
        val root = rootDir(context)
        copyAssetTree(am, "xpacks", root)
    }

    private fun copyAssetTree(am: android.content.res.AssetManager, assetPath: String, destDir: File) {
        val children = try { am.list(assetPath) } catch (_: Throwable) { null } ?: return
        if (children.isEmpty()) {
            // It's a file — copy if not present.
            val out = File(destDir.parentFile, destDir.name)
            if (!out.exists()) {
                out.parentFile?.mkdirs()
                am.open(assetPath).use { input ->
                    out.outputStream().use { output -> input.copyTo(output) }
                }
            }
            return
        }
        destDir.mkdirs()
        for (child in children) {
            val childAsset = "$assetPath/$child"
            val grandchildren = try { am.list(childAsset) } catch (_: Throwable) { null }
            if (grandchildren == null || grandchildren.isEmpty()) {
                // file — but assets.list returns [] for both files AND empty dirs; try open
                val out = File(destDir, child)
                if (!out.exists()) {
                    try {
                        am.open(childAsset).use { input ->
                            out.outputStream().use { output -> input.copyTo(output) }
                        }
                    } catch (_: Throwable) {
                        // empty directory in assets — create it
                        out.mkdirs()
                    }
                }
            } else {
                copyAssetTree(am, childAsset, File(destDir, child))
            }
        }
    }

    /** Scan disk and return all discovered packs. Filesystem is the source of truth. */
    fun discover(context: Context): List<Pack> {
        val root = rootDir(context)
        val packs = mutableListOf<Pack>()
        val titleDirs = root.listFiles { f -> f.isDirectory && f.name.matches(Regex("[0-9A-Fa-f]{8}")) }
            ?: return emptyList()
        for (titleDir in titleDirs) {
            val titleIdHex = titleDir.name.uppercase(Locale.ROOT)
            val packDirs = titleDir.listFiles { f -> f.isDirectory } ?: continue
            for (packDir in packDirs) {
                val manifest = File(packDir, "pack.toml")
                if (!manifest.isFile) continue
                packs.add(parseManifest(manifest, packDir, titleIdHex))
            }
        }
        packs.sortBy { it.id }
        return packs
    }

    /** Returns the set of pack ids ("<title_id>/<pack_dir>") that are currently enabled. */
    fun loadEnabled(context: Context): Set<String> {
        val file = File(rootDir(context), "enabled.txt")
        if (!file.isFile) return emptySet()
        return file.readLines()
            .map { it.trim() }
            .filter { it.isNotEmpty() && !it.startsWith("#") }
            .toSet()
    }

    /** Persist the enabled-set as enabled.txt. */
    fun saveEnabled(context: Context, ids: Set<String>) {
        val file = File(rootDir(context), "enabled.txt")
        file.parentFile?.mkdirs()
        file.writeText(
            buildString {
                append("# xpacks enabled list — one '<title_id>/<pack_dir>' per line\n")
                ids.sorted().forEach { append(it).append('\n') }
            }
        )
    }

    /** Toggle a pack and persist. Returns the new enabled-set. */
    fun toggle(context: Context, id: String): Set<String> {
        val cur = loadEnabled(context).toMutableSet()
        if (!cur.add(id)) cur.remove(id)
        saveEnabled(context, cur)
        return cur
    }

    /**
     * Minimal TOML-ish parser — mirrors hw/xbox/xpacks.c's parser. Only the
     * fields we surface in the UI are extracted; the C side re-parses for
     * authoritative apply.
     */
    private fun parseManifest(file: File, packDir: File, titleIdHex: String): Pack {
        var name = packDir.name
        var description = ""
        var bytesCount = 0
        var patternCount = 0
        var caveCount = 0
        var shaderCount = 0
        var section = "root"
        var currentKind = "bytes"
        var sawPatch = false

        fun finishPatch() {
            if (!sawPatch) return
            when (currentKind) {
                "bytes" -> bytesCount++
                "pattern_bytes" -> patternCount++
                "cave" -> caveCount++
                "shader" -> shaderCount++
            }
            sawPatch = false
            currentKind = "bytes"
        }

        for (rawLine in file.readLines()) {
            var line = rawLine
            val hash = line.indexOf('#')
            if (hash >= 0) line = line.substring(0, hash)
            line = line.trim()
            if (line.isEmpty()) continue

            if (line.startsWith("[")) {
                finishPatch()
                section = line
                if (line == "[[patch]]") {
                    sawPatch = true
                    currentKind = "bytes"
                }
                continue
            }

            val eq = line.indexOf('=')
            if (eq < 0) continue
            val key = line.substring(0, eq).trim()
            var value = line.substring(eq + 1).trim()
            if (value.startsWith("\"") && value.endsWith("\"") && value.length >= 2) {
                value = value.substring(1, value.length - 1)
            }

            when (section) {
                "[meta]", "root" -> {
                    if (key == "name") name = value
                    else if (key == "description") description = value
                }
                "[[patch]]" -> {
                    if (key == "kind") currentKind = value
                    else if (key == "description" && description.isEmpty()) description = value
                }
            }
        }
        finishPatch()

        return Pack(
            id = "$titleIdHex/${packDir.name}",
            titleIdHex = titleIdHex,
            dir = packDir,
            name = name,
            description = description,
            bytesPatchCount = bytesCount,
            patternPatchCount = patternCount,
            cavePatchCount = caveCount,
            shaderOverrideCount = shaderCount,
        )
    }
}
