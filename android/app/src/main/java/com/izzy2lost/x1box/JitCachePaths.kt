package com.izzy2lost.x1box

import android.content.Context
import java.io.File

/**
 * Per-game tier-2 (Cranelift) JIT cache path conventions.
 *
 * Layout: <filesDir>/x1box/jit_cache/<jitCacheKey(relativePath)>/
 *
 * The launcher computes this directory and writes it to a pref the
 * emulator process reads; the library UI uses the same algorithm to
 * resolve the directory for the "Clear CPU JIT Cache" menu action.
 */
object JitCachePaths {
  /**
   * Collapse the per-game relativePath into a filesystem-safe key.
   * ASCII alphanumeric + '_'; capped at 96 chars.
   */
  fun jitCacheKey(relativePath: String): String {
    val sb = StringBuilder()
    for (c in relativePath) {
      if (sb.length >= 96) break
      val ok = (c in 'a'..'z') || (c in 'A'..'Z') || (c in '0'..'9')
      sb.append(if (ok) c else '_')
    }
    return if (sb.isEmpty()) "untitled" else sb.toString()
  }

  fun rootDir(context: Context): File =
    File(File(context.filesDir, "x1box"), "jit_cache")

  fun dirForRelativePath(context: Context, relativePath: String): File =
    File(rootDir(context), jitCacheKey(relativePath))
}
