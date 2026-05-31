package com.izzy2lost.x1box

import android.os.Bundle
import android.text.TextUtils
import android.view.Gravity
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.card.MaterialCardView
import com.google.android.material.materialswitch.MaterialSwitch

/**
 * "Mods (xpacks)" management screen.
 *
 * - Auto-installs bundled packs from APK assets on every launch (idempotent).
 * - Enumerates <external>/xpacks/<TITLE_ID>/<pack>/pack.toml entries.
 * - Toggling a switch persists to <external>/xpacks/enabled.txt; the native
 *   side reads that file when the guest xbe is first detected.
 *
 * Launch with optional EXTRA_FILTER_TITLE_ID (e.g. "4D530064") to scope the
 * list to one title id; omit to show everything.
 */
class XPackActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_FILTER_TITLE_ID = "com.izzy2lost.x1box.extra.XPACK_FILTER_TITLE_ID"
    }

    private val dpScale by lazy { resources.displayMetrics.density }
    private fun dp(v: Int) = (v * dpScale).toInt()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_xpack)

        // Ensure bundled assets are present before we list.
        XPackManager.installBundledPacks(this)

        val storagePath = findViewById<TextView>(R.id.tv_xpack_storage_path)
        storagePath.text = XPackManager.rootDir(this).absolutePath

        val listContainer = findViewById<LinearLayout>(R.id.xpack_list_container)
        val emptyView     = findViewById<TextView>(R.id.tv_xpack_empty)
        val sectionHeader = findViewById<TextView>(R.id.tv_xpack_section_header)

        val filter = intent.getStringExtra(EXTRA_FILTER_TITLE_ID)?.uppercase()
        val allPacks = XPackManager.discover(this)
        val packs = if (filter != null) {
            allPacks.filter { it.titleIdHex.equals(filter, ignoreCase = true) }
        } else {
            allPacks
        }

        if (packs.isEmpty()) {
            emptyView.visibility = android.view.View.VISIBLE
            sectionHeader.visibility = android.view.View.GONE
            return
        }

        val enabled = XPackManager.loadEnabled(this).toMutableSet()

        // Group by title id so the list stays readable when many games are present.
        val grouped = packs.groupBy { it.titleIdHex }.toSortedMap()
        var firstGroup = true
        for ((titleId, groupPacks) in grouped) {
            if (!firstGroup) {
                listContainer.addView(verticalSpacer(dp(16)))
            }
            firstGroup = false

            listContainer.addView(buildGroupHeader(titleId))
            for ((idx, pack) in groupPacks.withIndex()) {
                if (idx > 0) listContainer.addView(verticalSpacer(dp(10)))
                listContainer.addView(buildPackCard(pack, enabled))
            }
        }
    }

    private fun verticalSpacer(heightPx: Int): android.view.View {
        val v = android.view.View(this)
        v.layoutParams = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, heightPx
        )
        return v
    }

    private fun buildGroupHeader(titleIdHex: String): TextView {
        return TextView(this).apply {
            textSize = 13f
            alpha = 0.8f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            text = "Title ID $titleIdHex"
            setTextColor(resources.getColor(R.color.xemu_green_light, theme))
            setPadding(dp(4), dp(6), 0, dp(6))
        }
    }

    private fun buildPackCard(
        pack: XPackManager.Pack,
        enabled: MutableSet<String>,
    ): MaterialCardView {
        val card = MaterialCardView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
            setCardBackgroundColor(resources.getColor(R.color.xemu_surface_container, theme))
            radius = dp(20).toFloat()
            cardElevation = dp(2).toFloat()
            strokeWidth = dp(1)
            strokeColor = resources.getColor(R.color.xemu_outline_variant, theme)
        }

        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(20), dp(16), dp(16), dp(16))
        }

        val textCol = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f
            )
        }

        textCol.addView(TextView(this).apply {
            textSize = 16f
            text = pack.name
            ellipsize = TextUtils.TruncateAt.END
            maxLines = 2
            setTextColor(resources.getColor(R.color.xemu_text, theme))
        })

        val stats = buildString {
            append(pack.bytesPatchCount).append(" bytes")
            if (pack.patternPatchCount > 0) {
                append(" · ").append(pack.patternPatchCount).append(" pattern")
            }
            if (pack.cavePatchCount > 0) {
                append(" · ").append(pack.cavePatchCount).append(" cave")
            }
            if (pack.shaderOverrideCount > 0) {
                append(" · ").append(pack.shaderOverrideCount).append(" shader")
            }
        }
        textCol.addView(TextView(this).apply {
            textSize = 12f
            text = stats
            setTextColor(resources.getColor(R.color.xemu_text_muted, theme))
            (layoutParams as? LinearLayout.LayoutParams)?.topMargin = dp(2)
        })

        if (pack.description.isNotEmpty()) {
            textCol.addView(TextView(this).apply {
                textSize = 13f
                text = pack.description
                setTextColor(resources.getColor(R.color.xemu_text_muted, theme))
                val lp = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
                )
                lp.topMargin = dp(6)
                layoutParams = lp
            })
        }

        row.addView(textCol)

        val sw = MaterialSwitch(this).apply {
            isChecked = enabled.contains(pack.id)
            setOnCheckedChangeListener { _, isChecked ->
                if (isChecked) enabled.add(pack.id) else enabled.remove(pack.id)
                XPackManager.saveEnabled(this@XPackActivity, enabled)
            }
        }
        val swLp = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        )
        swLp.marginStart = dp(12)
        row.addView(sw, swLp)

        // Tap anywhere on the card toggles the switch.
        card.isClickable = true
        card.isFocusable = true
        card.setOnClickListener { sw.toggle() }

        card.addView(row)
        return card
    }
}
