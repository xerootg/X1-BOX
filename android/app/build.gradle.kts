import java.util.Properties
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
  id("com.android.application")
  id("org.jetbrains.kotlin.android")
}

val keystorePropertiesFile = rootProject.file("key.properties")
val keystoreProperties = Properties()
val hasKeystoreProperties = keystorePropertiesFile.exists()

if (hasKeystoreProperties) {
  keystorePropertiesFile.inputStream().use { keystoreProperties.load(it) }
}

val hasReleaseKeystore = hasKeystoreProperties &&
  listOf("storeFile", "storePassword", "keyAlias", "keyPassword").all {
    !keystoreProperties.getProperty(it).isNullOrBlank()
  }

android {
  namespace = "com.izzy2lost.x1box"
  compileSdk = 36
  buildToolsVersion = "37.0.0"
  ndkVersion = "29.0.14206865"

  defaultConfig {
    applicationId = "com.izzy2lost.x1box"
    minSdk = 26
    targetSdk = 36

    versionCode = 26
    versionName = "1.2.5"

    ndk {
      abiFilters += listOf("arm64-v8a")
    }

    externalNativeBuild {
      cmake {
        /*
         * Per-variant optimization split:
         *   DEBUG / perftest   -O2           less aggressive inlining + loop
         *                                    transforms; symbols + call sites
         *                                    stay close to source for
         *                                    profiling / simpleperf / lldb.
         *                                    perftest inherits DEBUG flags via
         *                                    matchingFallbacks so its codegen
         *                                    stays identical to debug — the
         *                                    whole point of that variant.
         *   RELWITHDEBINFO     -O3 + unroll  shipping codegen with debug-info.
         *   RELEASE            -O3 + unroll  shipping codegen + LTO opt-in.
         *
         * Deliberately NOT enabled anywhere:
         *   -ffast-math / -fno-signed-zeros / -freciprocal-math: would break
         *     the bit-exact NaN/denorm/rounding semantics that the x87 lib
         *     and softfloat code rely on (same reason x87/CMakeLists guard
         *     against these). SSE-scalar inline emit also depends on IEEE
         *     ordering — see project_sse_scalar_inline_gated memory.
         *   -fno-math-errno: technically a small win but breaks some libm
         *     callers that observe errno (the samplerate stub does).
         *   LTO: release variant opts in via -DXEMU_ENABLE_LTO=ON (consumed
         *     by the option() + check_ipo_supported() probe in cpp/CMakeLists.txt,
         *     applied per-target on libxemu.so and its first-party static deps).
         *     Adding it to debug or perftest would change inlining and muddy
         *     the debug/perftest codegen-parity contract.
         */
        arguments += listOf(
          "-DXEMU_ANDROID_BUILD_ID=3",
          "-DXEMU_ENABLE_XISO_CONVERTER=ON",
          "-DCMAKE_C_FLAGS_DEBUG=-O2 -g0 -fvisibility=hidden",
          "-DCMAKE_CXX_FLAGS_DEBUG=-O2 -g0 -fvisibility=hidden -fvisibility-inlines-hidden",
          "-DCMAKE_C_FLAGS_RELWITHDEBINFO=-O3 -funroll-loops -g0 -fvisibility=hidden",
          "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O3 -funroll-loops -g0 -fvisibility=hidden",
          "-DCMAKE_C_FLAGS_RELEASE=-O3 -funroll-loops -g0 -fvisibility=hidden",
          "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -funroll-loops -g0 -fvisibility=hidden",
        )
        cppFlags += listOf("-std=c++17", "-fexceptions", "-frtti")
      }
    }
  }

  signingConfigs {
    if (hasReleaseKeystore) {
      create("release") {
        storeFile = file(keystoreProperties.getProperty("storeFile"))
        storePassword = keystoreProperties.getProperty("storePassword")
        keyAlias = keystoreProperties.getProperty("keyAlias")
        keyPassword = keystoreProperties.getProperty("keyPassword")
      }
    }
  }

  buildTypes {
    debug {
      ndk {
        debugSymbolLevel = "NONE"
      }
    }
    /*
     * perftest variant — sits side-by-side with the standard debug install
     * (different applicationId via the ".perftest" suffix and its own data
     * dir under /data/data/com.izzy2lost.x1box.perftest), so we can A/B the
     * debug-logging-on build against a debug-logging-off build without
     * uninstalling the original. Same NDK flags as `debug` so the
     * libxemu.so codegen is identical (lets us isolate runtime-config /
     * sched-debug-log overhead from compile-flag differences). `isDebuggable
     * = true` keeps simpleperf / run-as / lldb-server attachable; pair with
     * a sched_config.txt that omits the X1BOX_*_DEBUG=1 flags to actually
     * measure the no-logging steady state.
     *
     * Build with: `./gradlew assemblePerftest`
     * Install with: `adb install -r app/build/outputs/apk/perftest/app-perftest.apk`
     */
    create("perftest") {
      isDebuggable = true
      applicationIdSuffix = ".perftest"
      versionNameSuffix = "-perftest"
      ndk {
        debugSymbolLevel = "NONE"
      }
      matchingFallbacks += listOf("debug")
      /* Sign with the debug keystore so `adb install` works without
       * needing key.properties. Distinct from the release variant which
       * uses the project's release keystore when available. */
      signingConfig = signingConfigs.getByName("debug")
    }
    release {
      externalNativeBuild {
        cmake {
          arguments += listOf("-DXEMU_ENABLE_LTO=ON")
        }
      }
      isMinifyEnabled = false
      proguardFiles(
        getDefaultProguardFile("proguard-android-optimize.txt"),
        "proguard-rules.pro"
      )
      /*
       * Distinct applicationId so the release APK installs side-by-side with
       * the standard debug install (com.izzy2lost.x1box) and the perftest
       * install (com.izzy2lost.x1box.perftest). This lets us A/B all three
       * codegen variants on one device without uninstalling.
       *
       * Strip the suffix before publishing to the Play Store — a real
       * release should ship as the canonical com.izzy2lost.x1box.
       */
      applicationIdSuffix = ".release"
      versionNameSuffix = "-release"
      if (hasReleaseKeystore) {
        signingConfig = signingConfigs.getByName("release")
      } else {
        /* No release keystore on this machine — fall back to the debug
         * keystore so the unsigned-APK install failure path doesn't bite.
         * Production builds for the Play Store still go through the
         * hasReleaseKeystore path; this only kicks in for local dev. */
        signingConfig = signingConfigs.getByName("debug")
      }
    }
  }

  externalNativeBuild {
    cmake {
      path = file("src/main/cpp/CMakeLists.txt")
      version = "3.30.3"
    }
  }

  packaging {
    resources.excludes += setOf(
      "**/*.md",
      "META-INF/LICENSE*",
      "META-INF/NOTICE*"
    )
    /* Extract .so to disk (nativeLibraryDir); required for adrenotools hooks / custom GPU drivers. */
    jniLibs.useLegacyPackaging = true
    jniLibs.keepDebugSymbols += setOf("**/*.so")
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }

}

dependencies {
  implementation("androidx.core:core-ktx:1.15.0")
  implementation("androidx.appcompat:appcompat:1.7.0")
  implementation("androidx.constraintlayout:constraintlayout:2.1.4")
  implementation("androidx.documentfile:documentfile:1.0.1")
  implementation("io.coil-kt:coil:2.7.0")
  implementation("com.google.android.material:material:1.12.0")
}

kotlin {
  compilerOptions {
    jvmTarget.set(JvmTarget.JVM_17)
  }
}
