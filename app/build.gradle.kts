plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

// MediaPipe ждёт ковариантный Any.build():Any (есть только в ПОЛНОЙ protobuf-java, не в javalite)
configurations.all {
    exclude(group = "com.google.protobuf", module = "protobuf-lite")
    exclude(group = "com.google.protobuf", module = "protobuf-javalite")
    resolutionStrategy { force("com.google.protobuf:protobuf-java:4.26.1") }
}

android {
    namespace = "com.example.generet_image_ai"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "com.example.generet_image_ai"
        minSdk = 31
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk { abiFilters.add("arm64-v8a") }   // NPU только arm64
    }

    // Без этого нативный NPU-dispatch (.so из dynamic-feature) не извлекается →
    // "No usable Dispatch runtime found".
    packaging { jniLibs { useLegacyPackaging = true } }

    buildTypes {
        release {
            optimization {
                enable = false
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        compose = true
        aidl = true
    }

    // Собственный Vulkan compute backend (vk_bench.cpp)
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt") } }

    // NPU-рантайм Google Tensor как dynamic feature
    dynamicFeatures.add(":litert_npu_runtime_libraries:google_tensor_runtime")

    bundle {
        deviceTargetingConfig = file("device_targeting_configuration.xml")
        deviceGroup {
            enableSplit = true
            defaultGroup = "other"
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    debugImplementation(libs.androidx.ui.tooling)

    implementation(libs.kotlinx.coroutines.android)

    // On-device inference (LiteRT Next 2.1.5 — CompiledModel, GPU встроен, NPU-провайдер)
    implementation(libs.litert)
    implementation(project(":litert_npu_runtime_libraries:runtime_strings"))
    implementation("com.google.android.play:feature-delivery:2.1.0")

    // MediaPipe Image Generator — нативный GPU-движок SD1.5 (бенчмарк ~15с/20шагов)
    implementation("com.google.mediapipe:tasks-vision-image-generator:0.10.26.1")
    // полная protobuf-java (ковариантный Any.build():Any для MediaPipe)
    implementation("com.google.protobuf:protobuf-java:4.26.1")

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}
