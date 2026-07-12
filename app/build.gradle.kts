plugins {
    alias(libs.plugins.androidApplication)
    alias(libs.plugins.jetbrainsKotlinAndroid)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.undercouchDownload)
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_1_8)
    }
}

android {
    namespace = "com.google.aiedge.examples.imageclassification"
    compileSdk = 30

    defaultConfig {
        applicationId = "com.google.aiedge.examples.imageclassification"
        minSdk = 24
        targetSdk = 30
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    buildFeatures {
        compose = true
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("src/main/jniLibs")
        }
    }
}

// Import DownloadModels task
project.ext.set("ASSET_DIR", "$projectDir/src/main/assets")
apply(from = "download_model.gradle")

configurations.all {
    resolutionStrategy {
        force("androidx.lifecycle:lifecycle-runtime:2.7.0")
        force("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
        force("androidx.lifecycle:lifecycle-runtime-compose:2.7.0")
        force("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")
        force("androidx.compose.ui:ui:1.5.0")
        force("androidx.compose.ui:ui-graphics:1.5.0")
        force("androidx.compose.ui:ui-tooling:1.5.0")
        force("androidx.compose.ui:ui-tooling-preview:1.5.0")
        force("androidx.compose.runtime:runtime:1.5.0")
        force("androidx.compose.runtime:runtime-saveable:1.5.0")
        force("androidx.compose.foundation:foundation:1.5.0")
        force("androidx.compose.foundation:foundation-layout:1.5.0")
        force("androidx.compose.animation:animation:1.5.0")
        force("androidx.compose.animation:animation-core:1.5.0")
    }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material2)
    implementation(libs.litert)
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.coil.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.ui.test.junit4)
    debugImplementation(libs.androidx.ui.tooling)
    debugImplementation(libs.androidx.ui.test.manifest)
}