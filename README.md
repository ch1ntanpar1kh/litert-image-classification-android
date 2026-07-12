# LiteRT V2 Android Image Classification Sample

An Android application demonstrating image classification on live camera feeds and gallery photos using Google AI Edge's **LiteRT V2 CompiledModel API** with NPU, GPU, and CPU hardware acceleration.

---

## 🚀 Key Features

*   **LiteRT V2 CompiledModel API**: Built using `com.google.ai.edge.litert:litert` (`2.1.6`), replacing legacy TensorFlow Lite `Interpreter` APIs.
*   **Dynamic Hardware Acceleration Fallback Cascade**: Automatically initializes and attempts acceleration in the following fallback order:
    1.  **NPU Acceleration (JIT Execution)**: Leverages Qualcomm Hexagon / Google Tensor NPU accelerators with High Performance mode (`HTPPerformanceMode.HIGH_PERFORMANCE`).
    2.  **GPU Acceleration**: Falls back to GPU execution if NPU drivers are unavailable on the device.
    3.  **CPU Multithreading**: Default fallback to multi-threaded CPU execution.
*   **Automatic INT8 Quantized Model Loading**: Automatically detects and loads INT8 quantized model assets (`*_int8.tflite`) on NPU hardware for sub-millisecond execution.
*   **Zero-Copy Native Image Preprocessing**: Replaced legacy `org.tensorflow.lite.support` libraries with direct Kotlin `Bitmap` matrix transformations and `FloatArray` normalization.
*   **Dynamic Metadata Extraction**: Native `ZipInputStream` parser to extract class labels (`labels.txt`) directly from the `.tflite` model package.
*   **Hexagon DSP FastRPC Support**: Pre-configured with `<uses-native-library android:name="libcdsprpc.so" android:required="false" />` in `AndroidManifest.xml` for Android 12+ DSP channel binding.

---

## 🛠️ Prerequisites & Requirements

*   **Android Studio**: Android Studio Ladybug (2024.2.1+) or newer.
*   **JDK Version**: Java 17 or Java 21 (configured in Android Studio under **Settings > Build, Execution, Deployment > Build Tools > Gradle > Gradle JDK**).
*   **Target Android SDK**: API Level 34 (Android 14).
*   **Minimum Android SDK**: API Level 24 (Android 7.0 Nougat).
*   **Physical Device / Emulator**: Android device running API level 24 or higher (Physical device with NPU/GPU recommended for testing hardware acceleration).

---

## 📱 Running & Testing in Android Studio

### 1. Open the Project in Android Studio
1. Launch **Android Studio**.
2. Click **Open** and select the root project folder:
   ```text
   interpreter_api/image_classification/android
   ```
3. Allow Android Studio to complete Gradle sync.

### 2. Run the Application
1. Connect your Android physical device via USB (with **USB Debugging** enabled) or start an Android Virtual Device (AVD) emulator.
2. Select the **`app`** configuration in the top toolbar.
3. Click the green **Run ▶** button (or press `Shift + F10`).

### 3. Run the Automated Phase 2 Migration Tests
This project includes a dedicated instrumentation test suite ([`MigrationValidationTest.kt`](file:///google/src/cloud/chintanparikh/karpathy-litert-migration-skill/google3/litert-samples/interpreter_api/image_classification/android/app/src/androidTest/java/com/google/aiedge/examples/imageclassification/MigrationValidationTest.kt)) that validates `CompiledModel` creation, zero-copy `ByteBuffer` read/write operations, and NPU `Environment` configuration.

To execute the tests in Android Studio:
1. In the Project tool window, navigate to `app/src/androidTest/java/com/google/aiedge/examples/imageclassification/`.
2. Right-click **`MigrationValidationTest`**.
3. Click **Run 'MigrationValidationTest'**.

Or run via terminal:
```bash
./gradlew assembleDebug assembleDebugAndroidTest
adb shell am instrument -w -r -e class com.google.aiedge.examples.imageclassification.MigrationValidationTest com.google.aiedge.examples.imageclassification.test/androidx.test.runner.AndroidJUnitRunner
```

---

## 🔄 Summary of Migration to LiteRT V2

This codebase was migrated from legacy TensorFlow Lite (TFLite V1) to LiteRT V2.

| Legacy TFLite API | Modern LiteRT V2 API |
|---|---|
| `org.tensorflow.lite.Interpreter` | `com.google.ai.edge.litert.CompiledModel` |
| `org.tensorflow.lite.gpu.GpuDelegate` | `CompiledModel.Options(Accelerator.GPU)` |
| `org.tensorflow.lite.support.image.ImageProcessor` | Native `Bitmap` & `FloatArray` pixel matrix ops |
| `org.tensorflow.lite.support.label.Category` | Direct data class `Category(label, score)` |
| `org.tensorflow:tensorflow-lite:*` dependencies | `com.google.ai.edge.litert:litert:2.1.6` |

---

## 🐙 Pushing to your GitHub Repository

To push this migrated app to your GitHub profile ([ch1ntanpar1kh](https://github.com/ch1ntanpar1kh)):

1. Open your terminal in this project directory:
   ```bash
   cd interpreter_api/image_classification/android
   ```
2. Initialize Git (if not already initialized) and make a commit:
   ```bash
   git init
   git add .
   git commit -m "Migrate Android Image Classification sample to LiteRT V2 CompiledModel API"
   ```
3. Create a new repository on GitHub (e.g., `litert-image-classification-android`) at `https://github.com/new`.
4. Link your remote and push:
   ```bash
   git branch -M main
   git remote add origin git@github.com:ch1ntanpar1kh/litert-image-classification-android.git
   git push -u origin main
   ```
