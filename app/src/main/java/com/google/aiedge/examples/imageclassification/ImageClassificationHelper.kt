/*
 * Copyright 2024 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.aiedge.examples.imageclassification

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.core.graphics.scale
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.zip.ZipInputStream

class ImageClassificationHelper(
    private val context: Context,
    private var options: Options = Options(),
) {
    class Options(
        /** The enum contains the model file name, relative to the assets/ directory */
        var model: Model = DEFAULT_MODEL,
        /** The delegate for running computationally intensive operations*/
        var delegate: Delegate = DEFAULT_DELEGATE,
        /** Number of output classes of the TFLite model.  */
        var resultCount: Int = DEFAULT_RESULT_COUNT,
        /** Probability value above which a class is labeled as active (i.e., detected) the display.  */
        var probabilityThreshold: Float = DEFAULT_THRESHOLD,
    )

    companion object {
        private const val TAG = "ImageClassification"

        val DEFAULT_MODEL = Model.EfficientNetLite0
        val DEFAULT_DELEGATE = Delegate.CPU
        const val DEFAULT_RESULT_COUNT = 3
        const val DEFAULT_THRESHOLD = 0.3f
        const val DEFAULT_THREAD_COUNT = 2
    }

    /** As the result of sound classification, this value emits map of probabilities */
    val classification: SharedFlow<ClassificationResult>
        get() = _classification
    private val _classification = MutableSharedFlow<ClassificationResult>(
        extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    val error: SharedFlow<Throwable?>
        get() = _error
    private val _error = MutableSharedFlow<Throwable?>()

    private var compiledModel: CompiledModel? = null
    private var env: Environment? = null
    private lateinit var labels: List<String>
    private val mutex = Mutex()

    /** Init a CompiledModel from [Model] with [Delegate]*/
    suspend fun initClassifier() {
        mutex.withLock {
            closeModel()
            try {
                withContext(Dispatchers.IO) {
                    labels = loadLabels(options.model.fileName)
                    compiledModel = when (options.delegate) {
                        Delegate.CPU -> {
                            val compOptions = CompiledModel.Options(Accelerator.CPU)
                            CompiledModel.create(context.assets, options.model.fileName, compOptions, null)
                        }
                        Delegate.GPU -> {
                            val compOptions = CompiledModel.Options(Accelerator.GPU)
                            CompiledModel.create(context.assets, options.model.fileName, compOptions, null)
                        }
                        Delegate.NPU -> {
                            createCompiledModelWithNpuCascade(options.model.fileName)
                        }
                    }
                    Log.i(TAG, "Done creating CompiledModel from ${options.model.fileName}")
                }
            } catch (e: Exception) {
                Log.i(TAG, "Create CompiledModel from ${options.model.fileName} failed: ${e.message}")
                _error.emit(e)
            }
        }
    }

    private fun createCompiledModelWithNpuCascade(modelFileName: String): CompiledModel {
        var npuEnv: Environment? = null
        try {
            Log.i(TAG, "Attempting LiteRT NPU JIT compilation...")
            val envOptions = mapOf(
                Environment.Option.DispatchLibraryDir to context.applicationInfo.nativeLibraryDir,
                Environment.Option.CompilerPluginLibraryDir to context.applicationInfo.nativeLibraryDir
            )
            npuEnv = Environment.create(
                BuiltinNpuAcceleratorProvider(context),
                envOptions
            )
            val compOptions = CompiledModel.Options(Accelerator.NPU).apply {
                qualcommOptions = CompiledModel.QualcommOptions(
                    htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.HIGH_PERFORMANCE
                )
            }
            val int8ModelPath = modelFileName.replace(".tflite", "_int8.tflite")
            val targetAsset = try {
                context.assets.open(int8ModelPath).close()
                int8ModelPath
            } catch (_: Exception) {
                modelFileName
            }
            val model = CompiledModel.create(context.assets, targetAsset, compOptions, npuEnv)
            env = npuEnv
            return model
        } catch (_: Exception) {
            Log.w(TAG, "NPU JIT compilation unsupported on this device; falling back to GPU.")
            npuEnv?.close()
            try {
                val compOptions = CompiledModel.Options(Accelerator.GPU)
                return CompiledModel.create(context.assets, modelFileName, compOptions, null)
            } catch (_: Exception) {
                Log.w(TAG, "GPU fallback failed; defaulting to CPU.")
                val compOptions = CompiledModel.Options(Accelerator.CPU)
                return CompiledModel.create(context.assets, modelFileName, compOptions, null)
            }
        }
    }

    suspend fun close() {
        mutex.withLock {
            closeModel()
        }
    }

    private fun closeModel() {
        compiledModel?.close()
        compiledModel = null
        env?.close()
        env = null
    }

    fun setOptions(options: Options) {
        this.options = options
    }

    suspend fun classify(bitmap: Bitmap, rotationDegrees: Int) {
        Log.d(TAG, "classify() called")
        try {
            withContext(Dispatchers.IO) {
                val currentModel = mutex.withLock { compiledModel }
                if (currentModel == null) {
                    Log.w(TAG, "compiledModel is null, returning")
                    return@withContext
                }
                val startTime = SystemClock.uptimeMillis()

                // Try to find input tensor dimensions. If failing, default to 224x224 which is common for EfficientNet-lite.
                val names = listOf("images", "image", "args_0", "input_0", "input_1", "serving_default_input_1:0", "input_1:0", "input", "input_tensor", "data")
                var inputTensorType: com.google.ai.edge.litert.TensorType? = null
                val signatures = listOf("", "serving_default")
                
                for (sig in signatures) {
                    for (name in names) {
                        try {
                            inputTensorType = currentModel.getInputTensorType(name, sig)
                            break
                        } catch (_: Exception) {
                        }
                    }
                    if (inputTensorType != null) break
                }

                val h: Int
                val w: Int
                if (inputTensorType != null) {
                    val dimensions = inputTensorType.layout?.dimensions
                    h = if ((dimensions != null) && (dimensions.size >= 3)) dimensions[1] else 224
                    w = if ((dimensions != null) && (dimensions.size >= 3)) dimensions[2] else 224
                } else {
                    Log.w(TAG, "Could not find input tensor type. Defaulting to 224x224.")
                    h = 224
                    w = 224
                }

                val inputFloatArray = preprocessBitmapToFloatArray(bitmap, w, h, rotationDegrees)
                Log.d(TAG, "Running inference with size ${w}x${h}...")
                
                val output = mutex.withLock {
                    classifyWithCompiledModel(currentModel, inputFloatArray)
                }
                
                Log.d(TAG, "Inference done. Output size: ${output.size}")

                val outputList = output.map {
                    /** Scores in range 0..1.0 for each of the output classes. */
                    if (it < options.probabilityThreshold) 0f else it
                }

                val categories = labels.zip(outputList).asSequence().map {
                    Category(label = it.first, score = it.second)
                }.sortedByDescending { it.score }.take(options.resultCount).toList()

                val inferenceTime = SystemClock.uptimeMillis() - startTime
                Log.d(TAG, "Emitting classification result: ${categories.size} categories")
                if (isActive) {
                    _classification.emit(ClassificationResult(categories, inferenceTime))
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Image classification error occurred: ${e.message}", e)
            _error.emit(e)
        }
    }

    private fun preprocessBitmapToFloatArray(
        bitmap: Bitmap,
        targetWidth: Int,
        targetHeight: Int,
        rotationDegrees: Int
    ): FloatArray {
        val rotation = -rotationDegrees / 90
        var scaled = bitmap.scale(targetWidth, targetHeight, filter = true)
        scaled = rot90Clockwise(scaled, rotation)

        val width = scaled.width
        val height = scaled.height
        val numPixels = width * height
        val pixelsIntArray = IntArray(numPixels)
        val outputFloatArray = FloatArray(numPixels * 3)

        scaled.getPixels(pixelsIntArray, 0, width, 0, 0, width, height)
        for (i in 0 until numPixels) {
            val pixel = pixelsIntArray[i]
            val r = Color.red(pixel).toFloat()
            val g = Color.green(pixel).toFloat()
            val b = Color.blue(pixel).toFloat()
            val idx = i * 3
            outputFloatArray[idx] = (r - 127.5f) / 127.5f
            outputFloatArray[idx + 1] = (g - 127.5f) / 127.5f
            outputFloatArray[idx + 2] = (b - 127.5f) / 127.5f
        }
        return outputFloatArray
    }

    private fun rot90Clockwise(image: Bitmap, numRotation: Int): Bitmap {
        val effectiveRotation = (numRotation % 4 + 4) % 4
        if (effectiveRotation == 0) return image
        val matrix = Matrix()
        val (w, h) = Pair(image.width, image.height)
        matrix.postRotate(-90f * effectiveRotation)
        return Bitmap.createBitmap(image, 0, 0, w, h, matrix, false)
    }

    private fun classifyWithCompiledModel(model: CompiledModel, inputFloatArray: FloatArray): FloatArray {
        val inputBuffers = model.createInputBuffers()
        val outputBuffers = model.createOutputBuffers()
        return try {
            inputBuffers[0].writeFloat(inputFloatArray)
            model.run(inputBuffers, outputBuffers)
            outputBuffers[0].readFloat()
        } finally {
            inputBuffers.forEach { it.close() }
            outputBuffers.forEach { it.close() }
        }
    }

    /** Load metadata/labels from assets or model asset zip */
    private fun loadLabels(modelFileName: String): List<String> {
        val labelsList = mutableListOf<String>()
        try {
            // Try reading from a standalone labels file first
            try {
                context.assets.open("labels.txt").use { labelStream ->
                    BufferedReader(InputStreamReader(labelStream)).use { reader ->
                        reader.forEachLine { line ->
                            if (line.isNotBlank()) {
                                labelsList.add(line.trim())
                            }
                        }
                    }
                }
                if (labelsList.isNotEmpty()) {
                    Log.i(TAG, "Loaded ${labelsList.size} labels from labels.txt")
                    return labelsList
                }
            } catch (e: Exception) {
                Log.w(TAG, "labels.txt not found, trying other options: ${e.message}")
            }

            // Fallback: Try reading from the model file if it's a zip (contains metadata)
            val inputStream = context.assets.open(modelFileName)
            val zipInputStream = ZipInputStream(inputStream)
            var entry = zipInputStream.nextEntry
            var readFromZip = false
            while (entry != null) {
                if (entry.name == "labels_without_background.txt" || entry.name.endsWith(".txt")) {
                    val reader = BufferedReader(InputStreamReader(zipInputStream))
                    reader.forEachLine { line ->
                        if (line.isNotBlank()) {
                            labelsList.add(line.trim())
                        }
                    }
                    readFromZip = true
                    break
                }
                entry = zipInputStream.nextEntry
            }
            zipInputStream.close()
            inputStream.close()

            if (!readFromZip || labelsList.isEmpty()) {
                try {
                    context.assets.open("labels_without_background.txt").use { labelStream ->
                        BufferedReader(InputStreamReader(labelStream)).use { reader ->
                            reader.forEachLine { line ->
                                if (line.isNotBlank()) {
                                    labelsList.add(line.trim())
                                }
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Labels file not found in assets fallback: ${e.message}")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading labels for $modelFileName: ${e.message}")
        }
        Log.i(TAG, "Loaded ${labelsList.size} labels")
        return labelsList
    }

    enum class Delegate {
        CPU, GPU, NPU
    }

    enum class Model(val fileName: String) {
        EfficientNetLite0("efficientnet_lite0.tflite"), EfficientNetLite2("efficientnet_lite2.tflite")
    }

    data class ClassificationResult(
        val categories: List<Category>, val inferenceTime: Long
    )

    data class Category(val label: String, val score: Float)
}