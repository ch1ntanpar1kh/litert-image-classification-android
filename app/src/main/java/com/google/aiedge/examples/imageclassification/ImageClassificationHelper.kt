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
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
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
        /** Number of threads to be used for ops that support multi-threading.
         * threadCount>= -1. Setting numThreads to 0 has the effect of disabling multithreading,
         * which is equivalent to setting numThreads to 1. If unspecified, or set to the value -1,
         * the number of threads used will be implementation-defined and platform-dependent.
         * */
        var threadCount: Int = DEFAULT_THREAD_COUNT
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

    /** Init a CompiledModel from [Model] with [Delegate]*/
    suspend fun initClassifier() {
        close()
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

    private fun createCompiledModelWithNpuCascade(modelFileName: String): CompiledModel? {
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
            } catch (e: Exception) {
                modelFileName
            }
            val model = CompiledModel.create(context.assets, targetAsset, compOptions, npuEnv)
            env = npuEnv
            return model
        } catch (e: Exception) {
            Log.w(TAG, "NPU JIT compilation unsupported on this device; falling back to GPU.", e)
            npuEnv?.close()
            npuEnv = null
            try {
                val compOptions = CompiledModel.Options(Accelerator.GPU)
                return CompiledModel.create(context.assets, modelFileName, compOptions, null)
            } catch (e2: Exception) {
                Log.w(TAG, "GPU fallback failed; defaulting to CPU.", e2)
                val compOptions = CompiledModel.Options(Accelerator.CPU)
                return CompiledModel.create(context.assets, modelFileName, compOptions, null)
            }
        }
    }

    fun close() {
        compiledModel?.close()
        compiledModel = null
        env?.close()
        env = null
    }

    fun setOptions(options: Options) {
        this.options = options
    }

    suspend fun classify(bitmap: Bitmap, rotationDegrees: Int) {
        try {
            withContext(Dispatchers.IO) {
                val currentModel = compiledModel ?: return@withContext
                val startTime = SystemClock.uptimeMillis()

                val inputTensorType = try {
                    currentModel.getInputTensorType("args_0")
                } catch (e: Exception) {
                    try {
                        currentModel.getInputTensorType("input_0")
                    } catch (e2: Exception) {
                        try {
                            currentModel.getInputTensorType("input_1")
                        } catch (e3: Exception) {
                            Log.e(TAG, "Could not find input tensor type: ${e3.message}")
                            null
                        }
                    }
                } ?: return@withContext

                val dimensions = inputTensorType.layout?.dimensions
                val h = if (dimensions != null && dimensions.size >= 3) dimensions[1] else 224
                val w = if (dimensions != null && dimensions.size >= 3) dimensions[2] else 224

                val inputFloatArray = preprocessBitmapToFloatArray(bitmap, w, h, rotationDegrees)
                val output = classifyWithCompiledModel(inputFloatArray)

                val outputList = output.map {
                    /** Scores in range 0..1.0 for each of the output classes. */
                    if (it < options.probabilityThreshold) 0f else it
                }

                val categories = labels.zip(outputList).map {
                    Category(label = it.first, score = it.second)
                }.sortedByDescending { it.score }.take(options.resultCount)

                val inferenceTime = SystemClock.uptimeMillis() - startTime
                if (isActive) {
                    _classification.emit(ClassificationResult(categories, inferenceTime))
                }
            }
        } catch (e: Exception) {
            Log.i(TAG, "Image classification error occurred: ${e.message}")
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
        var scaled = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
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

    private fun classifyWithCompiledModel(inputFloatArray: FloatArray): FloatArray {
        val model = compiledModel ?: return FloatArray(0)
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

    /** Load metadata/labels from model asset zip or fallback */
    private fun loadLabels(modelFileName: String): List<String> {
        val labelsList = mutableListOf<String>()
        try {
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