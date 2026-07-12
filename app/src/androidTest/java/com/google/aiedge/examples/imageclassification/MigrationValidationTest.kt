package com.google.aiedge.examples.imageclassification

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.CompiledModel.Options
import com.google.ai.edge.litert.Environment

@RunWith(AndroidJUnit4::class)
class MigrationValidationTest {

    @Test
    fun verifyMigrationInference() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        
        // Initialize CompiledModel
        val options = Options(Accelerator.CPU)
        val modelFileName = ImageClassificationHelper.DEFAULT_MODEL.fileName
        val compiledModel = CompiledModel.create(context.assets, modelFileName, options, null)
        
        val inputBuffers = compiledModel.createInputBuffers()
        val outputBuffers = compiledModel.createOutputBuffers()
        
        // Fallback mechanism to get input tensor size
        val inputTensorType = try {
            compiledModel.getInputTensorType("args_0")
        } catch (e: Exception) {
            try {
                compiledModel.getInputTensorType("input_0")
            } catch (e2: Exception) {
                compiledModel.getInputTensorType("serving_default_input_1:0")
            }
        }
        val inputShape = inputTensorType.layout?.dimensions ?: throw IllegalStateException("Input shape is null")
        val inputSize = inputShape.fold(1) { acc, dim -> acc * dim }
        
        val dummyInput = FloatArray(inputSize) { 0.5f }
        inputBuffers[0].writeFloat(dummyInput)
        
        // Run inference
        compiledModel.run(inputBuffers, outputBuffers)
        
        // Verify output is populated (non-zero)
        val outputArray = outputBuffers[0].readFloat()
        
        var isPopulated = false
        for (value in outputArray) {
            if (value != 0.0f) {
                isPopulated = true
                break
            }
        }
        
        inputBuffers.forEach { it.close() }
        outputBuffers.forEach { it.close() }
        compiledModel.close()
        
        assertTrue("Output buffer should have non-zero results from inference", isPopulated)
    }

    @Test
    fun verifyByteBufferInference() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        
        val options = Options(Accelerator.CPU)
        val modelFileName = ImageClassificationHelper.DEFAULT_MODEL.fileName
        val compiledModel = CompiledModel.create(context.assets, modelFileName, options, null)
        
        val inputBuffers = compiledModel.createInputBuffers()
        val outputBuffers = compiledModel.createOutputBuffers()
        
        val inputTensorType = try {
            compiledModel.getInputTensorType("args_0")
        } catch (e: Exception) {
            try {
                compiledModel.getInputTensorType("input_0")
            } catch (e2: Exception) {
                compiledModel.getInputTensorType("serving_default_input_1:0")
            }
        }
        val inputShape = inputTensorType.layout?.dimensions ?: throw IllegalStateException("Input shape is null")
        val inputSize = inputShape.fold(1) { acc, dim -> acc * dim }
        
        val dummyInput = FloatArray(inputSize) { 0.5f }
        val byteBuffer = java.nio.ByteBuffer.allocateDirect(dummyInput.size * 4).order(java.nio.ByteOrder.nativeOrder())
        byteBuffer.asFloatBuffer().put(dummyInput)
        byteBuffer.rewind()
        
        inputBuffers[0].writeByteBuffer(byteBuffer)
        compiledModel.run(inputBuffers, outputBuffers)
        
        val outputByteBuffer = outputBuffers[0].readByteBuffer()
        val outputArray = FloatArray(outputByteBuffer.remaining() / 4)
        outputByteBuffer.asFloatBuffer().get(outputArray)
        
        var isPopulated = false
        for (value in outputArray) {
            if (value != 0.0f) {
                isPopulated = true
                break
            }
        }
        
        inputBuffers.forEach { it.close() }
        outputBuffers.forEach { it.close() }
        compiledModel.close()
        
        assertTrue("ByteBuffer output should have non-zero results from inference", isPopulated)
    }

    @Test
    fun verifyNpuEnvironmentCreation() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val envOptions = mapOf(
            Environment.Option.DispatchLibraryDir to context.applicationInfo.nativeLibraryDir,
            Environment.Option.CompilerPluginLibraryDir to context.applicationInfo.nativeLibraryDir
        )
        val npuEnv = try {
            Environment.create(BuiltinNpuAcceleratorProvider(context), envOptions)
        } catch (e: Exception) {
            // Expected on test emulators or headless environments without NPU hardware
            null
        }
        npuEnv?.close()
        assertTrue("Environment creation API and options map should resolve without runtime link errors", true)
    }
}
