package com.example.generet_image_ai.engine

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import kotlin.system.measureNanoTime

/**
 * Выбор ускорителя. GPU/CPU встроены в litert 2.1.5; NPU (Tensor TPU) требует
 * BuiltinNpuAcceleratorProvider + on-device рантайм libLiteRtDispatch_GoogleTensor.so
 * (в jniLibs) и МОДЕЛЬ, AOT-скомпилированную под Tensor (*_Google_Tensor_G5.tflite).
 */
enum class AcceleratorChoice(val displayName: String) {
    GPU("GPU (OpenCL/FP16)"),
    CPU("CPU"),
    NPU("NPU / TPU");

    fun toLiteRt(): Accelerator = when (this) {
        GPU -> Accelerator.GPU
        CPU -> Accelerator.CPU
        NPU -> Accelerator.NPU
    }
}

enum class ModelRole { TEXT_ENCODER, UNET, VAE_DECODER, OTHER }

data class ModelSpec(
    val id: String,
    val displayName: String,
    val fileName: String,
    val role: ModelRole,
)

data class LatencyStats(
    val iterations: Int,
    val avgMs: Double,
    val minMs: Double,
    val p50Ms: Double,
    val maxMs: Double,
)

data class ModelResult(
    val spec: ModelSpec,
    val stats: LatencyStats? = null,
    val error: String? = null,
) {
    val ok: Boolean get() = stats != null
}

class LiteRtBenchmark(
    private val context: Context,
    private val warmupRuns: Int = 2,
    private val timedRuns: Int = 5,
) {
    fun run(spec: ModelSpec, modelPath: String, accelerator: AcceleratorChoice): ModelResult {
        var env: Environment? = null
        var model: CompiledModel? = null
        val buffers = mutableListOf<AutoCloseable>()
        return try {
            // Для NPU нужен провайдер Tensor-ускорителя (иначе NPU не регистрируется).
            env = if (accelerator == AcceleratorChoice.NPU) {
                Environment.create(BuiltinNpuAcceleratorProvider(context))
            } else {
                Environment.create()
            }
            model = CompiledModel.create(modelPath, optionsFor(accelerator), env)

            val inputs = model.createInputBuffers().also { buffers.addAll(it) }
            val outputs = model.createOutputBuffers().also { buffers.addAll(it) }

            // GPU/NPU исполняют асинхронно: чтение выхода форсирует синхронизацию.
            fun runOnce() {
                model.run(inputs, outputs)
                outputs.firstOrNull()?.readFloat()
            }

            repeat(warmupRuns) { runOnce() }
            val timesMs = DoubleArray(timedRuns)
            for (i in 0 until timedRuns) {
                timesMs[i] = measureNanoTime { runOnce() } / 1_000_000.0
            }
            ModelResult(spec = spec, stats = timesMs.toStats())
        } catch (t: Throwable) {
            android.util.Log.e("GeneretBench", "fail on ${spec.id}", t)
            ModelResult(spec = spec, error = t.message ?: t.javaClass.simpleName)
        } finally {
            buffers.forEach { runCatching { it.close() } }
            runCatching { model?.close() }
            runCatching { env?.close() }
        }
    }

    private fun optionsFor(accelerator: AcceleratorChoice): CompiledModel.Options = when (accelerator) {
        AcceleratorChoice.GPU ->
            CompiledModel.Options(Accelerator.GPU, Accelerator.CPU).apply {
                gpuOptions = CompiledModel.GpuOptions(
                    precision = CompiledModel.GpuOptions.Precision.FP16,
                    backend = CompiledModel.GpuOptions.Backend.OPENCL,
                )
            }
        AcceleratorChoice.CPU -> CompiledModel.Options(Accelerator.CPU)
        AcceleratorChoice.NPU -> CompiledModel.Options(Accelerator.NPU, Accelerator.CPU)
    }

    private fun DoubleArray.toStats(): LatencyStats {
        val sorted = sortedArray()
        return LatencyStats(
            iterations = size,
            avgMs = average(),
            minMs = sorted.first(),
            p50Ms = sorted[size / 2],
            maxMs = sorted.last(),
        )
    }
}
