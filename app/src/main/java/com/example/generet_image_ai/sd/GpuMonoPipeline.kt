package com.example.generet_image_ai.sd

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import kotlin.random.Random

/**
 * SD1.5 на GPU, МОНОЛИТНЫЙ UNet, всё в одном процессе.
 *
 * GPU-делегат LiteRT (а) не использует AOT-компилятор Tensor → монолит компилится
 * (нет INTERNAL-краша на 3388 ops), (б) не имеет лимита «1 контекст» TPU → все модели
 * держим открытыми весь цикл, 20 шагов без крашей.
 *
 * Модели держатся ОТКРЫТЫМИ между генерациями (ensureLoaded): первая генерация платит
 * загрузку+компиляцию шейдеров (~17с), последующие — только денойз+vae.
 *
 * LCM-режим: gpu_unet_lcm_b1.tflite (batch=1, без CFG), LcmScheduler, 4 шага.
 *   clip:  tokens int32[1,77]        -> context[1,77,768]
 *   unet:  latents[1,4,64,64], ctx[1,77,768], time_emb[1,320] -> eps[1,4,64,64]
 *   vae:   latents[1,4,64,64]        -> image[1,3,512,512] в [-1,1]
 */
class GpuMonoPipeline(private val context: Context) {
    private val tag = "GpuMonoPipe"
    private val dir = File(context.getExternalFilesDir(null), "models")
    private val cacheDir = File(context.getExternalFilesDir(null), "gpu_cache").apply { mkdirs() }

    private var env: Environment? = null
    private var clip: CompiledModel? = null
    private var unet: CompiledModel? = null
    private var vae: CompiledModel? = null
    private var loadedLcm: Boolean? = null   // какой режим сейчас загружен

    /**
     * Чистый GPU FP32. preferTextureWeights/TEXTURE_2D хранят веса как fp16-текстуры (даже при
     * Precision.FP32 вычислений) → обрезают выход (UNet cond max 3.78 vs 4.43, VAE [-0.5,0.17]) →
     * слабое следование промпту и серая каша. Поэтому только precision FP32 + кэш шейдеров.
     */
    /** UNet/CLIP: fp16-веса как текстуры (FP32 на GPU = OOM, GPU-heap < весов). Вычисления FP32. */
    private fun open(name: String): CompiledModel {
        val opts = CompiledModel.Options(Accelerator.GPU, Accelerator.CPU).apply {
            gpuOptions = CompiledModel.GpuOptions(
                precision = CompiledModel.GpuOptions.Precision.FP32,
                constantTensorSharing = true,
                preferTextureWeights = true,
                bufferStorageType = CompiledModel.GpuOptions.BufferStorageType.TEXTURE_2D,
                serializeProgramCache = true,
                serializationDir = cacheDir.absolutePath,
                modelCacheKey = name,
                priority = CompiledModel.GpuOptions.Priority.HIGH,
                backend = CompiledModel.GpuOptions.Backend.OPENCL,
            )
        }
        return CompiledModel.create(File(dir, name).absolutePath, opts, env!!)
    }

    /** VAE: чистый FP32 без текстур (texture-веса сжимали выход декодера [-0.5,0.17]). VAE мал → влезает. */
    private fun openVae(name: String): CompiledModel {
        val opts = CompiledModel.Options(Accelerator.GPU, Accelerator.CPU).apply {
            gpuOptions = CompiledModel.GpuOptions(
                precision = CompiledModel.GpuOptions.Precision.FP32,
                backend = CompiledModel.GpuOptions.Backend.OPENCL,
            )
        }
        return CompiledModel.create(File(dir, name).absolutePath, opts, env!!)
    }

    /** Тёплые модели: fp16-текстурные веса помещаются все разом → быстрые повторные генерации. */
    fun ensureLoaded(lcm: Boolean) {
        if (loadedLcm == lcm && unet != null) return
        close()
        env = Environment.create()
        clip = open("gpu_clip.tflite")
        unet = open(if (lcm) "gpu_unet_lcm.tflite" else "gpu_unet.tflite")   // batch=2 (CFG одним прогоном)
        vae = openVae("gpu_vae.tflite")
        loadedLcm = lcm
        Log.d(tag, "models open (gpu, lcm=$lcm)")
    }

    fun generate(
        condTokens: IntArray, uncondTokens: IntArray,
        steps: Int = 4, cfgScale: Float = 7.5f, seed: Long = 42L,
        lcm: Boolean = true,
        onStep: (Int, Int) -> Unit = { _, _ -> },
    ): Bitmap {
        ensureLoaded(lcm)
        val clip = clip!!; val unet = unet!!; val vae = vae!!
        val live = mutableListOf<TensorBuffer>()
        val latSize = 4 * 64 * 64
        val rnd = Random(seed)
        try {
            val condCtx = runModel(clip, listOf(condTokens), live)[0]
            val uncondCtx = runModel(clip, listOf(uncondTokens), live)[0]
            val ctx = FloatArray(condCtx.size + uncondCtx.size)
            System.arraycopy(condCtx, 0, ctx, 0, condCtx.size)
            System.arraycopy(uncondCtx, 0, ctx, condCtx.size, uncondCtx.size)
            Log.d(tag, "clip done")

            val sched = LcmScheduler(steps)
            var latents = FloatArray(latSize) { gaussian(rnd) * sched.initialScale }
            for ((idx, t) in sched.timesteps.withIndex()) {
                onStep(idx, steps)
                val inLat = FloatArray(2 * latSize)
                for (i in 0 until latSize) { val v = latents[i]; inLat[i] = v; inLat[latSize + i] = v }
                val te = LcmScheduler.timeEmbedding(t)
                val out2 = runModel(unet, listOf(inLat, ctx, te), live)[0]   // [2,4,64,64]
                val eps = FloatArray(latSize)
                for (i in 0 until latSize) { val c = out2[i]; val u = out2[latSize + i]; eps[i] = u + cfgScale * (c - u) }
                val noise = if (idx == steps - 1) null else FloatArray(latSize) { gaussian(rnd) }
                latents = sched.step(latents, eps, noise)
                Log.d(tag, "step ${idx + 1}/$steps done")
            }

            val img = runModel(vae, listOf(latents), live)[0]
            Log.d(tag, "vae done")
            return toBitmap(img)
        } finally {
            live.forEach { runCatching { it.close() } }
        }
    }

    fun close() {
        runCatching { clip?.close() }; clip = null
        runCatching { unet?.close() }; unet = null
        runCatching { vae?.close() }; vae = null
        runCatching { env?.close() }; env = null
        loadedLcm = null
    }

    /** Прогон: буферы создаём заново на вызов, в рамках ОДНОГО GPU-контекста. */
    private fun runModel(m: CompiledModel, inputs: List<Any>, live: MutableList<TensorBuffer>): List<FloatArray> {
        val t0 = System.nanoTime()
        val inB = m.createInputBuffers(); val outB = m.createOutputBuffers()
        val t1 = System.nanoTime()
        try {
            inputs.forEachIndexed { i, v ->
                when (v) {
                    is FloatArray -> inB[i].writeFloat(v)
                    is IntArray -> inB[i].writeInt(v)
                    else -> error("bad input $v")
                }
            }
            val t2 = System.nanoTime()
            m.run(inB, outB)
            val t3 = System.nanoTime()
            val out = outB.map { it.readFloat() }
            val t4 = System.nanoTime()
            fun ms(a: Long, b: Long) = (b - a) / 1_000_000
            Log.d(tag, "TIMING buf=${ms(t0,t1)} write=${ms(t1,t2)} run=${ms(t2,t3)} read=${ms(t3,t4)}")
            return out
        } finally {
            inB.forEach { runCatching { it.close() } }
            outB.forEach { runCatching { it.close() } }
        }
    }

    private fun toBitmap(chw: FloatArray): Bitmap {
        val hw = 512 * 512
        val px = IntArray(hw)
        for (i in 0 until hw) {
            val r = ((chw[i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            val g = ((chw[hw + i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            val b = ((chw[2 * hw + i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            px[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(px, 512, 512, Bitmap.Config.ARGB_8888)
    }

    private fun gaussian(r: Random): Float {
        val u1 = r.nextDouble().coerceAtLeast(1e-9); val u2 = r.nextDouble()
        return (Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2)).toFloat()
    }
}
