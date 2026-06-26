package com.example.generet_image_ai.sd

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.File
import kotlin.random.Random

/**
 * Полный t2i-пайплайн Stable Diffusion 1.5 на Tensor TPU.
 *
 * 5 AOT-скомпонованных под Tensor G5 моделей (в filesDir):
 *   clip.tflite, unet_a.tflite, unet_b1.tflite, unet_b2.tflite, vae.tflite
 * UNet разрезан на 3 части (порог компилятора), численно эквивалентен цельному (проверено).
 *
 * Порядок I/O (из litert_torch-трейса, формы из dump_io):
 *   CLIP:   in tokens int32[1,77]            -> out context[1,77,768]
 *   unet_a: in latents[2,4,64,64], ctx[2,77,768], time_emb[1,320]
 *           -> out [x_mid, t, skip0..11]      (14 выходов)
 *   unet_b1:in x_mid, t, ctx, skip6..11       -> out x
 *   unet_b2:in x, t, ctx, skip0..5            -> out noise[2,4,64,64]
 *   vae:    in latents[1,4,64,64]             -> out image[1,3,512,512] в [-1,1]
 */
class TpuSdPipeline(
    private val context: Context,
    private val useNpu: Boolean = true,
) {
    private val tag = "TpuSdPipeline"
    private val dir = File(context.getExternalFilesDir(null), "models")


    fun load() {
        Log.d(tag, "pipeline ready (npu=$useNpu)")
    }

    companion object { var DEBUG_IO = false }

    /** Дамп ИМЁН входов/выходов модели через классический Interpreter (без прогона). */
    private fun dumpNames(name: String) {
        try {
            val it = org.tensorflow.lite.Interpreter(File(dir, name))
            val ins = (0 until it.inputTensorCount).map { i -> it.getInputTensor(i).name() }
            val outs = (0 until it.outputTensorCount).map { i -> it.getOutputTensor(i).name() }
            Log.d(tag, "NAME $name IN=$ins")
            Log.d(tag, "NAME $name OUT=$outs")
            it.close()
        } catch (t: Throwable) { Log.d(tag, "NAME $name FAIL ${t.message}") }
    }

    /** Дамп размеров входов/выходов модели (число float-элементов в порядке буферов). */
    private fun dumpIO(name: String) {
        val env = if (useNpu) Environment.create(BuiltinNpuAcceleratorProvider(context))
                  else Environment.create()
        val m = CompiledModel.create(File(dir, name).absolutePath,
            CompiledModel.Options(Accelerator.NPU, Accelerator.CPU), env)
        try {
            val ins = m.createInputBuffers().map { it.readFloat().size }
            val outs = m.createOutputBuffers().map { it.readFloat().size }
            Log.d(tag, "IO $name IN=$ins")
            Log.d(tag, "IO $name OUT=$outs")
        } finally { runCatching { m.close() }; runCatching { env.close() } }
    }

    private fun stat(name: String, a: FloatArray) {
        var mn = Float.MAX_VALUE; var mx = -Float.MAX_VALUE; var sum = 0.0; var nan = 0
        for (v in a) {
            if (v.isNaN() || v.isInfinite()) nan++
            else { if (v < mn) mn = v; if (v > mx) mx = v; sum += v }
        }
        Log.d(tag, "STAT %-10s n=%d min=%.4f max=%.4f mean=%.4f nan=%d"
            .format(name, a.size, mn, mx, sum / a.size, nan))
    }

    /**
     * Один прогон модели: open → (write inputs) → run → read → close.
     * Эмпирика EdgeTPU на Tensor (этот билд рантайма): после прогона одной модели
     * прогон ДРУГОЙ модели в том же процессе крашит диспетчер; режим «закрыть прошлую
     * перед следующей» работает, но накапливает fd (~10–12 открытий на процесс).
     * Поэтому держим число шагов малым, чтобы весь пайплайн уложился в бюджет.
     */
    private fun runOnce(name: String, inputs: List<Any>): List<FloatArray> {
        val env = if (useNpu) Environment.create(BuiltinNpuAcceleratorProvider(context))
                  else Environment.create()
        val opts = if (useNpu) CompiledModel.Options(Accelerator.NPU, Accelerator.CPU)
                   else CompiledModel.Options(Accelerator.GPU, Accelerator.CPU)
        var model: CompiledModel? = null
        val bufs = mutableListOf<AutoCloseable>()
        try {
            val t0 = System.currentTimeMillis()
            model = CompiledModel.create(File(dir, name).absolutePath, opts, env)
            val inB = model.createInputBuffers().also { bufs.addAll(it) }
            val outB = model.createOutputBuffers().also { bufs.addAll(it) }
            val tLoad = System.currentTimeMillis()
            inputs.forEachIndexed { i, v ->
                when (v) {
                    is FloatArray -> inB[i].writeFloat(v)
                    is IntArray -> inB[i].writeInt(v)
                    else -> error("bad input type $v")
                }
            }
            model.run(inB, outB)
            val res = outB.map { it.readFloat() }
            val tRun = System.currentTimeMillis()
            Log.d(tag, "TIMING $name load=${tLoad - t0}ms run=${tRun - tLoad}ms")
            return res
        } finally {
            bufs.forEach { runCatching { it.close() } }
            runCatching { model?.close() }
            runCatching { env.close() }
        }
    }

    /**
     * Генерация. [condTokens]/[uncondTokens] — массивы по 77 int (BOS..EOS).
     * Возвращает Bitmap 512×512.
     */
    fun generate(
        condTokens: IntArray,
        uncondTokens: IntArray,
        steps: Int = 4,
        cfgScale: Float = 7.5f,
        seed: Long = 42L,
        onStep: (Int, Int) -> Unit = { _, _ -> },
    ): Bitmap {
        if (DEBUG_IO) {
            dumpNames("unet_a.tflite"); dumpNames("unet_b1.tflite"); dumpNames("unet_b2.tflite")
            return toBitmap(FloatArray(3 * 512 * 512))
        }
        val condCtx = runOnce("clip.tflite", listOf(condTokens))[0]    // [1,77,768]
        val uncondCtx = runOnce("clip.tflite", listOf(uncondTokens))[0]
        val ctx = FloatArray(condCtx.size + uncondCtx.size)
        System.arraycopy(condCtx, 0, ctx, 0, condCtx.size)
        System.arraycopy(uncondCtx, 0, ctx, condCtx.size, uncondCtx.size)
        Log.d(tag, "clip done"); stat("condCtx", condCtx)

        val sched = KEulerScheduler(steps)
        val latSize = 4 * 64 * 64
        val rnd = Random(seed)
        var latents = FloatArray(latSize) { gaussian(rnd) * sched.initialScale }

        for ((idx, t) in sched.timesteps.withIndex()) {
            onStep(idx, steps)
            val scale = sched.inputScale()
            val inLat = FloatArray(2 * latSize)
            for (i in 0 until latSize) {
                val v = latents[i] * scale
                inLat[i] = v; inLat[latSize + i] = v
            }
            val te = KEulerScheduler.timeEmbedding(t)

            val a = runOnce("unet_a.tflite", listOf(inLat, ctx, te))
            val xMid = a[0]; val tProc = a[1]
            val skip = (2..13).map { a[it] }       // skip0..11
            val b1in = ArrayList<Any>(listOf(xMid, tProc, ctx)); for (k in 6..11) b1in.add(skip[k])
            val x = runOnce("unet_b1.tflite", b1in)[0]
            val b2in = ArrayList<Any>(listOf(x, tProc, ctx)); for (k in 0..5) b2in.add(skip[k])
            val noise2 = runOnce("unet_b2.tflite", b2in)[0]

            val noise = FloatArray(latSize)
            for (i in 0 until latSize) {
                val c = noise2[i]; val u = noise2[latSize + i]
                noise[i] = u + cfgScale * (c - u)
            }
            if (idx == 0) {
                stat("inLat", inLat); stat("xMid", xMid); stat("a.t", tProc)
                stat("skip0", skip[0]); stat("skip11", skip[11])
                stat("b1.x", x); stat("noise2", noise2); stat("noiseCFG", noise)
            }
            latents = sched.step(latents, noise)
            Log.d(tag, "step ${idx + 1}/$steps done"); if (idx == 0 || idx == steps - 1) stat("latents$idx", latents)
        }
        Log.d(tag, "denoise done, vae decode…"); stat("finalLat", latents)
        val img = runOnce("vae.tflite", listOf(latents))[0]   // [1,3,512,512] в [-1,1]
        stat("vaeOut", img)
        return toBitmap(img)
    }

    /** [1,3,512,512] CHW в [-1,1] -> ARGB Bitmap 512×512. */
    private fun toBitmap(chw: FloatArray): Bitmap {
        val hw = 512 * 512
        val px = IntArray(hw)
        for (p in 0 until hw) {
            val r = ((chw[p] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            val g = ((chw[hw + p] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            val b = ((chw[2 * hw + p] + 1f) * 127.5f).toInt().coerceIn(0, 255)
            px[p] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(px, 512, 512, Bitmap.Config.ARGB_8888)
    }

    /** Box-Muller для N(0,1). */
    private fun gaussian(r: Random): Float {
        val u1 = r.nextDouble().coerceAtLeast(1e-9); val u2 = r.nextDouble()
        return (Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2)).toFloat()
    }

    fun close() {
        // Каждая Model владеет своим Environment и закрывает его в Model.close().
    }
}
