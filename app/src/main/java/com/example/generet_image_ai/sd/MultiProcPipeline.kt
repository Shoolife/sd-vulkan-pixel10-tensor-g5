package com.example.generet_image_ai.sd

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.graphics.Bitmap
import android.os.IBinder
import android.util.Log
import com.example.generet_image_ai.IModelWorker
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.File
import kotlin.coroutines.resume
import kotlin.random.Random
import kotlinx.coroutines.suspendCancellableCoroutine

/**
 * Multi-process пайплайн SD1.5 на Tensor TPU.
 *  - CLIP: в main-процессе (1 модель, 2 прогона — безопасно).
 *  - unet_a/b1/b2: каждый в своём процессе (:wa/:wb1/:wb2), модель открыта весь цикл
 *    → обходит epoll-баг EdgeTPU и убирает перезагрузку (быстрее).
 *  - VAE: в процессе :wb2 после цикла (единичный переход b2→vae).
 * Тензоры между процессами — через файлы filesDir/ipc (raw float32).
 */
class MultiProcPipeline(private val context: Context, private val useNpu: Boolean = true) {
    private val tag = "MultiProcPipe"
    private val dir = File(context.getExternalFilesDir(null), "models")
    private val ipc = File(context.getExternalFilesDir(null), "ipc").apply { mkdirs() }
    private fun p(name: String) = File(ipc, name).absolutePath

    private suspend fun bind(cls: Class<*>): Pair<IModelWorker, ServiceConnection> =
        suspendCancellableCoroutine { cont ->
            val conn = object : ServiceConnection {
                override fun onServiceConnected(n: ComponentName?, b: IBinder?) {
                    cont.resume(IModelWorker.Stub.asInterface(b) to this)
                }
                override fun onServiceDisconnected(n: ComponentName?) {}
            }
            context.bindService(Intent(context, cls), conn, Context.BIND_AUTO_CREATE)
        }

    /** CLIP в main-процессе: токены int32[1,77] -> context[1,77,768]. */
    private fun clipEncode(tokens: IntArray): FloatArray {
        val env = if (useNpu) Environment.create(BuiltinNpuAcceleratorProvider(context)) else Environment.create()
        val opts = if (useNpu) CompiledModel.Options(Accelerator.NPU, Accelerator.CPU) else CompiledModel.Options(Accelerator.GPU, Accelerator.CPU)
        val m = CompiledModel.create(File(dir, "clip.tflite").absolutePath, opts, env)
        val inB = m.createInputBuffers(); val outB = m.createOutputBuffers()
        try {
            inB[0].writeInt(tokens); m.run(inB, outB); return outB[0].readFloat()
        } finally {
            inB.forEach { runCatching { it.close() } }; outB.forEach { runCatching { it.close() } }
            runCatching { m.close() }; runCatching { env.close() }
        }
    }

    suspend fun generate(
        condTokens: IntArray, uncondTokens: IntArray,
        steps: Int = 20, cfgScale: Float = 7.5f, seed: Long = 42L,
        onStep: (Int, Int) -> Unit = { _, _ -> },
    ): Bitmap {
        // 1) CLIP (main) -> ctx[2,77,768]
        val condCtx = clipEncode(condTokens)
        val uncondCtx = clipEncode(uncondTokens)
        val ctx = FloatArray(condCtx.size + uncondCtx.size)
        System.arraycopy(condCtx, 0, ctx, 0, condCtx.size)
        System.arraycopy(uncondCtx, 0, ctx, condCtx.size, uncondCtx.size)
        ModelIO.write(p("ctx.f32"), ctx)
        Log.d(tag, "clip done")

        // 2) поднять 3 воркера, загрузить unet-части
        val (wa, ca) = bind(WorkerServiceA::class.java)
        val (wb1, cb1) = bind(WorkerServiceB1::class.java)
        val (wb2, cb2) = bind(WorkerServiceB2::class.java)
        try {
            wa.load("unet_a.tflite", useNpu); wb1.load("unet_b1.tflite", useNpu); wb2.load("unet_b2.tflite", useNpu)
            Log.d(tag, "workers loaded")

            val sched = KEulerScheduler(steps)
            val latSize = 4 * 64 * 64
            val rnd = Random(seed)
            var latents = FloatArray(latSize) { gaussian(rnd) * sched.initialScale }

            // имена выходов unet_a: [xmid, t, skip0..11]
            val aOut = Array(14) { p("a_out_$it.f32") }
            val b1Out = arrayOf(p("b1_out.f32"))
            val b2Out = arrayOf(p("b2_out.f32"))

            for ((idx, t) in sched.timesteps.withIndex()) {
                onStep(idx, steps)
                val scale = sched.inputScale()
                val inLat = FloatArray(2 * latSize)
                for (i in 0 until latSize) { val v = latents[i] * scale; inLat[i] = v; inLat[latSize + i] = v }
                ModelIO.write(p("inLat.f32"), inLat)
                ModelIO.write(p("te.f32"), KEulerScheduler.timeEmbedding(t))

                wa.run(arrayOf(p("inLat.f32"), p("ctx.f32"), p("te.f32")), aOut)
                // b1 in: xmid, t, ctx, skip6..11
                wb1.run(arrayOf(aOut[0], aOut[1], p("ctx.f32"), aOut[8], aOut[9], aOut[10], aOut[11], aOut[12], aOut[13]), b1Out)
                // b2 in: x, t, ctx, skip0..5
                wb2.run(arrayOf(b1Out[0], aOut[1], p("ctx.f32"), aOut[2], aOut[3], aOut[4], aOut[5], aOut[6], aOut[7]), b2Out)

                val noise2 = ModelIO.read(b2Out[0])
                val noise = FloatArray(latSize)
                for (i in 0 until latSize) { val c = noise2[i]; val u = noise2[latSize + i]; noise[i] = u + cfgScale * (c - u) }
                latents = sched.step(latents, noise)
                Log.d(tag, "step ${idx + 1}/$steps done")
            }

            // 3) VAE в процессе :wb2 (переход b2->vae)
            wb2.release(); wb2.load("vae.tflite", useNpu)
            ModelIO.write(p("lat.f32"), latents)
            wb2.run(arrayOf(p("lat.f32")), arrayOf(p("img.f32")))
            val img = ModelIO.read(p("img.f32"))
            Log.d(tag, "vae done")
            return toBitmap(img)
        } finally {
            runCatching { wa.release() }; runCatching { wb1.release() }; runCatching { wb2.release() }
            runCatching { context.unbindService(ca) }
            runCatching { context.unbindService(cb1) }
            runCatching { context.unbindService(cb2) }
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
