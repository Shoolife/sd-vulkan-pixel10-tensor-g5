package com.example.generet_image_ai.sd

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.DataInputStream
import java.io.File
import java.net.Socket
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random

/**
 * Генерация цельным UNet на Google Tensor TPU.
 *
 * На production-Pixel (без root) EdgeTPU не пускает приложение под своим uid (allowlist Google),
 * поэтому UNet-forward выполняет маленький shell-демон `litert_tpu_daemon` (uid shell → TPU разрешён),
 * запущенный через adb. Общение — файлы в каталоге приложения (tpu_ipc/). CLIP/VAE остаются в
 * приложении (CPU/GPU, TPU им не нужен).
 *
 * Модель демона: (latent[1,4,64,64], timestep[1] СЫРОЙ, ctx[1,77,768]) -> noise[1,4,64,64], bf16.
 */
class TpuUnetPipeline(private val context: Context) {
    private val tag = "TpuUnetPipe"
    private val dir = File(context.getExternalFilesDir(null), "models")
    private val latSize = 4 * 64 * 64
    private val tpuPort = 8763   // TCP-loopback к shell-демону litert_tpu_daemon

    private fun f32bytes(a: FloatArray): ByteArray {
        val bb = ByteBuffer.allocate(a.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (v in a) bb.putFloat(v)
        return bb.array()
    }

    /** Forward UNet на TPU через shell-демон по TCP. lat/ctx любого батча, out — outN элементов. */
    private fun tpuForward(lat: FloatArray, t: Int, ctx: FloatArray, outN: Int): FloatArray {
        Socket("127.0.0.1", tpuPort).use { s ->
            s.tcpNoDelay = true
            s.soTimeout = 60_000
            val os = s.getOutputStream()
            os.write(f32bytes(lat))
            os.write(f32bytes(floatArrayOf(t.toFloat())))
            os.write(f32bytes(ctx))
            os.flush()
            val ins = DataInputStream(s.getInputStream())
            val buf = ByteArray(outN * 4)
            ins.readFully(buf)
            val bb = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN)
            return FloatArray(outN) { bb.float }
        }
    }

    private fun clipContext(tokens: IntArray, env: Environment): FloatArray {
        val opts = CompiledModel.Options(Accelerator.CPU)
        val m = CompiledModel.create(File(dir, "gpu_clip.tflite").absolutePath, opts, env)
        val inB = m.createInputBuffers(); val outB = m.createOutputBuffers()
        try { inB[0].writeInt(tokens); m.run(inB, outB); return outB[0].readFloat() }
        finally { inB.forEach { it.close() }; outB.forEach { it.close() }; m.close() }
    }
    private fun vaeDecode(latents: FloatArray, env: Environment): FloatArray {
        val opts = CompiledModel.Options(Accelerator.GPU, Accelerator.CPU)
        // TAESD — крошечный декодер (5 МБ против 95 МБ) с тем же контрактом; полный VAE занимал
        // ~4 с из 34 с генерации. Ошибка не накапливается (декодер работает один раз), corr 0.99.
        val taesd = File(dir, "taesd_decoder.tflite")
        val vae = if (taesd.exists()) taesd else File(dir, "gpu_vae.tflite")
        val m = CompiledModel.create(vae.absolutePath, opts, env)
        val inB = m.createInputBuffers(); val outB = m.createOutputBuffers()
        try { inB[0].writeFloat(latents); m.run(inB, outB); return outB[0].readFloat() }
        finally { inB.forEach { it.close() }; outB.forEach { it.close() }; m.close() }
    }

    /**
     * Число первых шагов с CFG. По умолчанию 2 из 4: CFG задаёт структуру сцены на первых шагах,
     * дальше почти не влияет, но стоит второго forward. Замерено при одном seed:
     * 4 шага с CFG — 44.5 с, 2 шага — 34.1 с (картинка та же), 0 — 24.6 с, но астронавт
     * вырождается в жокея (теряется следование промпту).
     * Переопределяется файлом models/cfg_steps.txt — для замеров через adb без пересборки.
     */
    private fun cfgStepsFrom(file: File, steps: Int): Int =
        runCatching { file.readText().trim().toInt().coerceIn(0, steps) }
            .getOrDefault(minOf(2, steps))

    fun generate(condTokens: IntArray, uncondTokens: IntArray, steps: Int = 4, cfgScale: Float = 1.5f,
                 seed: Long = 42L, onStep: (Int, Int) -> Unit = { _, _ -> }): Bitmap {
        val cfgSteps = cfgStepsFrom(File(dir, "cfg_steps.txt"), steps)
        Log.d(tag, "cfgSteps=$cfgSteps из $steps (forward'ов будет ${steps + cfgSteps})")
        val condCtx: FloatArray; val uncondCtx: FloatArray
        run {
            val env = Environment.create()
            try { condCtx = clipContext(condTokens, env); uncondCtx = clipContext(uncondTokens, env) }
            finally { env.close() }
        }
        Log.d(tag, "clip done")

        val sched = LcmScheduler(steps)
        val rnd = Random(seed)
        var latents = FloatArray(latSize) { gaussian(rnd) * sched.initialScale }
        for ((idx, t) in sched.timesteps.withIndex()) {
            onStep(idx, steps)
            // CFG считается только на первых cfgSteps шагах: там формируется структура, дальше он
            // почти не влияет, а стоит второго forward. cfgSteps=steps — обычный CFG (8 forward),
            // 0 — без CFG (4 forward). B=1: cond/uncond раздельно (batch-CFG B=2 крашит рантайм edgetpu).
            val eps = if (cfgScale <= 1f || idx >= cfgSteps) {
                tpuForward(latents, t, condCtx, latSize)
            } else {
                val cond = tpuForward(latents, t, condCtx, latSize)
                val uncond = tpuForward(latents, t, uncondCtx, latSize)
                FloatArray(latSize) { uncond[it] + cfgScale * (cond[it] - uncond[it]) }
            }
            val noise = if (idx == steps - 1) null else FloatArray(latSize) { gaussian(rnd) }
            latents = sched.step(latents, eps, noise)
            Log.d(tag, "step ${idx + 1}/$steps done (TPU daemon)")
        }
        Log.d(tag, "unet(TPU) done")

        val env2 = Environment.create()
        val img = try { vaeDecode(latents, env2) } finally { env2.close() }
        Log.d(tag, "vae done")
        return toBitmap(img)
    }

    private fun toBitmap(chw: FloatArray): Bitmap {
        val hw = 512 * 512; val px = IntArray(hw)
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
