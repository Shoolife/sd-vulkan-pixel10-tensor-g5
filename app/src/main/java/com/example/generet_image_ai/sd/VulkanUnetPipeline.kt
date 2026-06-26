package com.example.generet_image_ai.sd

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random

/**
 * Генерация СВОИМ Vulkan UNet-движком в LCM-пайплайне.
 * CLIP и VAE — через LiteRT (gpu_clip/gpu_vae.tflite), денойз — наш Vulkan UNet
 * (unetInit/unetForward). LCM, batch=1 (без CFG, как требует наш UNet [1,...]).
 */
class VulkanUnetPipeline(private val context: Context) {
    private val tag = "VkUnetPipe"
    private val dir = File(context.getExternalFilesDir(null), "models")
    private val latSize = 4 * 64 * 64

    // float[] → fp16 little-endian bytes
    private fun toFp16(f: FloatArray): ByteArray {
        val bb = ByteBuffer.allocate(f.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (v in f) bb.putShort(floatToHalf(v))
        return bb.array()
    }
    private fun fromFp16(b: ByteArray): FloatArray {
        val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(b.size / 2) { halfToFloat(bb.short) }
    }
    private fun floatToHalf(v: Float): Short {
        val bits = java.lang.Float.floatToIntBits(v)
        val sign = (bits ushr 16) and 0x8000
        var exp = ((bits ushr 23) and 0xff) - 127 + 15
        val mant = bits and 0x7fffff
        if (exp <= 0) return sign.toShort()
        if (exp >= 0x1f) return (sign or 0x7c00).toShort()
        return (sign or (exp shl 10) or (mant ushr 13)).toShort()
    }
    private fun halfToFloat(h: Short): Float {
        val bits = h.toInt() and 0xffff
        val sign = (bits and 0x8000) shl 16
        val exp = (bits ushr 10) and 0x1f
        val mant = bits and 0x3ff
        val out = when {
            exp == 0 -> if (mant == 0) sign else { // subnormal
                var e = -1; var m = mant
                do { e++; m = m shl 1 } while (m and 0x400 == 0)
                sign or ((e + 127 - 15 + 1) shl 23) or ((m and 0x3ff) shl 13)
            }
            exp == 0x1f -> sign or 0x7f800000 or (mant shl 13)
            else -> sign or ((exp - 15 + 127) shl 23) or (mant shl 13)
        }
        return java.lang.Float.intBitsToFloat(out)
    }

    // sinusoidal time-embedding (diffusers get_timestep_embedding, dim=320, flip_sin_to_cos, shift=0)
    private fun timeProj(t: Int): FloatArray {
        val dim = 320; val half = dim / 2
        val out = FloatArray(dim)
        for (i in 0 until half) {
            val freq = Math.exp(-Math.log(10000.0) * i / half)
            val a = t * freq
            out[i] = Math.cos(a).toFloat()        // flip_sin_to_cos: cos первыми
            out[half + i] = Math.sin(a).toFloat()
        }
        return out
    }

    private fun clipContext(tokens: IntArray, env: Environment): FloatArray {
        val opts = CompiledModel.Options(Accelerator.GPU, Accelerator.CPU)
        val m = CompiledModel.create(File(dir, "gpu_clip.tflite").absolutePath, opts, env)
        val inB = m.createInputBuffers(); val outB = m.createOutputBuffers()
        try { inB[0].writeInt(tokens); m.run(inB, outB); return outB[0].readFloat() }  // [1,77,768]
        finally { inB.forEach { it.close() }; outB.forEach { it.close() }; m.close() }
    }
    private fun vaeDecode(latents: FloatArray, env: Environment): FloatArray {
        val opts = CompiledModel.Options(Accelerator.GPU, Accelerator.CPU)
        val m = CompiledModel.create(File(dir, "gpu_vae.tflite").absolutePath, opts, env)
        val inB = m.createInputBuffers(); val outB = m.createOutputBuffers()
        try { inB[0].writeFloat(latents); m.run(inB, outB); return outB[0].readFloat() }
        finally { inB.forEach { it.close() }; outB.forEach { it.close() }; m.close() }
    }

    fun generate(condTokens: IntArray, steps: Int = 4, seed: Long = 42L,
                 onStep: (Int, Int) -> Unit = { _, _ -> }): Bitmap {
        // CLIP в своём env, закрываем ПОЛНОСТЬЮ перед нашим Vulkan UNet (иначе 2 GPU-контекста → OOM)
        val ctxFp16: ByteArray
        run {
            val env = Environment.create()
            try { ctxFp16 = toFp16(clipContext(condTokens, env)) } finally { env.close() }
        }
        Log.d(tag, "clip done")
        // наш Vulkan UNet (один GPU-контекст, 1.7ГБ весов)
        VulkanBench.unetInit(VulkanBench.unetShaders(context),
            File(context.getExternalFilesDir(null), "unet_w").absolutePath)
        Log.d(tag, "unet init")
        val sched = LcmScheduler(steps)
        val rnd = Random(seed)
        var latents = FloatArray(latSize) { gaussian(rnd) * sched.initialScale }
        for ((idx, t) in sched.timesteps.withIndex()) {
            onStep(idx, steps)
            val noiseB = VulkanBench.unetForward(toFp16(latents), toFp16(timeProj(t)), ctxFp16)
            val eps = fromFp16(noiseB)
            val noise = if (idx == steps - 1) null else FloatArray(latSize) { gaussian(rnd) }
            latents = sched.step(latents, eps, noise)
            Log.d(tag, "step ${idx + 1}/$steps done")
        }
        VulkanBench.unetRelease()  // освобождаем 1.7ГБ весов перед VAE
        // VAE в своём env
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
