package com.example.generet_image_ai.sd

import android.content.Context
import android.util.Log

/** Бенчмарк собственного Vulkan compute backend на Tensor G5 (PowerVR). */
object VulkanBench {
    init { System.loadLibrary("vkbench") }

    external fun benchMatmul(spirv: ByteArray, m: Int, n: Int, k: Int, iters: Int): Double
    external fun benchConv(spirv: ByteArray, cin: Int, cout: Int, h: Int, w: Int, kh: Int, kw: Int, iters: Int): Double
    external fun benchConvGemm(im2col: ByteArray, matmul: ByteArray, cin: Int, cout: Int, h: Int, w: Int, kh: Int, kw: Int, iters: Int): Double
    external fun benchSilu(spirv: ByteArray, n: Int, iters: Int): Double
    external fun benchGroupNorm(spirv: ByteArray, c: Int, hw: Int, g: Int, iters: Int): Double

    fun run(context: Context): String {
        val mm = context.assets.open("shaders/matmul.spv").use { it.readBytes() }
        val cv = context.assets.open("shaders/conv2d.spv").use { it.readBytes() }
        val ic = context.assets.open("shaders/im2col.spv").use { it.readBytes() }
        val sb = StringBuilder()
        // conv2d на реальных слоях SD UNet: наивный прямой vs im2col+GEMM
        val convs = arrayOf(
            intArrayOf(320, 320, 64, 64, 3, 3),
            intArrayOf(640, 640, 32, 32, 3, 3),
            intArrayOf(1280, 1280, 16, 16, 3, 3),
        )
        for (c in convs) {
            val gemm = benchConvGemm(ic, mm, c[0], c[1], c[2], c[3], c[4], c[5], 10)
            sb.appendLine("conv ${c[0]}→${c[1]} ${c[2]}²: ${"%.0f".format(gemm)} GFLOPS")
        }
        // простые ядра + groupnorm на размерах SD UNet
        val si = context.assets.open("shaders/silu.spv").use { it.readBytes() }
        val gn = context.assets.open("shaders/groupnorm.spv").use { it.readBytes() }
        sb.appendLine("silu 320×64²: ${"%.2f".format(benchSilu(si, 320 * 64 * 64, 30))} ms")
        sb.appendLine("groupnorm 320×64² g32: ${"%.2f".format(benchGroupNorm(gn, 320, 64 * 64, 32, 30))} ms")
        sb.appendLine("groupnorm 1280×16² g32: ${"%.2f".format(benchGroupNorm(gn, 1280, 16 * 16, 32, 30))} ms")
        val res = sb.toString().trim()
        Log.d("VulkanBench", res)
        return res
    }
}
