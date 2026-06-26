package com.example.generet_image_ai.sd

import android.content.Context
import android.util.Log

/** Бенчмарк собственного Vulkan compute backend на Tensor G5 (PowerVR). */
object VulkanBench {
    init { System.loadLibrary("vkbench") }

    external fun benchMatmul(spirv: ByteArray, m: Int, n: Int, k: Int, iters: Int): Double
    external fun benchConv(spirv: ByteArray, cin: Int, cout: Int, h: Int, w: Int, kh: Int, kw: Int, iters: Int): Double

    fun run(context: Context): String {
        val mm = context.assets.open("shaders/matmul.spv").use { it.readBytes() }
        val cv = context.assets.open("shaders/conv2d.spv").use { it.readBytes() }
        val sb = StringBuilder()
        // matmul (регрессия фундамента)
        for (sz in intArrayOf(1024, 2048)) {
            val g = benchMatmul(mm, sz, sz, sz, 20)
            sb.appendLine("matmul ${sz}³: ${"%.0f".format(g)} GFLOPS")
        }
        // conv2d на реальных слоях SD UNet (512×512 → латент 64×64, каналы 320/640/1280)
        val convs = arrayOf(
            intArrayOf(320, 320, 64, 64, 3, 3),
            intArrayOf(640, 640, 32, 32, 3, 3),
            intArrayOf(1280, 1280, 16, 16, 3, 3),
        )
        for (c in convs) {
            val g = benchConv(cv, c[0], c[1], c[2], c[3], c[4], c[5], 10)
            sb.appendLine("conv ${c[0]}→${c[1]} ${c[2]}²: ${"%.0f".format(g)} GFLOPS")
        }
        val res = sb.toString().trim()
        Log.d("VulkanBench", res)
        return res
    }
}
