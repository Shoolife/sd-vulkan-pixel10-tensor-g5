package com.example.generet_image_ai.sd

import android.content.Context
import android.util.Log

/** Бенчмарк собственного Vulkan compute backend на Tensor G5 (PowerVR). */
object VulkanBench {
    init { System.loadLibrary("vkbench") }

    /** matmul C[M,N]=A[M,K]*B[K,N] fp32 на GPU, возвращает GFLOPS (среднее по iters). */
    external fun benchMatmul(spirv: ByteArray, m: Int, n: Int, k: Int, iters: Int): Double

    fun run(context: Context): String {
        val spv = context.assets.open("shaders/matmul.spv").use { it.readBytes() }
        val sb = StringBuilder()
        for (sz in intArrayOf(512, 1024, 2048)) {
            val g = benchMatmul(spv, sz, sz, sz, 20)
            val line = "matmul ${sz}³: ${"%.0f".format(g)} GFLOPS"
            Log.d("VulkanBench", line); sb.appendLine(line)
        }
        return sb.toString().trim()
    }
}
