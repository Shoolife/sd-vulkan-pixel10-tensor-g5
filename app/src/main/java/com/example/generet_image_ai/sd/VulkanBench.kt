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
    external fun benchAttention(spirv: ByteArray, h: Int, seqQ: Int, seqK: Int, d: Int, iters: Int): Double
    external fun runResnetBlock(shaders: Array<ByteArray>, weights: Array<ByteArray>, c: Int, h: Int, w: Int, temb: Int): Double
    external fun runTransformerBlock(shaders: Array<ByteArray>, weights: Array<ByteArray>, c: Int, h: Int, w: Int, ctxN: Int, ctxD: Int, nh: Int): Double
    external fun unetInit(shaders: Array<ByteArray>, weightsDir: String): Int
    external fun unetForward(lat: ByteArray, temb: ByteArray, ctx: ByteArray): ByteArray
    external fun unetRelease()
    external fun unetSelfTest(): Double

    fun unetShaders(ctx: Context): Array<ByteArray> {
        fun sh(n: String) = ctx.assets.open("shaders/$n.spv").use { it.readBytes() }
        // порядок = enum S в C++
        return arrayOf(sh("groupnorm"), sh("silu"), sh("conv2d"), sh("matmul"), sh("addbias"),
            sh("addbias2"), sh("add"), sh("layernorm"), sh("split_heads"), sh("attention"),
            sh("merge_heads"), sh("geglu"), sh("t_chw2hwc"), sh("t_hwc2chw"), sh("upsample"), sh("attention_big"))
    }

    fun transformerCheck(ctx: Context): String {
        fun sh(n: String) = ctx.assets.open("shaders/$n.spv").use { it.readBytes() }
        fun wt(n: String) = ctx.assets.open("tf/$n.bin").use { it.readBytes() }
        val shaders = arrayOf(sh("groupnorm"), sh("conv2d"), sh("addbias"), sh("addbias2"), sh("t_chw2hwc"),
            sh("layernorm"), sh("matmul"), sh("split_heads"), sh("attention"), sh("merge_heads"), sh("add"), sh("t_hwc2chw"), sh("geglu"))
        val weights = arrayOf(
            wt("x"), wt("ctx"), wt("gn_w"), wt("gn_b"), wt("pin_w"), wt("pin_b"), wt("n1_w"), wt("n1_b"),
            wt("q1"), wt("k1"), wt("v1"), wt("o1"), wt("o1_b"), wt("n2_w"), wt("n2_b"),
            wt("q2"), wt("k2"), wt("v2"), wt("o2"), wt("o2_b"), wt("n3_w"), wt("n3_b"),
            wt("geglu"), wt("geglu_b"), wt("ffout"), wt("ffout_b"), wt("pout_w"), wt("pout_b"), wt("y"),
        )
        val err = runTransformerBlock(shaders, weights, 320, 64, 64, 77, 768, 8)
        return "Transformer блок vs PyTorch: relErr=${"%.4f".format(err)} ${if (err < 0.06) "OK(fp16)" else "FAIL"}"
    }

    fun resnetCheck(ctx: Context): String {
        fun sh(n: String) = ctx.assets.open("shaders/$n.spv").use { it.readBytes() }
        fun wt(n: String) = ctx.assets.open("rb/$n.bin").use { it.readBytes() }
        val shaders = arrayOf(sh("groupnorm"), sh("silu"), sh("conv2d"), sh("matmul"), sh("addbias"), sh("add"))
        val weights = arrayOf(
            wt("x"), wt("temb"), wt("norm1_w"), wt("norm1_b"), wt("conv1_w"), wt("conv1_b"),
            wt("tproj_wT"), wt("tproj_b"), wt("norm2_w"), wt("norm2_b"), wt("conv2_w"), wt("conv2_b"), wt("y"),
        )
        val err = runResnetBlock(shaders, weights, 320, 64, 64, 1280)
        return "ResNet блок vs PyTorch: relErr=${"%.4f".format(err)} ${if (err < 0.03) "OK" else "FAIL"}"
    }

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
        // attention: self (seq=4096, d=40, 8 голов) и cross (seqK=77)
        val at = context.assets.open("shaders/attention.spv").use { it.readBytes() }
        sb.appendLine("self-attn 64² d40 h8: ${"%.1f".format(benchAttention(at, 8, 4096, 4096, 40, 5))} ms")
        sb.appendLine("cross-attn 64² d40 h8: ${"%.1f".format(benchAttention(at, 8, 4096, 77, 40, 5))} ms")
        sb.appendLine(resnetCheck(context))   // сборка ResNet блока + сверка с PyTorch
        sb.appendLine(transformerCheck(context))  // сборка Transformer блока + сверка
        val res = sb.toString().trim()
        Log.d("VulkanBench", res)
        return res
    }
}
