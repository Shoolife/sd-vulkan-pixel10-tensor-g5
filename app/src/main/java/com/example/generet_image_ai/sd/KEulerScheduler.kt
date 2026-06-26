package com.example.generet_image_ai.sd

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * K-Euler sampler для Stable Diffusion — точный порт эталона litert_torch
 * (samplers/k_euler.py + util.get_alphas_cumprod). Чистая математика, без TPU.
 *
 * Использование:
 *   val s = KEulerScheduler(steps)
 *   latents = randn(shape) * s.initialScale
 *   for (t in s.timesteps) { inLat = latents * s.inputScale(); ... model ...; latents = s.step(latents, noise) }
 */
class KEulerScheduler(
    val nInferenceSteps: Int,
    private val nTrainingSteps: Int = 1000,
) {
    /** Целочисленные timesteps (передаются в time-embedding). */
    val timesteps: IntArray
    /** Полный массив сигм длиной nInferenceSteps+1 (последняя = 0). */
    private val sigmas: FloatArray
    /** Масштаб начального шума. */
    val initialScale: Float
    private var stepCount = 0

    init {
        // timesteps = linspace(nTraining-1, 0, nInferenceSteps)
        val ts = FloatArray(nInferenceSteps) {
            (nTrainingSteps - 1) * (1f - it.toFloat() / (nInferenceSteps - 1))
        }
        timesteps = IntArray(nInferenceSteps) { Math.round(ts[it]) }

        // alphas_cumprod (beta linspace в sqrt-пространстве, как в diffusers)
        val betaStart = 0.00085; val betaEnd = 0.0120
        val alphasCumprod = DoubleArray(nTrainingSteps)
        var acc = 1.0
        for (i in 0 until nTrainingSteps) {
            val b = Math.pow(
                sqrt(betaStart) + (sqrt(betaEnd) - sqrt(betaStart)) * i / (nTrainingSteps - 1),
                2.0,
            )
            acc *= (1.0 - b)
            alphasCumprod[i] = acc
        }
        // sigma(i) = sqrt((1-acp)/acp); log; интерполируем по timesteps; exp; append 0
        val logSigmasFull = DoubleArray(nTrainingSteps) {
            0.5 * ln((1.0 - alphasCumprod[it]) / alphasCumprod[it])
        }
        sigmas = FloatArray(nInferenceSteps + 1)
        for (k in 0 until nInferenceSteps) {
            sigmas[k] = exp(interp(ts[k].toDouble(), logSigmasFull)).toFloat()
        }
        sigmas[nInferenceSteps] = 0f
        initialScale = sigmas.max()
    }

    /** input_scale текущего шага = 1/sqrt(sigma^2+1). */
    fun inputScale(): Float {
        val s = sigmas[stepCount]
        return (1.0 / sqrt(s.toDouble() * s + 1.0)).toFloat()
    }

    /** k-euler шаг: latents += output*(sigma_to - sigma_from). Возвращает обновлённые latents (in-place). */
    fun step(latents: FloatArray, output: FloatArray): FloatArray {
        val from = sigmas[stepCount]
        val to = sigmas[stepCount + 1]
        val d = to - from
        for (i in latents.indices) latents[i] += output[i] * d
        stepCount++
        return latents
    }

    /** Линейная интерполяция log-сигмы по дробному training-индексу x (range 0..nTraining-1). */
    private fun interp(x: Double, table: DoubleArray): Double {
        if (x <= 0) return table[0]
        if (x >= table.size - 1) return table[table.size - 1]
        val lo = x.toInt(); val frac = x - lo
        return table[lo] * (1 - frac) + table[lo + 1] * frac
    }

    companion object {
        /** time-embedding: cos(t*freq)[160] ++ sin(t*freq)[160], freq=10000^(-i/160). 320-dim. */
        fun timeEmbedding(timestep: Int): FloatArray {
            val out = FloatArray(320)
            for (i in 0 until 160) {
                val freq = Math.pow(10000.0, -i.toDouble() / 160.0)
                val x = timestep * freq
                out[i] = Math.cos(x).toFloat()
                out[160 + i] = Math.sin(x).toFloat()
            }
            return out
        }
    }
}
