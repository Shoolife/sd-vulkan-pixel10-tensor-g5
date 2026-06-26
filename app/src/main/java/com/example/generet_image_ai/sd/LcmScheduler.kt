package com.example.generet_image_ai.sd

import kotlin.math.sqrt

/**
 * LCM-sampler (Latent Consistency Model) — точный порт diffusers LCMScheduler
 * (epsilon-prediction, num_train=1000, original_inference=50, timestep_scaling=10, sigma_data=0.5).
 *
 * Даёт качество ~20-шагового SD за 4 шага. Использование как KEuler, но:
 *   initialScale = 1.0, inputScale() = 1.0 (LCM не масштабирует вход),
 *   step(latents, eps, rnd) повторно зашумляет между шагами (кроме последнего).
 */
class LcmScheduler(
    val nInferenceSteps: Int,
    private val nTrainingSteps: Int = 1000,
    private val originalInferenceSteps: Int = 50,
    private val timestepScaling: Float = 10.0f,
) {
    val timesteps: IntArray
    val initialScale: Float = 1.0f          // init_noise_sigma = 1.0
    private val alphasCumprod: DoubleArray
    private val sigmaData = 0.5
    private var stepCount = 0

    init {
        // alphas_cumprod: beta scaled_linear (sqrt-пространство), как в SD1.5
        val betaStart = 0.00085; val betaEnd = 0.0120
        alphasCumprod = DoubleArray(nTrainingSteps)
        var acc = 1.0
        for (i in 0 until nTrainingSteps) {
            val b = Math.pow(
                sqrt(betaStart) + (sqrt(betaEnd) - sqrt(betaStart)) * i / (nTrainingSteps - 1),
                2.0,
            )
            acc *= (1.0 - b)
            alphasCumprod[i] = acc
        }

        // LCM timesteps: k=1000//50=20; origin=[1..50]*k-1=[19..999]; reversed; evenly spaced indices
        val k = nTrainingSteps / originalInferenceSteps
        val origin = IntArray(originalInferenceSteps) { (it + 1) * k - 1 }   // [19,39,...,999]
        val rev = origin.reversedArray()                                      // [999,...,19]
        timesteps = IntArray(nInferenceSteps) {
            val idx = Math.floor(it.toDouble() * rev.size / nInferenceSteps).toInt()
            rev[idx]
        }
    }

    fun inputScale(): Float = 1.0f   // LCM scale_model_input = identity

    /**
     * LCM-шаг: eps (model output) -> обновлённые latents.
     * predicted_x0 = (x - sqrt(beta)*eps)/sqrt(alpha); denoised = c_out*x0 + c_skip*x;
     * не последний шаг: x' = sqrt(alpha_prev)*denoised + sqrt(beta_prev)*noise; иначе x'=denoised.
     */
    fun step(latents: FloatArray, eps: FloatArray, gaussianNoise: FloatArray?): FloatArray {
        val t = timesteps[stepCount]
        val alphaT = alphasCumprod[t]
        val betaT = 1.0 - alphaT
        val st = t * timestepScaling.toDouble()
        val cSkip = sigmaData * sigmaData / (st * st + sigmaData * sigmaData)
        val cOut = st / sqrt(st * st + sigmaData * sigmaData)

        val sqrtAlpha = sqrt(alphaT); val sqrtBeta = sqrt(betaT)
        val isLast = stepCount == nInferenceSteps - 1
        val tPrev = if (!isLast) timesteps[stepCount + 1] else -1
        val alphaPrev = if (tPrev >= 0) alphasCumprod[tPrev] else 1.0
        val sqrtAlphaPrev = sqrt(alphaPrev); val sqrtBetaPrev = sqrt(1.0 - alphaPrev)

        for (i in latents.indices) {
            val x = latents[i].toDouble()
            val x0 = (x - sqrtBeta * eps[i]) / sqrtAlpha
            val denoised = cOut * x0 + cSkip * x
            latents[i] = if (isLast) {
                denoised.toFloat()
            } else {
                (sqrtAlphaPrev * denoised + sqrtBetaPrev * (gaussianNoise!![i])).toFloat()
            }
        }
        stepCount++
        return latents
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
