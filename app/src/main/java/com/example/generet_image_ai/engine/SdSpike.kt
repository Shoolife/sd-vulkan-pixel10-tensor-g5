package com.example.generet_image_ai.engine

import android.content.Context
import java.io.File

/**
 * Набор компонентов SD1.5 для baseline-спайка и расчёт проекции полного времени
 * генерации из латентности отдельных компонентов.
 *
 * Файлы моделей НЕ кладутся в APK (вместе ~1–2 ГБ). Их нужно загрузить на устройство
 * вручную (см. README) в каталог: getExternalFilesDir("models").
 */
object SdSpike {

    /** Имена файлов соответствуют экспортируемым компонентам SD1.5 (MediaPipe/diffusers формат). */
    val components: List<ModelSpec> = listOf(
        ModelSpec(
            id = "text_encoder",
            displayName = "Text Encoder (CLIP)",
            fileName = "text_encoder.tflite",
            role = ModelRole.TEXT_ENCODER,
        ),
        ModelSpec(
            id = "unet",
            displayName = "UNet (denoiser)",
            fileName = "unet.tflite",
            role = ModelRole.UNET,
        ),
        ModelSpec(
            id = "vae_decoder",
            displayName = "VAE Decoder",
            fileName = "vae_decoder.tflite",
            role = ModelRole.VAE_DECODER,
        ),
    )

    /** Каталог на устройстве, где ожидаются .tflite-файлы. */
    fun modelsDir(context: Context): File =
        File(context.getExternalFilesDir(null), "models").apply { mkdirs() }

    fun modelPath(context: Context, spec: ModelSpec): File =
        File(modelsDir(context), spec.fileName)
}

/** Параметры расчёта проекции полного времени генерации одной картинки. */
data class ProjectionParams(
    val steps: Int,
    /**
     * Множитель CFG для UNet: 1, если экспорт батчит cond/uncond в один проход (batch=2),
     * 2 — если они идут отдельными forward-проходами. Наша конвертация (ai-edge-torch SD)
     * экспортирует diffusion с batch=2, поэтому по умолчанию 1.
     */
    val cfgFactor: Int = 1,
)

/** Спроецированное полное время генерации одной картинки на основе латентности компонентов. */
data class PipelineProjection(
    val params: ProjectionParams,
    val totalMs: Double,
    val breakdown: List<Pair<String, Double>>, // (что, суммарные мс) для UI
)

/**
 * Считает ожидаемое время генерации:
 *   total = textEncoder·1 + unet·(steps·cfgFactor) + vaeDecoder·1
 * Компоненты без успешного замера в сумму не входят (и помечаются в breakdown).
 */
fun List<ModelResult>.project(params: ProjectionParams): PipelineProjection {
    val breakdown = mutableListOf<Pair<String, Double>>()
    var total = 0.0
    for (result in this) {
        val avg = result.stats?.avgMs ?: continue
        val runs = when (result.spec.role) {
            ModelRole.TEXT_ENCODER -> 1
            ModelRole.UNET -> params.steps * params.cfgFactor
            ModelRole.VAE_DECODER -> 1
            ModelRole.OTHER -> 1
        }
        val sub = avg * runs
        total += sub
        breakdown += "${result.spec.displayName} ×$runs" to sub
    }
    return PipelineProjection(params = params, totalMs = total, breakdown = breakdown)
}
