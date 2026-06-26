package com.example.generet_image_ai.sd

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapExtractor
import com.google.mediapipe.tasks.vision.imagegenerator.ImageGenerator
import com.google.mediapipe.tasks.vision.imagegenerator.ImageGenerator.ImageGeneratorOptions
import java.io.File

/**
 * Генерация через нативный GPU-движок MediaPipe Image Generator (SD1.5).
 * Бенчмарк Google: ~15с / 20 шагов 512×512 на high-end GPU — специализированный движок
 * (fused-attention, оптимизированные ядра) против нашего generic LiteRT-графа.
 *
 * Модель — папка fp16 .bin (конвертёр mediapipe image_generator_converter) в filesDir/mp_bins.
 */
class MediaPipePipeline(private val context: Context) {
    private val tag = "MediaPipePipe"
    private val modelDir = File(context.getExternalFilesDir(null), "mp_bins").absolutePath
    private var generator: ImageGenerator? = null

    fun ensureLoaded() {
        if (generator != null) return
        val opts = ImageGeneratorOptions.builder()
            .setImageGeneratorModelDirectory(modelDir)
            .build()
        generator = ImageGenerator.createFromOptions(context, opts)
        Log.d(tag, "MediaPipe ImageGenerator загружен из $modelDir")
    }

    fun generate(prompt: String, steps: Int, seed: Int, onStep: (Int, Int) -> Unit = { _, _ -> }): Bitmap {
        ensureLoaded()
        val gen = generator!!
        // пошаговый вызов даёт промежуточный прогресс
        gen.setInputs(prompt, steps, seed)
        var last: Bitmap? = null
        for (i in 0 until steps) {
            onStep(i, steps)
            val res = gen.execute(i == steps - 1)   // показать картинку только на последнем шаге
            if (i == steps - 1) res?.generatedImage()?.let { last = BitmapExtractor.extract(it) }
            Log.d(tag, "step ${i + 1}/$steps")
        }
        Log.d(tag, "done")
        return last ?: error("нет результата")
    }

    fun close() {
        runCatching { generator?.close() }
        generator = null
    }
}
