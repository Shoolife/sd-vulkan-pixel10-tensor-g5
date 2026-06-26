package com.example.generet_image_ai.sd

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import java.util.concurrent.Executors
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class Engine { GPU_LCM, TPU, MEDIAPIPE, VULKAN }

data class GenUiState(
    val prompt: String = "a photograph of an astronaut riding a horse",
    val engine: Engine = Engine.VULKAN,  // свой Vulkan UNet-движок
    val steps: Int = 4,           // LCM: 4 шага
    val cfgScale: Float = 1.5f,
    val running: Boolean = false,
    val status: String = "Готов",
    val image: Bitmap? = null,
    val elapsedMs: Long = 0,
)

class GenerateViewModel(app: Application) : AndroidViewModel(app) {
    // Все операции с моделью — строго на одном выделенном потоке (EdgeTPU dispatch потоко-чувствителен).
    private val gpuDispatcher = Executors.newSingleThreadExecutor { r -> Thread(r, "GpuThread") }.asCoroutineDispatcher()
    private val _state = MutableStateFlow(GenUiState())
    val state: StateFlow<GenUiState> = _state.asStateFlow()

    private val gpuPipe = GpuMonoPipeline(app)
    private val mpPipe = MediaPipePipeline(app)
    private val vkPipe = VulkanUnetPipeline(app)
    private val tokenizer by lazy { ClipTokenizer(app) }

    fun setEngine(e: Engine) = _state.update { it.copy(engine = e) }
    fun setPrompt(v: String) = _state.update { it.copy(prompt = v) }

    /** Бенчмарк собственного Vulkan compute backend (фундамент своего GPU-движка). */
    fun benchVulkan() {
        if (_state.value.running) return
        val app = getApplication<Application>()
        viewModelScope.launch {
            _state.update { it.copy(running = true, status = "Vulkan бенч…") }
            try {
                val res = withContext(gpuDispatcher) {
                    VulkanBench.unetInit(VulkanBench.unetShaders(app),
                        java.io.File(app.getExternalFilesDir(null), "unet_w").absolutePath)
                    val c = VulkanBench.unetSelfTest()
                    VulkanBench.unetRelease()
                    "UNet self-test fp32: relErr=${"%.4f".format(c)}"
                }
                _state.update { it.copy(running = false, status = res) }
            } catch (t: Throwable) {
                android.util.Log.e("GenerateVM", "vk bench fail", t)
                _state.update { it.copy(running = false, status = "VK ошибка: ${t.message}") }
            }
        }
    }

    fun generate() {
        if (_state.value.running) return
        val app = getApplication<Application>()
        val engine = _state.value.engine
        val steps = _state.value.steps
        val cfg = _state.value.cfgScale
        val prompt = _state.value.prompt.ifBlank { "a photograph" }
        viewModelScope.launch {
            _state.update { it.copy(running = true, status = "Генерация…", image = null) }
            try {
                val result = withContext(gpuDispatcher) {
                    val t0 = System.currentTimeMillis()
                    val bmp = when (engine) {
                        Engine.MEDIAPIPE -> mpPipe.generate(prompt, steps, (System.nanoTime() and 0x7FFFFFFF).toInt()) { i, n ->
                            _state.update { it.copy(status = "Денойз $i/$n (MediaPipe)…") }
                        }
                        Engine.TPU -> {
                            val cond = tokenizer.encode(prompt); val uncond = tokenizer.encode("")
                            MultiProcPipeline(app, useNpu = true).generate(cond, uncond, steps = steps) { i, n ->
                                _state.update { it.copy(status = "Денойз $i/$n (TPU)…") }
                            }
                        }
                        Engine.GPU_LCM -> {
                            val cond = tokenizer.encode(prompt); val uncond = tokenizer.encode("")
                            gpuPipe.generate(cond, uncond, steps = steps, cfgScale = cfg, seed = System.nanoTime(), lcm = true) { i, n ->
                                _state.update { it.copy(status = "Денойз $i/$n (GPU·LCM)…") }
                            }
                        }
                        Engine.VULKAN -> {
                            val cond = tokenizer.encode(prompt); val uncond = tokenizer.encode("")
                            vkPipe.generate(cond, uncond, steps = steps, cfgScale = cfg, seed = System.nanoTime()) { i, n ->
                                _state.update { it.copy(status = "Денойз $i/$n (свой Vulkan UNet)…") }
                            }
                        }
                    }
                    bmp to (System.currentTimeMillis() - t0)
                }
                _state.update {
                    it.copy(running = false, status = "Готово", image = result.first, elapsedMs = result.second)
                }
            } catch (t: Throwable) {
                android.util.Log.e("GenerateVM", "fail", t)
                _state.update { it.copy(running = false, status = "Ошибка: ${t.message}") }
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        runCatching { gpuPipe.close() }
        runCatching { mpPipe.close() }
    }
}
