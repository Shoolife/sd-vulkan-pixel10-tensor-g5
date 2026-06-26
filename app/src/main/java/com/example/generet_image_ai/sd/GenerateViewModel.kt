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
import org.json.JSONArray

data class GenUiState(
    val useNpu: Boolean = false,  // GPU+LCM — рабочий путь; TPU оставлен экспериментально (нестабилен на бете)
    val steps: Int = 4,           // LCM: 4 шага = качество ~20-шагового SD1.5
    val cfgScale: Float = 1.5f,   // умеренный CFG (LCM не терпит высокий)
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

    fun setNpu(v: Boolean) = _state.update { it.copy(useNpu = v) }

    fun generate() {
        if (_state.value.running) return
        val app = getApplication<Application>()
        val npu = _state.value.useNpu
        val steps = _state.value.steps
        val cfg = _state.value.cfgScale
        viewModelScope.launch {
            _state.update { it.copy(running = true, status = "Генерация…", image = null) }
            try {
                val result = withContext(gpuDispatcher) {
                    val cond = loadTokens("sd/tokens_astronaut.json")
                    val uncond = loadTokens("sd/tokens_uncond.json")
                    val t0 = System.currentTimeMillis()
                    val bmp = if (npu) {
                        // TPU: экспериментальный multi-process (нестабилен на бете), оставлен для исследований
                        MultiProcPipeline(app, useNpu = true).generate(cond, uncond, steps = steps) { i, n ->
                            _state.update { it.copy(status = "Денойз $i/$n (TPU)…") }
                        }
                    } else {
                        // GPU + LCM: 4 шага = качество 20, случайный seed → новая композиция каждый раз
                        gpuPipe.generate(cond, uncond, steps = steps, cfgScale = cfg, seed = System.nanoTime(), lcm = true) { i, n ->
                            _state.update { it.copy(status = "Денойз $i/$n (GPU·LCM)…") }
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
    }

    private fun loadTokens(asset: String): IntArray {
        val txt = getApplication<Application>().assets.open(asset).bufferedReader().use { it.readText() }
        val arr = JSONArray(txt)
        return IntArray(arr.length()) { arr.getInt(it) }
    }
}
