package com.example.generet_image_ai.bench

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.generet_image_ai.engine.AcceleratorChoice
import com.example.generet_image_ai.engine.LiteRtBenchmark
import com.example.generet_image_ai.engine.ModelResult
import com.example.generet_image_ai.engine.PipelineProjection
import com.example.generet_image_ai.engine.ProjectionParams
import com.example.generet_image_ai.engine.SdSpike
import com.example.generet_image_ai.engine.project
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Состояние экрана бенчмарка. Immutable — дружелюбно к рекомпозиции Compose. */
data class BenchmarkUiState(
    val accelerator: AcceleratorChoice = AcceleratorChoice.GPU,
    val steps: Int = 20,
    val running: Boolean = false,
    val statusLine: String = "Готов к прогону",
    val results: List<ModelResult> = emptyList(),
    val projection20: PipelineProjection? = null, // стандарт (20 шагов)
    val projection4: PipelineProjection? = null,  // few-step / LCM (4 шага)
)

class BenchmarkViewModel(app: Application) : AndroidViewModel(app) {

    private val benchmark = LiteRtBenchmark(app.applicationContext, warmupRuns = 2, timedRuns = 5)

    private val _state = MutableStateFlow(BenchmarkUiState())
    val state: StateFlow<BenchmarkUiState> = _state.asStateFlow()

    fun setAccelerator(choice: AcceleratorChoice) = _state.update { it.copy(accelerator = choice) }

    fun runBenchmark() {
        if (_state.value.running) return
        val context = getApplication<Application>()
        val accelerator = _state.value.accelerator

        viewModelScope.launch {
            _state.update { it.copy(running = true, results = emptyList(), statusLine = "Запуск…") }

            val results = mutableListOf<ModelResult>()
            for (spec in SdSpike.components) {
                val file = SdSpike.modelPath(context, spec)
                _state.update { it.copy(statusLine = "Бенчмарк: ${spec.displayName} на ${accelerator.displayName}…") }

                val result = if (!file.exists()) {
                    ModelResult(spec = spec, error = "Файл не найден: ${file.absolutePath}")
                } else {
                    withContext(Dispatchers.Default) { benchmark.run(spec, file.absolutePath, accelerator) }
                }
                results += result
                _state.update { it.copy(results = results.toList()) }
            }

            val proj20 = results.project(ProjectionParams(steps = 20))
            val proj4 = results.project(ProjectionParams(steps = 4))
            _state.update {
                it.copy(
                    running = false,
                    statusLine = "Готово",
                    projection20 = proj20,
                    projection4 = proj4,
                )
            }
        }
    }
}
