package com.example.generet_image_ai.bench

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.generet_image_ai.engine.AcceleratorChoice
import com.example.generet_image_ai.engine.ModelResult
import com.example.generet_image_ai.engine.PipelineProjection

@Composable
fun BenchmarkScreen(
    modifier: Modifier = Modifier,
    vm: BenchmarkViewModel = viewModel(),
) {
    val state by vm.state.collectAsState()

    Column(
        modifier = modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            text = "GPU Baseline · SD1.5",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold,
        )
        Text(
            text = "Замер чистой латентности компонентов на устройстве (LiteRT Next · CompiledModel). " +
                "Положи .tflite в Android/data/<pkg>/files/models — см. README.",
            style = MaterialTheme.typography.bodySmall,
        )

        AcceleratorPicker(
            selected = state.accelerator,
            enabled = !state.running,
            onSelect = vm::setAccelerator,
        )

        Button(
            onClick = vm::runBenchmark,
            enabled = !state.running,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(if (state.running) "Идёт прогон…" else "Запустить бенчмарк")
        }

        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            if (state.running) {
                CircularProgressIndicator(modifier = Modifier.height(18.dp))
            }
            Text(state.statusLine, style = MaterialTheme.typography.bodyMedium)
        }

        state.results.forEach { ResultCard(it) }

        state.projection20?.let { ProjectionCard("Стандарт · 20 шагов", it) }
        state.projection4?.let { ProjectionCard("Few-step / LCM · 4 шага", it) }
    }
}

@Composable
private fun AcceleratorPicker(
    selected: AcceleratorChoice,
    enabled: Boolean,
    onSelect: (AcceleratorChoice) -> Unit,
) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        AcceleratorChoice.entries.forEach { choice ->
            FilterChip(
                selected = choice == selected,
                enabled = enabled,
                onClick = { onSelect(choice) },
                label = { Text(choice.displayName) },
            )
        }
    }
}

@Composable
private fun ResultCard(result: ModelResult) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(result.spec.displayName, fontWeight = FontWeight.SemiBold)
            val stats = result.stats
            if (stats != null) {
                Text(
                    "avg ${stats.avgMs.fmt()} мс · p50 ${stats.p50Ms.fmt()} · " +
                        "min ${stats.minMs.fmt()} · max ${stats.maxMs.fmt()} (×${stats.iterations})",
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                )
            } else {
                Text(
                    result.error ?: "Ошибка",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }
        }
    }
}

@Composable
private fun ProjectionCard(title: String, projection: PipelineProjection) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(title, fontWeight = FontWeight.SemiBold)
            Text(
                "≈ ${(projection.totalMs / 1000.0).fmt()} c на картинку",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.primary,
            )
            projection.breakdown.forEach { (label, ms) ->
                Text(
                    "  $label → ${ms.fmt()} мс",
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                )
            }
            Text(
                "CFG-фактор = 1: diffusion экспортирован с batch=2 (cond+uncond в одном проходе).",
                style = MaterialTheme.typography.labelSmall,
            )
        }
    }
}

private fun Double.fmt(): String = "%.1f".format(this)
