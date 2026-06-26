package com.example.generet_image_ai.sd

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel

@Composable
fun GenerateScreen(
    modifier: Modifier = Modifier,
    vm: GenerateViewModel = viewModel(),
) {
    val s by vm.state.collectAsState()
    Column(
        modifier = modifier.fillMaxWidth().padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Generet AI · SD1.5", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        OutlinedTextField(
            value = s.prompt,
            onValueChange = vm::setPrompt,
            enabled = !s.running,
            label = { Text("Промпт") },
            modifier = Modifier.fillMaxWidth(),
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilterChip(selected = s.engine == Engine.VULKAN, enabled = !s.running, onClick = { vm.setEngine(Engine.VULKAN) }, label = { Text("Свой Vulkan") })
            FilterChip(selected = s.engine == Engine.GPU_LCM, enabled = !s.running, onClick = { vm.setEngine(Engine.GPU_LCM) }, label = { Text("GPU·LCM") })
            FilterChip(selected = s.engine == Engine.MEDIAPIPE, enabled = !s.running, onClick = { vm.setEngine(Engine.MEDIAPIPE) }, label = { Text("MediaPipe") })
        }
        Button(onClick = vm::generate, enabled = !s.running, modifier = Modifier.fillMaxWidth()) {
            Text(if (s.running) "Генерация…" else "Сгенерировать")
        }
        // Перегенерация: новый случайный seed → другая композиция того же промпта
        if (s.image != null && !s.running) {
            OutlinedButton(onClick = vm::generate, modifier = Modifier.fillMaxWidth()) {
                Text("↻ Ещё вариант")
            }
        }
        OutlinedButton(onClick = vm::benchVulkan, enabled = !s.running, modifier = Modifier.fillMaxWidth()) {
            Text("⚡ Vulkan бенч (свой GPU)")
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (s.running) CircularProgressIndicator(modifier = Modifier.padding(2.dp))
            Text(s.status, style = MaterialTheme.typography.bodyMedium)
        }
        if (s.elapsedMs > 0) {
            Text("⏱ ${"%.1f".format(s.elapsedMs / 1000.0)} c на картинку", style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.primary)
        }
        s.image?.let { bmp ->
            Card(Modifier.fillMaxWidth()) {
                Image(bitmap = bmp.asImageBitmap(), contentDescription = "результат", modifier = Modifier.fillMaxWidth().aspectRatio(1f))
            }
        }
    }
}
