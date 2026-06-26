package com.example.generet_image_ai.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val Accent = Color(0xFF7C5CFF)
private val AccentDark = Color(0xFFB9A8FF)

private val DarkColors = darkColorScheme(
    primary = AccentDark,
    secondary = Color(0xFF80DEEA),
    background = Color(0xFF0E0E12),
    surface = Color(0xFF17171F),
)

private val LightColors = lightColorScheme(
    primary = Accent,
    secondary = Color(0xFF00838F),
)

@Composable
fun GeneretTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkColors else LightColors,
        content = content,
    )
}
