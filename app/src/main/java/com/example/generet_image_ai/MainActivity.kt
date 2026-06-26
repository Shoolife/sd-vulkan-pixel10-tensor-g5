package com.example.generet_image_ai

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.example.generet_image_ai.sd.GenerateScreen
import com.example.generet_image_ai.ui.theme.GeneretTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContent { AppRoot() }
    }
}

@Composable
private fun AppRoot() {
    GeneretTheme {
        Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
            GenerateScreen(modifier = Modifier.padding(innerPadding))
        }
    }
}
