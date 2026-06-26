package com.example.generet_image_ai.sd

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import com.example.generet_image_ai.IModelWorker
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.File

/**
 * Базовый воркер: в СВОЁМ процессе держит одну модель открытой и гоняет её много раз.
 * Это обходит баг EdgeTPU-рантайма (epoll fd при переключении моделей в одном процессе):
 * один процесс = одна модель = много прогонов (проверено: 30+ ОК).
 * 3 подкласса с android:process=":wa/:wb1/:wb2".
 */
abstract class WorkerService : Service() {
    private val tag get() = "Worker_${javaClass.simpleName}"
    private val dir by lazy { File(getExternalFilesDir(null), "models") }

    private var env: Environment? = null
    private var model: CompiledModel? = null

    private val binder = object : IModelWorker.Stub() {
        override fun load(modelFileName: String, npu: Boolean) {
            release()
            env = if (npu) Environment.create(BuiltinNpuAcceleratorProvider(applicationContext))
                  else Environment.create()
            val opts = if (npu) CompiledModel.Options(Accelerator.NPU, Accelerator.CPU)
                       else CompiledModel.Options(Accelerator.GPU, Accelerator.CPU)
            model = CompiledModel.create(File(dir, modelFileName).absolutePath, opts, env!!)
            Log.d(tag, "loaded $modelFileName npu=$npu")
        }

        override fun run(inFiles: Array<String>, outFiles: Array<String>) {
            val m = model ?: error("model not loaded")
            val inB = m.createInputBuffers()
            val outB = m.createOutputBuffers()
            try {
                for (i in inFiles.indices) inB[i].writeFloat(ModelIO.read(inFiles[i]))
                m.run(inB, outB)
                for (i in outFiles.indices) ModelIO.write(outFiles[i], outB[i].readFloat())
            } finally {
                inB.forEach { runCatching { it.close() } }
                outB.forEach { runCatching { it.close() } }
            }
        }

        override fun release() {
            runCatching { model?.close() }; model = null
            runCatching { env?.close() }; env = null
        }
    }

    override fun onBind(intent: Intent?): IBinder = binder
}

class WorkerServiceA : WorkerService()
class WorkerServiceB1 : WorkerService()
class WorkerServiceB2 : WorkerService()
