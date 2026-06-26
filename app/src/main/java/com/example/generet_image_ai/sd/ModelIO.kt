package com.example.generet_image_ai.sd

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Чтение/запись FloatArray в файл как raw little-endian float32 (быстрый IPC-транспорт). */
object ModelIO {
    fun write(path: String, data: FloatArray) {
        val bb = ByteBuffer.allocate(data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        bb.asFloatBuffer().put(data)
        File(path).outputStream().channel.use { it.write(bb) }
    }

    fun read(path: String): FloatArray {
        val f = File(path)
        val n = (f.length() / 4).toInt()
        val bb = ByteBuffer.allocate(n * 4).order(ByteOrder.LITTLE_ENDIAN)
        f.inputStream().channel.use { ch -> while (bb.hasRemaining()) if (ch.read(bb) < 0) break }
        bb.flip()
        val out = FloatArray(n)
        bb.asFloatBuffer().get(out)
        return out
    }
}
