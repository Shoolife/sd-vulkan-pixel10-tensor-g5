package com.example.generet_image_ai

import android.content.Context
import android.system.Os
import android.system.OsConstants
import java.io.FileDescriptor
import com.google.android.play.core.splitcompat.SplitCompat
import com.google.android.play.core.splitcompat.SplitCompatApplication

/** SplitCompat подгружает нативный NPU-рантайм (libLiteRtDispatch_GoogleTensor.so) из
 *  dynamic-feature при local-testing — без него DISPATCH_OP не находит реализацию на TPU. */
class App : SplitCompatApplication() {

    // Держим дескрипторы, чтобы их не закрыл GC.
    private val reserved = mutableListOf<FileDescriptor>()

    override fun attachBaseContext(base: Context) {
        super.attachBaseContext(base)
        SplitCompat.install(this)
        reserveLowFds()
    }

    /**
     * EdgeTPU-диспетчер (libedgetpu_litert.so) падает `Unexpected fd from epoll: 0`,
     * если в процессе свободен дескриптор 0 — его eventfd получает fd=0, ломая bookkeeping.
     * Занимаем 0/1/2 тремя открытиями /dev/null, чтобы eventfd рантайма был ≥3.
     */
    private fun reserveLowFds() {
        runCatching {
            repeat(3) {
                reserved.add(Os.open("/dev/null", OsConstants.O_RDWR, 0))
            }
        }
    }
}
