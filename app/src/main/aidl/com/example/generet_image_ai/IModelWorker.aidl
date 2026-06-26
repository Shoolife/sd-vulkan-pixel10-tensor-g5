package com.example.generet_image_ai;

/** Воркер в отдельном процессе: держит ОДНУ модель открытой и гоняет её много раз.
 *  Тензоры передаются через файлы в filesDir/ipc (Binder мал для 50МБ-скипов). */
interface IModelWorker {
    /** Загрузить модель (.tflite из filesDir/models) на NPU и держать открытой. */
    void load(String modelFileName, boolean npu);

    /** Прогон: читает входы из inFiles[], пишет выходы в outFiles[] (raw float32). */
    void run(in String[] inFiles, in String[] outFiles);

    /** Закрыть текущую модель/окружение (для единичного перехода clip→a, b2→vae). */
    void release();
}
