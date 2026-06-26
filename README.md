# Generet AI — локальная генерация изображений (спайк: GPU baseline)

Цель проекта — самое производительное приложение **локальной** генерации изображений на
телефоне. Тестовое устройство — **Pixel 10 Pro (Google Tensor G5, 16 ГБ RAM)**.

Ключевая особенность рынка: лучшие конкуренты (Local Dream, Off Grid) ускоряются через
**Qualcomm Hexagon NPU** и на Tensor падают в медленный CPU/GPU-фолбэк. Мы целимся в эту
щель: сначала честный **GPU baseline** на LiteRT, затем — **TPU Tensor** через Tensor ML SDK.

> Подробный разбор рынка и стратегии — в истории обсуждения. Маршрут моделей: SD1.5 → SDXL.

---

## Что делает текущий спайк

Меряет **чистую латентность инференса** компонентов Stable Diffusion 1.5 на устройстве
через **LiteRT** (классический `Interpreter` + делегаты) и проецирует полное время
генерации одной картинки.

- Движок: `app/.../engine/Benchmark.kt` — один и тот же код для трёх бэкендов
  (`AcceleratorChoice`): **GPU** (OpenCL делегат), **CPU** (XNNPACK), **NPU** (NNAPI —
  на Tensor может уйти на TPU уже сейчас). Переход на LiteRT Next `CompiledModel` /
  Tensor ML SDK позже локализован в этом одном файле.
- Меряем компоненты по отдельности (`engine/SdSpike.kt`):
  `Text Encoder`, **`UNet`** (доминирующая стоимость), `VAE Decoder`.
- Проекция: `total = textEncoder·1 + unet·(steps·cfg) + vaeDecoder·1`, считается для
  **20 шагов** (стандарт) и **4 шагов** (few-step / LCM).

Значения входов не пишем намеренно: размеры буферов берём из модели (интроспекция
тензоров), а для замера латентности важен только размер — плотный UNet на GPU не
ветвится по данным. Реальная генерация картинки (токенизатор, scheduler, VAE→Bitmap) —
**следующий этап**, не часть baseline.

> ⚙️ Примечание про LiteRT Next: API `CompiledModel`/`Environment`/`Accelerator` на момент
> сборки есть только в nightly-снапшотах (стабильные `com.google.ai.edge.litert:litert` —
> до `1.3.0`, где этого API нет). Поэтому baseline сделан на стабильном `Interpreter` API.

---

## Как запустить

### 1. Собрать и поставить приложение

Нужен JDK 17. Версии запинены и **проверены сборкой** на этом репозитории:
Kotlin `2.2.10`, Compose BOM `2025.06.00`, LiteRT `1.3.0`, `compileSdk = 37`,
AGP `9.2.1` (Kotlin встроен в AGP — отдельный `kotlin-android` плагин не применяется).

```bash
./gradlew :app:installDebug
```

> Проверено: собирается, ставится и запускается на Pixel 10 Pro (Android 17) без падений.

### 2. Получить .tflite модели SD1.5

Модели НЕ в APK (вместе ~1–2 ГБ). Нужны три файла:

| Файл | Что это |
|------|---------|
| `text_encoder.tflite` | CLIP text encoder |
| `unet.tflite`         | denoiser (главный по времени) |
| `vae_decoder.tflite`  | латент → пиксели |

Способ конвертации (рекомендуется) — **ai-edge-torch** (генеративные примеры включают
Stable Diffusion: экспорт CLIP / diffusion / decoder в `.tflite`):
<https://github.com/google-ai-edge/ai-edge-torch> → `ai_edge_torch/generative/examples/stable_diffusion`.

Переименуй выходные файлы под имена из таблицы (или поправь `fileName` в `engine/SdSpike.kt`).

### 3. Залить модели на устройство

```bash
adb shell mkdir -p /sdcard/Android/data/com.example.generet_image_ai/files/models
adb push text_encoder.tflite /sdcard/Android/data/com.example.generet_image_ai/files/models/
adb push unet.tflite         /sdcard/Android/data/com.example.generet_image_ai/files/models/
adb push vae_decoder.tflite  /sdcard/Android/data/com.example.generet_image_ai/files/models/
```

### 4. Прогнать

Открой приложение → выбери ускоритель (**GPU** для baseline) → «Запустить бенчмарк».
Получишь per-компонент латентность (avg / p50 / min / max) и проекцию времени на картинку
для 20 и 4 шагов.

---

## Как читать результат

- **UNet avg × шаги × CFG** — почти всё время генерации. Если оно велико — это цель оптимизации.
- **Проекция «4 шага»** показывает потенциал few-step/LCM-дистилляции (наш алгоритмический рычаг).
- **CFG-фактор = 2** консервативен; если экспорт батчит cond/uncond в один проход, реальное
  время вдвое меньше по вкладу UNet.

## Дорожная карта

1. ✅ Скелет (Kotlin + Compose + LiteRT) и **GPU baseline** (этот спайк).
2. ⏭ Полный t2i-пайплайн на GPU: токенизатор + scheduler + CFG + VAE→Bitmap (реальные картинки).
3. ⏭ Few-step / LCM модель → проверить проекцию «4 шага» на практике.
4. ⏭ **TPU Tensor**: AOT-компиляция UNet под Tensor ML SDK, переключение `Accelerator.NPU`.
5. ⏭ SDXL (используем 16 ГБ RAM), затем LoRA / inpainting / ControlNet.
