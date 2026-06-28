# Архитектура движка Generet AI (свой Vulkan-движок SD1.5)

Документ описывает архитектуру **снизу вверх** — от примитивов Vulkan до полной
генерации картинки. Каждый уровень строится на предыдущем. Ядро движка —
`app/src/main/cpp/vk_engine.cpp` (нативный C++/Vulkan) + Kotlin-обвязка.

Цель проекта: локальная генерация изображений на Pixel 10 Pro (Google Tensor G5,
GPU **PowerVR D-Series DXT-48-1536**), где стек Google (ML Drift / TPU AOT) падает.
**UNet — полностью наш** на Vulkan; CLIP и VAE — через LiteRT.

---

## Уровень 0 — Фундамент Vulkan (абстракции железа)

Три базовые C++-структуры поверх Vulkan API:

| Структура | Роль |
|---|---|
| **`VkCtx`** | Обёртка устройства: `VkInstance`/`VkDevice`/очередь/командный пул; выделение памяти (`alloc`); заливка/скачивание через staging (`upload`/`download`); детект возможностей (fp16, cooperative matrix, subgroup=128, shared=32КБ, maxBuffer=128МБ, VRAM 16ГБ). Один на движок (`gCtx`, persistent). |
| **`Buf`** | Буфер + его GPU-память. Один блок данных (тензор активаций или веса). |
| **`Kernel`** | Один **compute-pipeline из SPIR-V**: descriptor set layout (привязка буферов), pipeline layout (push-константы), скомпилированный pipeline, пул дескрипторов. Методы `create / makeSet / record / resetPool / destroy`. |

Связующее звено — **`op()`**: пишет диспетч одного ядра в **единый command buffer**
(`gCmd`) с барьером памяти (`vkCmdPipelineBarrier`). Pipeline'ы **кэшируются**
(`gK[NSH]` — каждый шейдер компилируется один раз), весь forward = один `vkQueueSubmit`
+ один `vkQueueWaitIdle`. Это дало 354с→251с на старте оптимизации.

**Факты о железе (из дампа драйвера в `VkCtx::init`):** PowerVR DXT-48-1536, Vulkan 1.4,
shared 32КБ, subgroup 128, maxStorageBuffer 128МБ, FP32-пик ≈1.67 TFLOPS (под нагрузкой
троттлит). Cooperative matrix поддержан, но **эмулирован** (×5-15 медленнее FMA) — не используем.

---

## Уровень 1 — Compute-ядра (шейдеры `.comp` → `.spv`)

Примитивные операции, каждая — GLSL compute-шейдер (компиляция `glslc --target-env=vulkan1.1`,
кладётся в `assets/shaders/`). **Фундамент всего — `matmul`** (GEMM).

```
matmul (GEMM)  ◄── базовый кирпич. 128×128 тайл, vec4-A, BK=4, fp32-накопление,
                   ~253 GFLOPS (≈практический потолок на троттлящем GPU)
matmul_wino    — батч-GEMM для Winograd (36 спектральных позиций ξ через gl_WorkGroupID.z)
im2col         — лоуэринг свёртки в матрицу (со stride)
winograd_in/out/wt — трансформы Winograd F(4×4,3×3): B^T·d·B, A^T·m·A, G·g·G^T
conv2d         — прямая (наивная) свёртка, fallback
attention / attention_big — flash-attention (онлайн-softmax, TQ=128=полный subgroup)
groupnorm / groupnorm_silu — нормализация по группам (+слитый SiLU)
layernorm, silu, geglu (erf-gelu), addbias/addbias2, add,
split_heads/merge_heads (мультиголовый reshape),
t_chw2hwc/t_hwc2chw (транспоз layout), upsample (nearest 2×)
```

Ключевые оптимизации ядер:
- **matmul**: векторизация чтения A (vec4 вместо 8 скаляров), BK=4 для occupancy.
- **conv → Winograd F(4×4,3×3)**: 4× экономия FLOPs на 3×3, до 530 GFLOPS на 64².
- **attention TQ=128**: полный subgroup PowerVR (было BT=64 = 50% простоя лейнов).

---

## Уровень 2 — Операции (C++ хелперы, `namespace unet`)

Обёртки, выбирающие и параметризующие ядра:

- **`matmul()`** — линейные слои.
- **`conv()`** — умный диспетчер по форме:
  - `1×1` → чистый `matmul` (свёртка 1×1 = GEMM)
  - `3×3 stride1, H≥32` → **Winograd** (in-transform + 36 батч-GEMM + out-transform)
  - `3×3 H≤16` → `im2col + matmul` (Winograd там хуже: мало тайлов)
  - `stride2` (downsample) → `im2col + matmul`
  - слишком большой Col (>128МБ) → **N-тайлинг** по столбцам
- `groupnorm_silu()`, `silu()`, `addbias_c/l()`, `addv()`, attention через `op()`.
- **`W(name)`** — ленивая загрузка веса: fp16-файл → конверсия в fp32 на GPU → кэш
  (`WC_`, persistent между генерациями). **`Wwino()`** — кэш Winograd-весов U (по buf-хэндлу).
- `mk()` — временный буфер (scratch, освобождается после forward).

**Принцип уровня:** conv и линейные слои **сводятся к matmul** — ускорив фундамент,
ускорили всё выше.

---

## Уровень 3 — Блоки нейросети

Собираются из операций уровня 2:

- **`resnet()`**: `groupnorm_silu → conv → (+ проекция time-embedding) → groupnorm_silu →
  conv → + residual` (+ conv_shortcut если меняются каналы).
- **`transformer()`** (SpatialTransformer):
  `groupnorm → proj_in(1×1) →`
  **self-attn** (LN → q/k/v → flash-attention → to_out → +residual) `→`
  **cross-attn** (LN → q; k/v из контекста промпта → attention → to_out → +residual) `→`
  **ff** (LN → GEGLU → linear → +residual) `→ proj_out(1×1) → + residual`.
- **`downsample`** (conv stride2), **`upsample`** (nearest 2× + conv),
  **`concat`** (склейка по каналам = копии буферов в смежную память).

---

## Уровень 4 — Граф UNet (`runGraph`)

Собирает блоки в полный SD1.5 UNet:

```
lat[4,64,64] + tembProj[320] + ctx[77,768]
   │
time_embedding MLP          conv_in (4→320)
   │                            │
   └────────────►  DOWN[0..3]  (resnet + transformer + downsample;
                      │          копит 12 skip-связей; 64²→32²→16²→8²)
                     MID  (resnet + transformer + resnet; 8×8, 1280 каналов)
                      │
                   UP[0..3]  (concat skip + resnet + transformer + upsample;
                      │        8²→16²→32²→64²)
                conv_out (320→4)
                      ▼
                 noise[4,64,64]
```

Каналы по уровням: `down out = {320, 640, 1280, 1280}`, `up out = {1280, 1280, 640, 320}`.
12 skip-связей идут из down-пути в up-путь по принципу LIFO.

---

## Уровень 5 — JNI-мост (C++ ↔ Kotlin)

- **`unetInit(shaders, dir)`** — поднимает ctx + шейдеры + путь к весам (1 раз; веса резидентны).
- **`unetForward(lat, temb, ctx, mode, slot)`** → noise (один проход UNet).
- `unetRelease` — освобождает веса/кэши/ctx.
- `unetSelfTest` — сверка полного графа с эталоном PyTorch (**relErr = 0.0003**).
- `unetProfile` — раскладка времени forward по категориям операций.

---

## Уровень 6 — Пайплайн генерации (`VulkanUnetPipeline.kt`)

```
промпт → CLIP (LiteRT, на CPU — GPU-делегат искажает context) → ctx[77,768]
   │
unetInit (веса резидентны между генерациями)
   │
LCM-цикл, 4 шага:
   на каждом шаге CFG = 2 прохода нашего UNet (cond + uncond)
   eps = uncond + cfgScale·(cond − uncond)
   latents = LcmScheduler.step(latents, eps, шум)
   │
финальные latents → VAE (LiteRT, GPU) → пиксели [-1,1] → Bitmap
```

---

## Уровень 7 — UI

`GenerateViewModel` (выбор движка, состояние, поток GPU) + Compose-экран
(промпт, кнопки «Сгенерировать»/«Vulkan бенч», картинка).

---

## Как всё связано — главная идея

**Строгая иерархия зависимостей:**

```
Vulkan (уровень 0)
   └─ matmul (уровень 1)
        └─ {conv, linear, attention} (уровень 2)
             └─ {ResNet, Transformer} (уровень 3)
                  └─ стадии down/mid/up (уровень 4)
                       └─ UNet forward (уровень 4)
                            └─ LCM-денойз + CFG (уровень 6)
                                 └─ картинка
```

Оптимизация **снизу вверх** сработала именно из-за этой иерархии: ускорили **matmul**
(фундамент) → автоматически быстрее **conv** (= im2col/Winograd + matmul) и линейные слои
→ быстрее **блоки** → быстрее **весь forward** → быстрее **генерация**.

**Данные:** везде fp32, layout `[каналы, H×W]` (channels-first). Веса хранятся fp16 на
диске (~1.7ГБ), конвертируются в fp32 в памяти (~3.4ГБ, резидентны между генерациями).
CLIP/VAE — чужие (LiteRT), **UNet — полностью наш Vulkan**.

---

## Раскладка времени (warm-картинка ≈ 43с)

| Этап | Доля |
|---|---|
| CLIP (CPU) | ~5% |
| **денойз (4 шага × 2 forward CFG)** | **~87%** |
| VAE (LiteRT GPU) | ~8% |
| unet init (веса резидентны) | ~0% |

Один forward ≈ 4.7с; внутри: matmul ~39%, attention ~17%, Winograd-conv, остальное.

---

## История оптимизации (354с → 43с warm, ×8.2)

| Шаг | Картинка |
|---|---|
| старт (op-per-op) | 354с |
| single command buffer + кэш pipeline'ов | 251с |
| matmul 155→253 GFLOPS (vec4-A, BK=4) | — |
| conv → im2col+matmul | 100с |
| self-attn + attention_big на полный subgroup | 87с |
| Winograd F(4×4,3×3) на 3×3 | — |
| persistent веса между генерациями | 56с→43с |

**Проверено и отклонено:** cooperative matrix (эмуляция на PowerVR, медленнее),
DeepCache (несовместим с 4-шаговым LCM: relErr ~20%, т.к. LCM уже выжал межшаговую
избыточность). Точность движка везде сохранена: **relErr 0.0003 vs PyTorch**.

**Оставшиеся рычаги ускорения (все с компромиссом):** batch-CFG (lossless, ~1.5-1.8×,
большой рефактор), fp16 packed (~2×, риск точности), меньше шагов/CFG (риск качества).
