# Дорожная карта: собственный GPU-движок SD1.5 на Vulkan compute

## Зачем

На Google Tensor G5 (GPU = Imagination **PowerVR** DXT-48) недоступны быстрые пути:
- **TPU/NPU** — бета-рантайм Tensor сломан (нет рабочей комбинации runtime+dispatcher), компилятор не тянет монолит UNet.
- **MediaPipe ML Drift** (движок Google, ~15с/20шагов) — крашит на Tensor: его OpenCL-ядра генерят некорректные address space, которые прощает Adreno/Mali, но не PowerVR. Исходников нет (закрытый), фикса нет.
- **LiteRT GPU-делегат** — работает, но generic: fp16 даёт мусор (баг делегата, не математики — проверено: fp16 corr 1.0 с fp32 на CUDA), FP32 = 7с/шаг.

**Вывод:** единственный путь к контролю и скорости — свой GPU-backend. Доказано: свои Vulkan-шейдеры работают на PowerVR (где ML Drift падает) и оптимизируются.

## Текущий статус (фундамент готов)

- Vulkan compute инфраструктура: instance/device/buffers/pipeline/dispatch (`cpp/vk_bench.cpp`).
- GEMM-ядро (matmul) с оптимизацией: **26 → 186 GFLOPS fp16** (×7), register blocking + vec4 + fp16.
- Пик G5 = 1689.6 GFLOPS fp32. Сейчас ~10% пика, упор в memory/latency (не ALU).

## Что нужно для полного SD1.5 (список GPU-ядер)

| Ядро | Где | Сложность | Статус |
|---|---|---|---|
| **GEMM** (matmul) | Linear, attention | средняя | ✅ есть (дожать) |
| **Conv2d** 3×3/1×1 | UNet/VAE — главная стоимость | **высокая** (Winograd/im2col) | ⬜ |
| **Fused attention** (Q·K·softmax·V) | UNet self/cross-attn | **высокая** (flash-стиль) | ⬜ |
| **GroupNorm** | UNet/VAE | средняя (reduction) | ⬜ |
| **LayerNorm** | CLIP | низкая | ⬜ |
| **Elementwise** (SiLU/GELU/add/scale) | везде | низкая | ⬜ |
| **Upsample/Downsample** | UNet/VAE | низкая | ⬜ |
| **Embedding lookup** | CLIP, time-emb | низкая | ⬜ |

## Этапы

**Фаза 1 — Фундамент** (≈готова)
GEMM + инфраструктура исполнения. Далее: tensor-абстракция (буфер+форма+dtype), загрузка наших fp16-весов (.bin) на GPU, граф выполнения (очередь dispatch без CPU-синхронизации между ядрами).

**Фаза 2 — Ядра** (ядро проекта)
Каждое ядро: реализация → тест корректности против PyTorch (corr>0.999) → бенч.
Порядок: elementwise → groupnorm/layernorm → conv2d (старт im2col+GEMM, потом Winograd 3×3) → fused attention → upsample.

**Фаза 3 — Сборка UNet**
Граф из ядер по архитектуре SD1.5 UNet, прогон 1 forward, сверка выхода с PyTorch (численная отладка — самая долгая часть, как было с LiteRT).

**Фаза 4 — CLIP + VAE**
Остальные компоненты пайплайна. CLIP лёгкий, VAE decoder среднй.

**Фаза 5 — Оптимизация скорости**
Double buffering, fused-операции (norm+act, conv+bias+act), тюнинг workgroup/bank-conflicts под PowerVR, профилирование. Цель шага UNet → 0.5-1с.

**Фаза 6 — Интеграция**
LCM scheduler loop (уже есть на CPU) + наши ядра вместо LiteRT. Замена GpuMonoPipeline.

## Целевая производительность

При GEMM ~500 GFLOPS (дожатый) + Winograd-conv + fused-attention:
- UNet шаг ~0.5-1с (batch=1 fp16)
- 4 LCM-шага + VAE + CLIP ≈ **5-8с** — уровень/лучше бенчей Google, но на Tensor где их движок не работает.

## Риски и честная оценка

- **Объём**: полноценный inference-движок = недели интенсивной работы (Google делал командой). Каждое ядро (особенно conv2d, fused-attention) — самостоятельная задача.
- **Conv2d и fused-attention** — экспертный уровень GPU-программирования; основной риск по срокам.
- **Численная отладка** всего графа — долгая (опыт LiteRT: много времени на «почему мусор»).
- **Тюнинг под PowerVR** — diminishing returns, нужен профайлер.

**Стратегия снижения риска**: инкрементально, каждое ядро проверяется отдельно (корректность+скорость), движок собирается снизу вверх. На любом этапе есть рабочий результат. Параллельно остаётся рабочий LiteRT GPU+LCM (31с) как fallback.
