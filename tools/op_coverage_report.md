# Op-coverage SD1.5-блоков (этап 1 разведки TPU)

Сгенерировано автономно: `export_blocks_tflite.py` (random-init, офлайн, diffusers 0.38.0 +
torch 2.12.1 + litert-torch 0.9.1 — **все последние версии**) → два `.tflite` → `dump_ops.py`.
Это op-coverage **на уровне графа** (какие op-типы есть в модели), а не «что лёг на TPU» —
последнее требует Tensor ML SDK Beta + физический Pixel 10.

## Resnet-блок (down_blocks[0].resnets[0])
66 операций, 13 типов. Тяжёлое: `CONV_2D ×2`, `FULLY_CONNECTED ×1` (temb-проекция).
GroupNorm разложен в кластер `SUM/SUB/RSQRT/MUL/RESHAPE`; SiLU = `LOGISTIC + MUL`.
Кандидаты на CPU-фолбэк: **RSQRT, TRANSPOSE, LOGISTIC**.

## Transformer-блок (down_blocks[0].attentions[0])
120 операций, 18 типов. Тяжёлое: `FULLY_CONNECTED ×10` (q/k/v/o + GEGLU + proj_in/out),
`BATCH_MATMUL ×4` (QK^T и PV в attention), `CONV_2D ×2`.
LayerNorm/GroupNorm → `MEAN/SQUARED_DIFFERENCE/RSQRT/SUB`; внимание → `SOFTMAX ×2`;
FF-активация → `GELU ×1`. **38 RESHAPE + 10 TRANSPOSE** — перестановки голов/формы.
Кандидаты на CPU-фолбэк: **RSQRT, TRANSPOSE, MEAN, SQUARED_DIFFERENCE, BATCH_MATMUL, SOFTMAX, GELU**.

## Вывод (на уровне графа)

- 🟢 **Главная стоимость — стандартные ops**: matmul = `FULLY_CONNECTED`/`BATCH_MATMUL`,
  свёртки = `CONV_2D`. Это «родной» для матричного TPU класс — почти наверняка ляжет на TPU.
- 🟡 **Главный риск — не сами матмулы, а обвязка**:
  1. **Нормализации не атомарны** — GroupNorm/LayerNorm разложены в кластер
     `MEAN/SUM/SQUARED_DIFFERENCE/RSQRT/SUB/MUL`. Сфьюзит ли их компилятор Tensor в TPU-узел
     или выкинет на CPU — открытый вопрос (классическая болевая точка NPU-делегатов).
  2. **Много RESHAPE/TRANSPOSE** (в transformer 38+10) вокруг attention. Если хоть один паттерн
     не поддержан, граф фрагментируется на много мелких TPU-островов с CPU-перескоками между ними —
     это убивает скорость, **даже если каждый отдельный op «поддержан»**.
  3. `SOFTMAX`, `GELU`, `BATCH_MATMUL` с динамическими формами — типовые «может/не может» у NPU.

## Что это меняет

Go/no-go теперь **точно упирается в одно**: отчёт делегации Tensor-компилятора по этим двум
`.tflite` — сколько узлов он берёт на TPU и не дробит ли граф на острова. Граф готов, всё
автономно-проверяемое сделано. Остаётся device-шаг (Tensor ML SDK Beta + Pixel 10):
```
# скомпилировать под Tensor, прочитать partial-delegation report
# (точные CLI — из Tensor ML SDK Beta: developers.google.com/edge/litert/next/tensor-sdk)
```
Если нормализации и attention-обвязка уходят в CPU-фолбэк — пробуем переписать паттерны
в графе экспорта (правка тут, не в Vulkan-движке) до того, как делегировать.

## Полный UNet (этап 2 — прогнан с настоящими весами SD1.5)

`export_unet_tflite.py --check` (fp32, веса `stable-diffusion-v1-5`): экспорт **прошёл**,
`out/unet.tflite` = **3440 МБ**. **Численная сверка с PyTorch-эталоном идеальна:
relErr=0.0000, corr=1.000000** — tflite-граф эквивалентен PyTorch (экспорт корректен).

op-coverage всей модели: **3393 операции, 23 типа**. Тяжёлое (🟢): `FULLY_CONNECTED ×184`,
`CONV_2D ×98`, `BATCH_MATMUL ×64`. Новые относительно блоков, но стандартные/низкорисковые:
`RESIZE_NEAREST_NEIGHBOR ×3` (upsample), `CONCATENATION ×13` (skip-связи), `SIN/COS` (time-embed),
`PAD ×3`. **Новых рискованных типов сверх блоков НЕ появилось** — те же 8 кандидатов на фолбэк
(LOGISTIC, TRANSPOSE, RSQRT, MEAN, SQUARED_DIFFERENCE, BATCH_MATMUL, SOFTMAX, GELU).

Главное подтверждение масштаба: **RESHAPE ×924 + TRANSPOSE ×266** — огромная обвязка вокруг
attention и нормализаций. Нормализации разложены массово (`RSQRT ×109`, `MEAN ×96`,
`SQUARED_DIFFERENCE ×48`). Это и есть главный риск фрагментации графа на TPU-острова.

**Итог этапа 2:** экспорт всей модели работает и численно точен; набор операций не преподнёс
сюрпризов. Go/no-go по-прежнему упирается ТОЛЬКО в отчёт делегации Tensor-компилятора
(SDK Beta + Pixel 10) — сколько из этих 3393 узлов он берёт на TPU и не дробит ли граф.

### fp16-вариант (этап 3 по точности — прогнан)
`--dtype fp16` через `full_fp16_recipe()` (хранение весов fp16, compute fp32 в CPU-интерпретаторе):
**unet_fp16.tflite = 1722 МБ (ровно 2× меньше fp32), relErr=0.0001, corr=1.000000** — fp16-хранение
практически не меняет выход. Это формат, ближайший к нашему боевому fp16-движку.

### Десктоп-baseline (bench_cpu.py)
forward fp16 на CPU ПК (8 потоков, B=1): **~5.5 с/forward** (med 5481 мс). Это лишь опорная точка
ПК-CPU; реальные TPU/GPU-числа даёт `bench_device.sh` на Pixel 10. Ориентир Vulkan-движка ≈ 9 с/forward.

### Замеры на РЕАЛЬНОМ Pixel 10 Pro (Tensor G5, benchmark_model, unet_fp16, B=1)

| Backend | forward | Делегировано | Примечание |
|---|---|---|---|
| **Наш Vulkan-движок** | **≈9 с** | весь граф (рукописно) | эталон |
| LiteRT CPU (XNNPACK, 8 пот.) | **≈10.5 с** (avg 1.054e7 us) | 3483/3610 узлов | init 7.4с, память ~5 ГБ |
| LiteRT GPU-делегат | **≈25 с** (avg 2.500e7 us) | **175/3610** узлов, 3 партиции | init 12.1с, precision_loss_allowed |

**GPU-делегат провалился из-за фрагментации:** `GATHER_ND` не поддержан GPU-делегатом → на GPU ушло
лишь ~5% графа, остальное на CPU, постоянные перекидывания CPU↔GPU на каждом forward → 25с, хуже
чистого CPU. **Это эмпирическое подтверждение риска фрагментации из op-coverage** (RESHAPE×924 +
TRANSPOSE×266 + GATHER_ND + разложенные нормализации).

**Вывод:** наш рукописный Vulkan (≈9с) обгоняет штатный LiteRT-GPU (≈25с) на том же PowerVR в **2.8×** —
потому что мы написали каждый op и граф не дробится. «Сконвертировать в tflite и отдать делегату»
на diffusers-графе не работает. Для TPU (иной, более мощный AOT-компилятор) это НЕ приговор, но
чтобы TPU взял граф целиком, экспорт, вероятно, придётся чистить (убрать GATHER_ND, фьюзить нормы).

## Воспроизвести
```bash
pip install -r tools/requirements-export.txt
python tools/export_blocks_tflite.py --out tools/out      # random-init, офлайн
python tools/dump_ops.py tools/out/*.tflite
```
