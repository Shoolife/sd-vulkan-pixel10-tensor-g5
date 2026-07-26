# tools — перенос UNet на Tensor G5 TPU

Инструменты для второго бэкенда приложения: экспорт SD1.5 в `.tflite`, AOT-компиляция под
**Google Tensor G5**, запуск на EdgeTPU и замеры. Итоги и выводы — в разделе «TPU» корневого README.

**Разведка пройдена, путь рабочий:** весь UNet идёт на NPU (3328/3328 операций, ноль fallback),
точность `corr 0.999996`, forward **4.95 с**, генерация **31.9 с**.

> На TPU отдаётся **граф**, а не пишутся ядра — публичного ISA и аналога SPIR-V нет.
> Пакет `ai-edge-torch` переименован в **`litert-torch`** (`convert` только там; скрипты берут оба).
> Доступ к компилятору — [Tensor ML SDK Beta](https://developers.google.com/edge/litert/next/tensor-sdk),
> его бинарники в репозиторий не входят.

## Окружение

```bash
python -m venv tools/.venv && source tools/.venv/bin/activate
pip install -r tools/requirements-export.txt
```

## 1. Экспорт модели

```bash
python tools/export_unet_lcm.py                 # SD1.5 + LCM-LoRA (слитая) -> out/unet_lcm.tflite
python tools/export_unet_tflite.py --check      # базовый UNet + сверка corr с PyTorch
python tools/export_taesd.py                    # TAESD-декодер (5 МБ) вместо VAE (95 МБ)
python tools/dump_ops.py out/unet_lcm.tflite    # состав графа по операциям
```

LCM-веса обязательны: базовая SD1.5 на 4 шагах даёт размытую картинку.
`export_taesd.py` заодно печатает corr TAESD против полного VAE и кладёт обе картинки в `out/`.

## 2. Компиляция под Tensor G5

Компилятор Google собран под Ubuntu 22.04 и на свежем хосте (glibc 2.43) падает с `INTERNAL`,
поэтому запускается в контейнере:

```bash
docker build --network host -t tensor-aot -f tools/Dockerfile.aot tools/    # нужен tools/_sdk.tar.gz из SDK
docker run --rm --network host -v "$PWD/tools:/work" tensor-aot \
    python3 /work/docker_compile.py /work/out/unet_lcm.tflite /work/out/compiled bfloat16
```

Только **bfloat16**: усечение до fp16 даёт NaN на всех выходах (узкий экспонент).
Успех выглядит как `Subgraph 0 fully compiled: 3328 / 3328 ops offloaded to 1 partitions.`
Скрипт дополнительно печатает содержимое `/tmp/*.error`, которое сам компилятор прячет.

Альтернатива без Docker — Colab: [`colab_tensor_aot.ipynb`](colab_tensor_aot.ipynb).

## 3. Запуск на устройстве

EdgeTPU не пускает приложение под его uid (allowlist Google), поэтому UNet считает shell-демон,
а приложение ходит к нему по TCP на `127.0.0.1:8763` (файловый IPC не работает — sdcardfs даёт EACCES):

```bash
NDK=$ANDROID_HOME/ndk/*/toolchains/llvm/prebuilt/linux-x86_64/bin
$NDK/aarch64-linux-android31-clang tools/npu/litert_tpu_daemon.c \
    -I tools/npu/cc_sdk -L tools/npu/libs -lLiteRt -lm -o tools/npu/litert_tpu_daemon

adb push tools/npu/litert_tpu_daemon /data/local/tmp/
adb push out/compiled/unet_lcm_*_Google_Tensor_G5.tflite /data/local/tmp/unet_lcm_g5.tflite
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp \
    setsid ./litert_tpu_daemon unet_lcm_g5.tflite /data/local/tmp 8763 < /dev/null > daemon.log 2>&1 &"
```

`setsid` обязателен, иначе демон умирает вместе с adb-сессией. Демон сам читает scale/zero из
модели, поэтому int8-модели работают без изменений протокола (наружу всегда float32).

## 4. Замеры и сверка точности

```bash
# один forward на TPU: время + дамп выхода
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp \
    ./litert_tpu_bench unet_lcm_g5.tflite /data/local/tmp 5 /data/local/tmp/ref /data/local/tmp/ref/out.bin"

python tools/prep_inputs.py out/unet_lcm.tflite npu/ref     # входы под dtype/scale модели
python tools/corr_out.py out/unet_lcm.tflite npu/ref/out.bin npu/ref/ref_fp32.bin
python tools/run_ref.py out/unet_lcm.tflite out.bin ref.bin # то же на CPU-интерпретаторе
python tools/bench_cpu.py out/unet_lcm.tflite               # десктопный baseline
DELEGATE=gpu ./tools/bench_device.sh out/unet_lcm.tflite    # CPU/GPU-делегаты на устройстве
```

**corr одного forward обманчив**: в диффузии ошибка накапливается по шагам, и corr 0.988 уже даёт
разрушенную картинку. Проверять только полной генерацией — см. таблицу int8 в корневом README.

## 5. Квантизация (проверено, в проде не используется)

```bash
python tools/collect_calib.py out/calib_real.npz          # реальные latent/t/ctx с траектории LCM
python tools/quantize_int8.py out/unet_lcm.tflite out/unet_lcm_int8.tflite 4 CHANNELWISE out/calib_real.npz
SKIP_OPS="MUL,SUB" python tools/quantize_int8.py ...      # смешанная точность
ALGO=MSE python tools/quantize_int8.py ...                # алгоритм подбора шкал (MSE/OCTAV)
python tools/quantize_wo8.py                              # weight-only, без калибровки
```

Калибровать только реальными данными: CLIP-эмбеддинги доходят до ±33, синтетика `N(0,1)` даёт ±4
и негодные шкалы. Калибровка требует **~57 ГБ RAM** (режим всегда `PRESERVE_ALL_TENSORS`,
self-attention 64×64 даёт матрицы 8×4096×4096) — результат кэшируется в `out/*_qsv_*.pkl`,
повторно этот этап не нужен (с кэшем пик 10 ГБ).

`quantize_int8.py` содержит обходы трёх багов `ai-edge-quantizer` 0.8.0: усреднение min/max через
дефолтный аргумент реестра операций, падение на весах >32 МБ и падение OCTAV на скалярах.
