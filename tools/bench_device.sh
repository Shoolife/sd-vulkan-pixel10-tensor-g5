#!/usr/bin/env bash
# Device-замер .tflite на Pixel 10 (Tensor G5) через LiteRT benchmark_model + adb.
# Заготовка: переключает делегат CPU / GPU / Tensor-TPU одной переменной.
#
# Требуется:
#   - adb + подключённый Pixel 10 (USB-debug).
#   - бинарь benchmark_model под arm64 (из LiteRT):
#       https://ai.google.dev/edge/litert/models/measurement  (готовый APK/бинарь)
#     либо собрать: bazel build -c opt --config=android_arm64 \
#       //tensorflow/lite/tools/benchmark:benchmark_model
#   - для DELEGATE=tensor — Tensor ML SDK Beta: его stable-delegate .so + settings json
#       https://developers.google.com/edge/litert/next/tensor-sdk
#
# Использование:
#   DELEGATE=cpu    ./bench_device.sh out/unet_fp16.tflite
#   DELEGATE=gpu    ./bench_device.sh out/unet_fp16.tflite
#   DELEGATE=tensor ./bench_device.sh out/unet_fp16.tflite   # нужен SDK Beta (см. ниже)
set -euo pipefail

MODEL="${1:?укажи .tflite, напр. out/unet_fp16.tflite}"
DELEGATE="${DELEGATE:-cpu}"
RUNS="${RUNS:-20}"
WARMUP="${WARMUP:-3}"
THREADS="${THREADS:-8}"
DEV=/data/local/tmp
BM="${BENCHMARK_BIN:-$DEV/benchmark_model}"   # путь к бинарю на устройстве

base=$(basename "$MODEL")
echo ">> push $MODEL -> $DEV/$base"
adb push "$MODEL" "$DEV/$base" >/dev/null

# Флаги делегата. Tensor берёт partial-delegation report (что лёг на TPU / в фолбэк).
case "$DELEGATE" in
  cpu)    FLAGS="--use_xnnpack=true --num_threads=$THREADS" ;;
  gpu)    FLAGS="--use_gpu=true" ;;
  tensor)
    # Tensor-TPU подключается как stable delegate из SDK Beta.
    # Залей на устройство его .so и settings json, путь укажи через переменные:
    : "${TENSOR_DELEGATE_SO:?установи путь к Tensor stable-delegate .so на устройстве}"
    : "${TENSOR_SETTINGS_JSON:?установи путь к settings json делегата на устройстве}"
    FLAGS="--stable_delegate_settings_file=$TENSOR_SETTINGS_JSON"
    echo "   (Tensor delegate: $TENSOR_DELEGATE_SO)"
    ;;
  *) echo "DELEGATE должно быть cpu|gpu|tensor"; exit 1 ;;
esac

echo ">> benchmark ($DELEGATE), runs=$RUNS warmup=$WARMUP"
adb shell "$BM \
  --graph=$DEV/$base \
  --num_runs=$RUNS \
  --warmup_runs=$WARMUP \
  --enable_op_profiling=true \
  $FLAGS"

echo ""
echo "Смотри в выводе: 'Inference (avg)' = время forward, и профиль по ops."
echo "Для go/no-go важно: сколько узлов делегировано на TPU vs осталось на CPU (partial delegation)."
echo "Ориентир: наш Vulkan-движок = ~9 с/forward (warm 32.9 с / 4 шага batch-CFG B=2)."
