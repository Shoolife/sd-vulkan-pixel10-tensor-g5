#!/usr/bin/env python3
"""Static int8 (W8A8) квантизация fp32-tflite через ai-edge-quantizer.
Калибровка: latent~N(0,1), timestep из LCM-набора, ctx~N(0,1).

  quantize_int8.py [src] [dst] [n_calib] [granularity=CHANNELWISE|TENSORWISE]
"""

import os
import pickle
import sys

import numpy as np
from ai_edge_quantizer import algorithm_manager as am
from ai_edge_quantizer import qtyping, quantizer, recipe_manager
from ai_edge_quantizer.algorithms.uniform_quantize import octav
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as uqt
from ai_edge_quantizer.utils import qsv_utils

# --- обход бага ai-edge-quantizer 0.8.0 -------------------------------------
# Для весов >32 MiB (conv 1280x1280x3x3 в mid_block) uniform_quantize идёт в
# chunked-ветку: сплющивает тензор в 2D, но scale/zero_point после
# fix_quantization_params_rank имеют ранг тензора ((O,1,1,1) или (1,1,1,1)),
# и broadcast_to(scales_4D, 2D_shape) падает с "input operand has more
# dimensions than allowed by the axis remapping". Ветка корректна только если
# quantized_dimension — последняя ось. Считаем такие тензоры без чанкинга:
# промежуточный float того же размера (~59 MB) машина держит.
_orig_uniform_quantize = uqt.uniform_quantize


def _uniform_quantize_nochunk(
    tensor_data, quantization_params, is_blockwise_quant=False
):
    big = tensor_data.ndim > 1 and tensor_data.nbytes > 32 * 1024 * 1024
    if is_blockwise_quant or not big:
        return _orig_uniform_quantize(
            tensor_data, quantization_params, is_blockwise_quant
        )
    qp = uqt.fix_quantization_params_rank(tensor_data, quantization_params)
    qtype = uqt.IntType(qp.num_bits, signed=True)
    narrow = qp.symmetric and qp.num_bits >= 8
    ret = np.divide(tensor_data, qp.scale)
    ret = np.add(ret, qp.zero_point, out=ret)
    ret = uqt._round_and_clip_inplace(ret, qtype, narrow)
    return uqt.assign_quantized_type(ret, qtype)


uqt.uniform_quantize = _uniform_quantize_nochunk


# --- абсолютный min/max вместо moving average --------------------------------
# Calibrator по умолчанию берёт qsv_utils.moving_average_update (smoothing 0.95),
# а Quantizer.calibrate не даёт его подменить. Для timestep это фатально: значения
# 259..999 усреднились в диапазон 0..340, и 999 обрезался до 340 -> модель получала
# неверное время (corr 0.891). Берём честные min/max по всем сэмплам.
def _absolute_minmax_update(qsv, new_qsv):
    if not qsv:
        return new_qsv
    return {
        "min": np.minimum(qsv["min"], new_qsv["min"]),
        "max": np.maximum(qsv["max"], new_qsv["max"]),
    }


qsv_utils.moving_average_update = _absolute_minmax_update


# --- обход бага OCTAV при per-tensor активациях ------------------------------
# На СКАЛЯРНЫХ тензорах (ndim=0, в графе есть константы-скаляры) буфер маски получает
# форму (), а guess при axis=None — форму (1,), и np.greater_equal(..., out=mask) падает:
# "non-broadcastable output operand with shape () doesn't match (1,)".
# Достаточно поднять скаляр до 1-D — математика та же.
_orig_octav_guess = octav._guess_clipping_with_octav


def _octav_guess_fix(x, bits, axis, *args, **kwargs):
    if getattr(x, "ndim", 1) == 0:
        return _orig_octav_guess(np.atleast_1d(x), bits, axis, *args, **kwargs)
    return _orig_octav_guess(x, bits, axis, *args, **kwargs)


octav._guess_clipping_with_octav = _octav_guess_fix

# Главный путь усреднения — НЕ атрибут модуля, а дефолтный аргумент
# QuantizedOperationInfo.update_qsv_func в algorithm_manager_api (вычисляется при импорте,
# патч модуля его не достаёт). Проверено арифметикой: 0.95*999+0.05*759 -> ... -> 927.42,
# ровно то значение, что оказалось в кэше вместо max=999. Переписываем реестр операций.
_patched_ops = 0
for _algo_info in am._alg_manager_instance._algorithm_registry.values():
    for _op_info in _algo_info.quantized_ops.values():
        _op_info.update_qsv_func = _absolute_minmax_update
        _patched_ops += 1
print(f">>> абсолютный min/max: пропатчено {_patched_ops} операций реестра", flush=True)
# ---------------------------------------------------------------------------

src = sys.argv[1] if len(sys.argv) > 1 else "out/unet_lcm.tflite"
dst = sys.argv[2] if len(sys.argv) > 2 else "out/unet_lcm_int8.tflite"
n_calib = int(sys.argv[3]) if len(sys.argv) > 3 else 4
gran = qtyping.QuantGranularity(sys.argv[4] if len(sys.argv) > 4 else "CHANNELWISE")
calib_npz = sys.argv[5] if len(sys.argv) > 5 else "out/calib_real.npz"

LCM_TS = [999.0, 759.0, 499.0, 259.0]


def sample():
    return {
        "args_0": np.random.randn(1, 4, 64, 64).astype(np.float32),
        "args_1": np.array([np.random.choice(LCM_TS)], dtype=np.float32),
        "args_2": np.random.randn(1, 77, 768).astype(np.float32),
    }


def real_samples(path, limit):
    """Реальные входы с траектории LCM-денойза (collect_calib.py)."""
    d = np.load(path)
    lat, ts, ctx = d["lat"], d["ts"], d["ctx"]
    idx = np.linspace(0, len(lat) - 1, min(limit, len(lat))).astype(int)
    print(f">>> калибровка из {path}: {len(lat)} сэмплов, беру {len(idx)}", flush=True)
    print(
        f"    lat [{lat.min():.2f}, {lat.max():.2f}]  ts {sorted(set(ts.tolist()))}"
        f"  ctx [{ctx.min():.2f}, {ctx.max():.2f}]",
        flush=True,
    )
    return [
        {
            "args_0": lat[i : i + 1].astype(np.float32),
            "args_1": ts[i : i + 1].astype(np.float32),
            "args_2": ctx[i : i + 1].astype(np.float32),
        }
        for i in idx
    ]


act_bits = int(os.environ.get("ACT_BITS", "8"))  # 8 = W8A8, 16 = W8A16 (точнее)
# MIN_MAX_UNIFORM_QUANT исчерпан (corr 0.954 максимум): min/max растягивает шкалу
# на выбросы (ctx до ±33). OCTAV/MSE подбирают порог клиппинга, HADAMARD_ROTATION
# гасит сами выбросы вращением.
algo = os.environ.get("ALGO", "min_max_uniform_quantize")
rm = recipe_manager.RecipeManager()
if algo == "OCTAV":
    # OCTAV покрывает все 50 операций (включая SOFTMAX/BATCH_MATMUL/MUL) -> полный int8-граф,
    # но поддерживает только СИММЕТРИЧНЫЕ активации, поэтому конфиг задаём вручную.
    # Для сравнения: MSE зарегистрирован лишь для 5 op (conv/fc/embedding), из-за чего
    # остальной граф остаётся float — качество 0.9987, но ускорения нет (4.83 с).
    rm.add_quantization_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=act_bits, symmetric=True
            ),
            weight_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=8, symmetric=True, granularity=gran
            ),
            compute_precision=qtyping.ComputePrecision.INTEGER,
        ),
        algorithm_key=algo,
    )
else:
    rm.add_static_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        activation_num_bits=act_bits,
        weight_num_bits=8,
        weight_granularity=gran,
        algorithm_key=algo,
    )

# Смешанная точность: перечисленные операции остаются во float (NO_QUANTIZE).
# GroupNorm разложен на MEAN/SQUARED_DIFFERENCE/RSQRT/SUM — считать дисперсию в int8
# губительно для точности; SOFTMAX/LOGISTIC тоже чувствительны. Тяжёлая арифметика
# (CONV_2D 98 + FULLY_CONNECTED 184 + BATCH_MATMUL 64 = ~90% FLOPs) остаётся в int8.
skip_ops = [s for s in os.environ.get("SKIP_OPS", "").split(",") if s]
for _op in skip_ops:
    rm.add_quantization_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName(getattr(qtyping.TFLOperationName, _op)),
        op_config=None,
        algorithm_key=am.AlgorithmName.NO_QUANTIZE,
    )
if skip_ops:
    print(f">>> во float остаются: {', '.join(skip_ops)}", flush=True)

if calib_npz != "random" and os.path.exists(calib_npz):
    samples = real_samples(calib_npz, n_calib)
else:
    print(f">>> калибровка СИНТЕТИКОЙ N(0,1) ({calib_npz} нет)", flush=True)
    samples = [sample() for _ in range(n_calib)]
calib = {"serving_default": samples}
print(
    f">>> int8 W8A8, веса {gran.value}, калибровка {len(samples)} сэмплов", flush=True
)
qt = quantizer.Quantizer(src, rm.get_quantization_recipe())

# Калибровка кэшируется на диск: это единственный этап, требующий десятков ГБ
# (режим всегда PRESERVE_ALL_TENSORS, а self-attention 64x64 даёт матрицы
# 8x4096x4096 ~537 МБ, 5 блоков). Пройдя её один раз, дальше работаем с кэшем.
# кэш привязан к ИСХОДНОЙ модели, а не к dst: min/max не зависят от битности,
# поэтому W8A8 и W8A16 переиспользуют один и тот же 57-гигабайтный прогон.
qsv_cache = f"{os.path.splitext(src)[0]}_qsv_{algo}.pkl"
if os.path.exists(qsv_cache):
    with open(qsv_cache, "rb") as f:
        qsvs = pickle.load(f)
    print(f">>> калибровка из кэша {qsv_cache} ({len(qsvs)} тензоров)", flush=True)
else:
    # num_threads=2 вместо 16: каждый поток XNNPACK держит свою арену.
    print(">>> калибровка (num_threads=2)...", flush=True)
    qsvs = qt.calibrate(calib, num_threads=2)
    with open(qsv_cache, "wb") as f:
        pickle.dump(qsvs, f)
    print(f">>> кэш калибровки сохранён: {qsv_cache}", flush=True)
print(">>> материализация параметров...", flush=True)
res = qt.quantize(qsvs)
print(">>> экспорт...", flush=True)
res.export_model(dst)
print("OK:", os.path.getsize(dst), "bytes ->", dst)
