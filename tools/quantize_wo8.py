#!/usr/bin/env python3
"""Weight-only int8: веса int8, активации fp32. БЕЗ калибровки.

Зачем: static W8A8 требует режима PRESERVE_ALL_TENSORS (экономного в библиотеке нет),
и на self-attention 64x64 промежуточные матрицы 8x4096x4096 (~537 МБ каждая, 5 блоков)
раздувают пик до десятков ГБ -> systemd-oomd убивает cgroup. Здесь калибровки нет вовсе,
пик ~ размер модели. Веса всё равно вчетверо меньше, а TPU memory-bound.

  quantize_wo8.py [src] [dst] [mode=dynamic|weight_only]
"""

import os
import sys

import numpy as np
from ai_edge_quantizer import quantizer, recipe
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as uqt

# тот же обход бага chunked-пути для весов >32 MiB, что и в quantize_int8.py
_orig = uqt.uniform_quantize


def _nochunk(tensor_data, quantization_params, is_blockwise_quant=False):
    big = tensor_data.ndim > 1 and tensor_data.nbytes > 32 * 1024 * 1024
    if is_blockwise_quant or not big:
        return _orig(tensor_data, quantization_params, is_blockwise_quant)
    qp = uqt.fix_quantization_params_rank(tensor_data, quantization_params)
    qtype = uqt.IntType(qp.num_bits, signed=True)
    narrow = qp.symmetric and qp.num_bits >= 8
    ret = np.divide(tensor_data, qp.scale)
    ret = np.add(ret, qp.zero_point, out=ret)
    ret = uqt._round_and_clip_inplace(ret, qtype, narrow)
    return uqt.assign_quantized_type(ret, qtype)


uqt.uniform_quantize = _nochunk

src = sys.argv[1] if len(sys.argv) > 1 else "out/unet_lcm.tflite"
dst = sys.argv[2] if len(sys.argv) > 2 else "out/unet_lcm_wo8.tflite"
mode = sys.argv[3] if len(sys.argv) > 3 else "dynamic"

# dynamic: активации квантуются на лету в int8 -> int8-вычисления (быстрее)
# weight_only: веса деквантуются перед op -> float-вычисления (точнее)
rcp = (
    recipe.dynamic_wi8_afp32() if mode == "dynamic" else recipe.weight_only_wi8_afp32()
)
print(f">>> {mode} int8 без калибровки: {src}", flush=True)
qt = quantizer.Quantizer(src, rcp)
res = qt.quantize()  # calibrate() не нужен
print(">>> экспорт...", flush=True)
res.export_model(dst)
print("OK:", os.path.getsize(dst), "bytes ->", dst)
