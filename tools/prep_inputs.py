#!/usr/bin/env python3
"""Готовит входные .bin под dtype/scale конкретной модели (через flatbuffer, без интерпретатора)."""

import os
import sys
import numpy as np
from ai_edge_quantizer.utils import tfl_flatbuffer_utils as fb

model, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
SRC = {
    16384: "npu/ref/in_16384.bin",
    1: "npu/ref/in_1.bin",
    59136: "npu/ref/in_59136.bin",
}
TYPE = {
    0: (np.float32, ""),
    9: (np.int8, "_i8"),
    7: (np.int32, ""),
    1: (np.float16, ""),
}
sg = fb.read_model(model).subgraphs[0]
for i in sg.inputs:
    t = sg.tensors[i]
    cnt = int(np.prod(t.shape))
    dt, suf = TYPE.get(t.type, (np.float32, ""))
    x = np.fromfile(SRC[cnt], dtype=np.float32)
    q = t.quantization
    if dt != np.float32 and q is not None and q.scale is not None and len(q.scale):
        s, z = float(q.scale[0]), int(q.zeroPoint[0])
        info = np.iinfo(dt)
        x = np.clip(np.round(x / s) + z, info.min, info.max)
    x.astype(dt).tofile(f"{out_dir}/in_{cnt}{suf}.bin")
    if suf:
        # раннер выбирает имя по размеру буфера, а тензор из 1 элемента выровнен до 64 байт
        # и не опознаётся как int8 — кладём файл под обоими именами
        x.astype(dt).tofile(f"{out_dir}/in_{cnt}.bin")
    print(f"  in_{cnt}{suf}.bin {dt.__name__}")
