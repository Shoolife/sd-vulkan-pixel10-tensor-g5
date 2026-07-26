#!/usr/bin/env python3
"""Прогон tflite-модели на фиксированных входах npu/ref/in_*.bin (те же, что на устройстве)
и запись выхода в .bin. Опционально — corr/relErr против эталона.

  run_ref.py <model.tflite> <out.bin> [ref.bin]
"""

import sys

import numpy as np
from ai_edge_litert.interpreter import Interpreter

model, out_path = sys.argv[1], sys.argv[2]
ref_path = sys.argv[3] if len(sys.argv) > 3 else None

REF = "npu/ref"
lat = np.fromfile(f"{REF}/in_16384.bin", dtype=np.float32).reshape(1, 4, 64, 64)
ts = np.fromfile(f"{REF}/in_1.bin", dtype=np.float32).reshape(1)
ctx = np.fromfile(f"{REF}/in_59136.bin", dtype=np.float32).reshape(1, 77, 768)
by_count = {16384: lat, 1: ts, 59136: ctx}

print(f">>> {model}", flush=True)
it = Interpreter(model_path=model, num_threads=8)
it.allocate_tensors()
for d in it.get_input_details():
    cnt = int(np.prod(d["shape"]))
    src = by_count[cnt].reshape(d["shape"])
    if d["dtype"] != np.float32:  # int8-вход: квантуем по его же params
        s, z = d["quantization"]
        src = np.clip(np.round(src / s) + z, -128, 127).astype(d["dtype"])
    it.set_tensor(d["index"], src.astype(d["dtype"]))

it.invoke()
od = it.get_output_details()[0]
res = it.get_tensor(od["index"])
if od["dtype"] != np.float32:
    s, z = od["quantization"]
    res = (res.astype(np.float32) - z) * s
res = res.astype(np.float32).reshape(-1)
res.tofile(out_path)
print(f"OK: {res.size} float -> {out_path}  mean={res.mean():.6f} std={res.std():.6f}")

if ref_path:
    ref = np.fromfile(ref_path, dtype=np.float32)
    corr = np.corrcoef(res, ref)[0, 1]
    rel = np.abs(res - ref).mean() / np.abs(ref).mean()
    print(
        f"corr={corr:.6f}  relErr={rel:.4f}  maxAbsDiff={np.abs(res - ref).max():.4f}"
    )
