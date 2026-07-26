#!/usr/bin/env python3
"""corr выхода TPU против fp32-эталона с деквантизацией по параметрам модели."""
import sys
import numpy as np
from ai_edge_quantizer.utils import tfl_flatbuffer_utils as fb
model, out_bin, ref_bin = sys.argv[1], sys.argv[2], sys.argv[3]
TYPE = {0: np.float32, 9: np.int8, 7: np.int32, 1: np.float16}
sg = fb.read_model(model).subgraphs[0]
t = sg.tensors[sg.outputs[0]]
dt = TYPE.get(t.type, np.float32)
q = t.quantization
raw = np.fromfile(out_bin, dtype=dt).astype(np.float32)
if dt != np.float32 and q is not None and q.scale is not None and len(q.scale):
    raw = (raw - int(q.zeroPoint[0])) * float(q.scale[0])
b = np.fromfile(ref_bin, dtype=np.float32)
print(f"TPU corr={np.corrcoef(raw, b)[0,1]:.6f} relErr={np.abs(raw-b).mean()/np.abs(b).mean():.4f} std={raw.std():.4f} (эталон {b.std():.4f})")
