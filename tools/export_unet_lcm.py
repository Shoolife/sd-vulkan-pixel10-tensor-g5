#!/usr/bin/env python3
"""Экспорт SD1.5 + LCM-LoRA (слитые) в fp32-tflite с СЫРЫМ timestep — под 4-шаговый LCM.
Чистый float (компилируемо под Tensor), тот же I/O что unet.tflite: (lat[1,4,64,64], ts[1], ctx[1,77,768])."""

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
import litert_torch as L

print(">>> загружаю SD1.5 + LCM-LoRA...", flush=True)
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None
)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.fuse_lora()
unet = pipe.unet.float().eval()


class W(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, lat, ts, ctx):
        return self.m(lat, ts, ctx, return_dict=False)[0]


import os
import sys

B = int(sys.argv[1]) if len(sys.argv) > 1 else 1  # 1 или 2 (batch-CFG)
m = W(unet).eval()
# timestep остаётся [1] — diffusers бродкастит его на батч
args = (torch.randn(B, 4, 64, 64), torch.full((1,), 999.0), torch.randn(B, 77, 768))
out = f"out/unet_lcm{'_b2' if B == 2 else ''}.tflite"
print(f">>> конвертирую в tflite (batch={B})...", flush=True)
L.convert(m, args).export(out)
print("OK:", os.path.getsize(out), "bytes ->", out)
