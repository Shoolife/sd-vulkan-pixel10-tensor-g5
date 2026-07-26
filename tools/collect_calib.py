#!/usr/bin/env python3
"""Сбор РЕАЛЬНЫХ калибровочных входов UNet с траектории LCM-денойза.

Синтетика N(0,1) даёт негодную калибровку (corr int8 = 0.89): реальные CLIP-эмбеддинги
имеют другой масштаб с выбросами, latent по ходу денойза меняет дисперсию.
Пишет npz со списком (latent, timestep, ctx) для cond и uncond на каждом шаге.

  collect_calib.py [out.npz] [steps=4]
"""

import sys

import numpy as np
import torch
from diffusers import LCMScheduler, StableDiffusionPipeline

out = sys.argv[1] if len(sys.argv) > 1 else "out/calib_real.npz"
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4

PROMPTS = [
    "astronaut riding a horse on mars, photorealistic",
    "a cozy wooden cabin in snowy pine forest at dusk",
    "portrait of an old fisherman, detailed skin, soft light",
    "bowl of ramen on a wooden table, steam, close-up",
]

print(">>> загружаю SD1.5 + LCM-LoRA...", flush=True)
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None
)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.fuse_lora()
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cpu")

rec = []
orig_unet = pipe.unet.forward


def spy(lat, t, encoder_hidden_states=None, **kw):
    ts = np.array([float(t)], dtype=np.float32)
    rec.append(
        (
            lat.detach().cpu().numpy().astype(np.float32),
            ts,
            encoder_hidden_states.detach().cpu().numpy().astype(np.float32),
        )
    )
    return orig_unet(lat, t, encoder_hidden_states=encoder_hidden_states, **kw)


pipe.unet.forward = spy

for i, p in enumerate(PROMPTS):
    print(f">>> прогон {i + 1}/{len(PROMPTS)}: {p[:40]}...", flush=True)
    pipe(
        p,
        num_inference_steps=steps,
        guidance_scale=1.5,
        generator=torch.manual_seed(42 + i),
    )

# батч может быть 2 (cond+uncond склеены) — разрезаем в отдельные B=1 сэмплы
lats, tss, ctxs = [], [], []
for lat, ts, ctx in rec:
    for b in range(lat.shape[0]):
        lats.append(lat[b : b + 1])
        tss.append(ts)
        ctxs.append(ctx[min(b, ctx.shape[0] - 1)][None])

np.savez_compressed(
    out,
    lat=np.concatenate(lats),
    ts=np.concatenate(tss),
    ctx=np.concatenate(ctxs),
)
print(f"OK: {len(lats)} сэмплов -> {out}")
